import math
import time
import numpy as np
import pandas as pd
from statsmodels.datasets import utils as du
from gurobipy import *


def GetFeatureAttributes(strFile):
    with open(strFile) as f:
        data = np.recfromtxt(f,delimiter=",",names=True,dtype=None)
    return du.process_recarray_pandas(data)

def GetProblemData(strFile):
    with open(strFile) as f:
        data = np.recfromtxt(f,delimiter=",",names=True,dtype=float)
    return du.process_recarray_pandas(data, endog_idx=0, exog_idx=None,dtype=float)

def GetGurobiVarType(strVarType):
    return {
        "Continuous": GRB.CONTINUOUS,
        "b'Continuous'": GRB.CONTINUOUS,
        "Integer": GRB.INTEGER,
        "b'Integer'": GRB.INTEGER,
        "Binary": GRB.BINARY,
        "b'Binary'": GRB.BINARY
        }.get(strVarType,GRB.CONTINUOUS)

def GetInitialSolutionMain(matData, matHeaders, K, w):
    nRecords = len(matData)
    matSorted = [[i] for i in range(nRecords)]
    nHeaders = len(matHeaders)
    for i in range(nRecords):
        for header in matHeaders:
            matSorted[i].append(float(matData[header][i]))
    
    #determine sort order according to variance
    arrVar = [np.var(matData[matHeaders[i]])/(w[i]*w[i]) for i in range(nHeaders)]
    arrVar = np.argsort(arrVar)
    dtype = [('index', 'int')]
    for header in matHeaders:
        dtype.append((header,'float'))
    for i in range(nRecords):
        matSorted[i] = tuple(matSorted[i])
    matSorted = np.array(matSorted,dtype=dtype)
    headerSorted = [matHeaders[var_i] for var_i in arrVar]
    #sort the matrix
    matSorted = np.sort(matSorted,order=headerSorted)

    #k consecutive rows are made equivalent
    nSets = int(nRecords / K)
    all_sets = []
    for i in range(nSets-1):
        i_set = []
        for j in range(K):
            i_set.append(matSorted['index'][i*K+j])
        all_sets.append(i_set)
    #add the last set, may have size >= K
    i_set = []
    for j in range(nRecords-(nSets-1)*K):
        i_set.append(matSorted['index'][(nSets-1)*K+j])
    all_sets.append(i_set)

    #construct initial feasible solution   
    init_dict_v = dict()
    init_dict_y = dict()
    init_dict_z = dict()
    init_vec_ysort = []
    init_vec_zsort = []
    for perm_set in all_sets:
        mn = [min([matData[header][p] for p in perm_set]) for header in matHeaders]
        mx = [max([matData[header][p] for p in perm_set]) for header in matHeaders]
        for j in range(len(perm_set)):
            s1 = perm_set[j]
            for i in range(len(perm_set)-j-1):
                s2 = perm_set[i+j+1]
                if (s1 < s2):
                    init_dict_v["v"+str(s1)+"_"+str(s2)]=1
                else:
                    init_dict_v["v"+str(s2)+"_"+str(s1)]=1
            for l in range(nHeaders):
                init_dict_y["y"+str(l)+"_"+str(s1)] = mn[l]
                init_dict_z["z"+str(l)+"_"+str(s1)] = mx[l]
        init_vec_ysort.append(mn)
        init_vec_zsort.append(mx)


    return (init_dict_v, init_dict_y, init_dict_z, init_vec_ysort, init_vec_zsort, all_sets)
	
#------------------------------------------------------------------------------------solve subproblem
def AnonymizeBySubLpGurobi(arrData, arrHeaders,arrVarType, v_dict, y_dict, z_dict, K, L, U, w):
    start_time = time.time()

    #--------------------- constants -------------------------
    l = len(arrHeaders) #number of attributes
    n = len(arrData) #number of records
    p = int(n*(n-1)/2)#int(nCr(n,2))
    #constant bounds on data
    mx = [[max(arrData[header][i],arrData[header][j]) for i in range(n) for j in range(n) if i < j] for header in arrHeaders]
    mn = [[min(arrData[header][i],arrData[header][j]) for i in range(n) for j in range(n) if i < j] for header in arrHeaders]

    #--------------------- model -----------------------------
    m = Model("K-Anonymization")
    #lowerbound variables yij, uppperbound variables zij
    y_name = [['y'+str(j)+"_"+str(i) for i in range(n)] for j in range(l)]
    z_name = [['z'+str(j)+"_"+str(i) for i in range(n)] for j in range(l)]
    y = [[m.addVar(vtype=GetGurobiVarType(arrVarType[j]), name='y'+str(j)+"_"+str(i),lb=L[j],ub=arrData[arrHeaders[j]][i]) for i in range(n)] for j in range(l)]
    z = [[m.addVar(vtype=GetGurobiVarType(arrVarType[j]), name='z'+str(j)+"_"+str(i),lb=arrData[arrHeaders[j]][i],ub=U[j]) for i in range(n)] for j in range(l)]
    #indicator variables for row i,j equivalence
    v_name = ['v'+str(i)+'_'+str(j) for i in range(n) for j in range(n) if i < j]
    v = [m.addVar(vtype=GRB.BINARY,name='v'+str(i)+'_'+str(j)) for i in range(n) for j in range(n) if i < j]
    
    #--------------------- constraint matrices ----------------
    #b = [0,n-1,n-1+(n-2),n-1+(n-2)+(n-3)...]
    c = n - 1
    b = [0 for j in range(n)]
    b[0] = 0
    for j in range(n-1):
        b[j+1] = b[j] + c
        c = c - 1
    #matrix for sum equivalent rows
    A = [[0 for j in range(p)] for i in range(n)]
    for i in range(n):
        for j in range(p):
            if (v_name[j].startswith('v'+str(i)+'_') or v_name[j].endswith('_'+str(i))):
                A[i][j] = 1
    
    #for i in range(i):
    #    print(A[i])
    #matrix [[1 -1 0 ... 0], [0 1 -1 0 ... 0]]...] for checking if bounds(i) = bounds(k)
    B = [[0 for j in range(n)] for i in range(p)]
    for a in range(n-1):
        for j in range(n): 
            for i in range(b[a],b[a+1]):      
                if (j == a):
                    B[i][j] = 1
                elif (j == i - b[a]+1+a):
                    B[i][j] = -1

    #objective function
    objf = LinExpr()
    for j in range(l):
        for i in range(n):
            objf += w[j]*(z[j][i]-y[j][i])/(U[j]-L[j])
    m.setObjective(objf, GRB.MINIMIZE)

    #constraints: checking equivlance
    for i in range(p):
        for j in range(l):
            const_equv_yl = LinExpr()
            const_equv_zl = LinExpr()
            const_equv_yr = LinExpr()
            const_equv_zr = LinExpr()
            for r in range(n):
                const_equv_yl += B[i][r]*y[j][r]
                const_equv_zl += B[i][r]*z[j][r]
                const_equv_yr += B[i][r]*y[j][r]
                const_equv_zr += B[i][r]*z[j][r]
            m.addConstr(const_equv_yl<=(mx[j][i]-L[j])*(1-v[i]),"equivance for y(" + str(j)+") y(k) direction <= at row "+str(i))
            m.addConstr(const_equv_zl<=(U[j]-mn[j][i])*(1-v[i]),"equivance for z(" + str(j)+") z(k) direction <= at row "+str(i))

            m.addConstr(const_equv_yr>=(L[j]-mx[j][i])*(1-v[i]),"equivance for y(" + str(j)+") y(k) direction >= at row "+str(i))
            m.addConstr(const_equv_zr>=(mn[j][i]-U[j])*(1-v[i]),"equivance for z(" + str(j)+") z(k) direction >= at row "+str(i))
    #k-anonymity constraint
    for i in range(n):
        const_k_anon = LinExpr()
        for j in range(p):
            const_k_anon +=A[i][j]*v[j]
        m.addConstr(const_k_anon == K-1, str(K)+"-anonymity for "+str(i))

    #solution
    m.write("k-Anonymization_sub.lp")
    #providing initial solution
    for v_var in v:
        if v_var.varName in v_dict:
            v_var.start = v_dict[v_var.varName]
        else:
            v_var.start = 0
    for y_row in y:
        for y_var in y_row:
            y_var.start = y_dict[y_var.varName]
    for z_row in z:
        for z_var in z_row:
            z_var.start = z_dict[z_var.varName]
    m.Params.MIPFocus = 2
    m.optimize()

    #print output
    print("Solver finished.")
    m.printQuality

    resMat = dict([val.varName, val.x] for val in m.getVars())
    return resMat

def GetEquivalentRecords(iRecord, arrIndices, dictRes, K):
    arrEqv = []
    arrNonEqv = []
    for s in arrIndices:
        if (dictRes["v"+str(s)+"_"+str(iRecord)] > 0):
            arrEqv.append(s)
        else:
            arrNonEqv.append(s)
    setAdd = set()
    for iEq in arrEqv:
        for iNon in arrNonEqv:
            if iEq < iNon:
                strEqv = "v"+str(iEq)+"_"+str(iNon)
            else:
                strEqv = "v"+str(iNon)+"_"+str(iEq)
            if (dictRes[strEqv] > 0):
                setAdd.add(iNon)
    if len(setAdd) < K:
        arrEqv.extend(setAdd)
    return arrEqv


#-------------------------------------------------------------------------------------------------------------------- Main problem
def AnonymizeByLpGurobiMain(strFile,strFileAtt,K,S, strFileOut):
    start_time = time.time()
    objData = GetProblemData(strFile)
    arrData = objData.data
    arrHeaders = objData.names

    #matrix of records
    attData = GetFeatureAttributes(strFileAtt).data
    #variable types
    arrVarType = [str(attData[header][0]) for header in arrHeaders]    
	#weights
    w = [float(attData[header][3]) for header in arrHeaders]
	#providing initial solution
    (v_dict, y_dict, z_dict, vec_ysort, vec_zsort, S_sets) = GetInitialSolutionMain(arrData, arrHeaders, K, w)
    f = open(strFileOut, 'a')
    f.write("Initial solution found in: " + str(time.time()-start_time) + "\n")
    #print("Initial solution found in: ", time.time()-start_time)
    #constants
    L = [min(arrData[header]) for header in arrHeaders]
    U = [max(arrData[header]) for header in arrHeaders]
    
    n = len(arrData)
    nHeaders = len(arrHeaders)
    for j in range(nHeaders ):
        if (L[j] == U[j]):
            L[j] = float(attData[arrHeaders[j]][1])
            U[j] = float(attData[arrHeaders[j]][2])
    r = 0
    lenSets  = len(S_sets)
    nSets = int(lenSets/S)
    if (lenSets > nSets*S):
        nSets = nSets + 1

    #print initial solution
    objVal = 0
    f.write("Initial solution by sorting: \n")
    #print("Initial solution by sorting: ")
    for j in range(n):
        strRes = ""
        for i in range(nHeaders):
            objVal += (z_dict["z"+str(i)+"_"+str(j)]-y_dict["y"+str(i)+"_"+str(j)])/(U[i]-L[i])
            strRes += "["+str(y_dict["y"+str(i)+"_"+str(j)])+" - "+str(z_dict["z"+str(i)+"_"+str(j)])+"],"
        f.write(strRes+"\n")

    f.write("Objective value by sorting: " + str(objVal) +"\n")

    #final solution
    v_dict_final = dict()
    y_dict_final = dict()
    z_dict_final = dict()

    #overlap records
    overLapData=[]
    tmp_dict = dict()
    r = 0        
    v_dict_sub = dict()
    y_dict_sub = dict()
    z_dict_sub = dict()
    for p in range(nSets):
        print("p = ", p)
        subData = []
        l = r       
        subData.extend(overLapData)
        #create initial solution for subproblem
        nS = min(S, len(S_sets)-p*S)
        for i in range(nS): 
            a_set = S_sets[p*S+i]
            n_a = len(a_set)
            for j in range(n_a):
                tmp_dict[l] = a_set[j]
                subData.append(arrData.values[a_set[j]])              
                for h in range(nHeaders):
                    y_dict_sub["y"+str(h)+"_"+str(l)] = vec_ysort[p*S+i][h]
                    z_dict_sub["z"+str(h)+"_"+str(l)] = vec_zsort[p*S+i][h]
                l = l + 1

            for u in range(n_a):
                for q in range(n_a):
                    if (u < q):
                        v_dict_sub["v"+str(l-n_a+u)+"_"+str(l-n_a+q)] = 1
        #solve subproblem        
        subDataDS = pd.DataFrame(subData, columns=arrHeaders)
        resMat = AnonymizeBySubLpGurobi(subDataDS, arrHeaders,arrVarType, v_dict_sub, y_dict_sub, z_dict_sub, K,L,U, w)
        #clear initial solution to subproblem
        if p < nSets-1:
            y_dict_sub = dict()
            z_dict_sub = dict()
            v_dict_sub = dict()
            endrecords = []
            #record solution and overlapping records
            endrecords = [l-K+t for t in range(K)]

            #for s in range(l-K):
            #    for t in range(l-K,l):
            #        if (s not in endrecords) and (resMat["v"+str(s)+"_"+str(t)] > 0):
            #            endrecords.append(s)
            tmprecords = []
            tmpIndices = [s for s in range(l-K)]
            for t in endrecords:
                tmprecords.extend(GetEquivalentRecords(t, tmpIndices, resMat, K))
            endrecords.extend(tmprecords)
            endrecords = list(set(endrecords))
            endrecords.sort()
            r = len(endrecords)
            tmp1_dict = dict()
            overLapData = []
            for s in range(len(endrecords)):
                for t in range(len(endrecords)):
                    if (s < t):
                        tmp_s = endrecords[s]
                        tmp_t = endrecords[t]
                        if (tmp_s < tmp_t):
                            v_dict_sub["v"+str(s)+"_"+str(t)] = resMat["v"+str(tmp_s)+"_"+str(tmp_t)]
                        else:
                            v_dict_sub["v"+str(s)+"_"+str(t)] = resMat["v"+str(tmp_t)+"_"+str(tmp_s)]
                for h in range(nHeaders):
                     y_dict_sub["y"+str(h)+"_"+str(s)] = resMat["y"+str(h)+"_"+str(endrecords[s])]
                     z_dict_sub["z"+str(h)+"_"+str(s)] = resMat["z"+str(h)+"_"+str(endrecords[s])]
                overLapData.append(arrData.values[tmp_dict[endrecords[s]]])
                tmp1_dict[s] = tmp_dict[endrecords[s]]        
    
        #if p < nSets-1:
            for t in range(l):
                if t not in endrecords:
                    for h in range(nHeaders):
                        y_dict_final["y"+str(h)+"_"+str(tmp_dict[t])] = resMat["y"+str(h)+"_"+str(t)]
                        z_dict_final["z"+str(h)+"_"+str(tmp_dict[t])] = resMat["z"+str(h)+"_"+str(t)]
            for s in range(l):
                for t in range(l):
                    if s < t:
                        tmp_s = tmp_dict[s]
                        tmp_t = tmp_dict[t]
                        if not(t in endrecords and s in endrecords):
                            if (tmp_s < tmp_t):
                                v_dict_final["v"+str(tmp_s)+"_"+str(tmp_t)] = resMat["v"+str(s)+"_"+str(t)]
                            else:
                                v_dict_final["v"+str(tmp_t)+"_"+str(tmp_s)] = resMat["v"+str(s)+"_"+str(t)]
            tmp_dict = tmp1_dict
        else:
            for t in range(l):
                for h in range(nHeaders):
                    y_dict_final["y"+str(h)+"_"+str(tmp_dict[t])] = resMat["y"+str(h)+"_"+str(t)]
                    z_dict_final["z"+str(h)+"_"+str(tmp_dict[t])] = resMat["z"+str(h)+"_"+str(t)]
            for s in range(l):
                for t in range(l):
                    if s < t:
                        tmp_s = tmp_dict[s]
                        tmp_t = tmp_dict[t]
                        if (tmp_s < tmp_t):
                            v_dict_final["v"+str(tmp_s)+"_"+str(tmp_t)] = resMat["v"+str(s)+"_"+str(t)]
                        else:
                            v_dict_final["v"+str(tmp_t)+"_"+str(tmp_s)] = resMat["v"+str(s)+"_"+str(t)]

    #print final solution
    objVal = 0
    f.write("Final solution found in: "+str(time.time()-start_time)+"\n")
    f.write("Final solution: \n")
    for j in range(n):
        strRes = ""
        for i in range(nHeaders):
            objVal += w[i]*(z_dict_final["z"+str(i)+"_"+str(j)]-y_dict_final["y"+str(i)+"_"+str(j)])/(U[i]-L[i])
            strRes += "["+format(y_dict_final["y"+str(i)+"_"+str(j)],'.1f')+" - "+format(z_dict_final["z"+str(i)+"_"+str(j)],'.1f')+"],"
        f.write(strRes+"\n")
    f.write("Objective value by algo with initial solution: " +str(objVal)+"\n\n")
    f.close()
                



