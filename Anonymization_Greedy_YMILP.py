import time
import numpy as np
from statsmodels.datasets import utils as du

class StaticData:
    records = []
    L = []
    U = []
    arrCat = []
    dictCat = dict()

def GetFeatureAttributes(strFile):
    with open(strFile) as f:
        data = np.recfromtxt(f,delimiter=",",names=True,dtype=None)
    return du.process_recarray_pandas(data)

def GetProblemData(strFile):
    with open(strFile) as f:
        data = np.recfromtxt(f,delimiter=",",names=True,dtype=float)
    return du.process_recarray_pandas(data, endog_idx=0, exog_idx=None,dtype=float)

def GetBoolCategorical(strVarType):
    return {
        "Categorical": True,
        "Numerical": False,
        "b'Categorical'": True,
        "b'Numerical'": False
        }.get(strVarType,False)
def NCP(Lj, Uj, bCategorical, objEntry, arrEntry):
    diff = 0   
    if bCategorical:
        dictCount = dict();
        nEntry = len(objEntry)
        for i in range(nEntry):
            if (objEntry[i] > 0):
                dictCount[i] = 1
            else:
                dictCount[i] = 0
            for objEntry2 in arrEntry:
                if (objEntry2[i] > 0):
                    dictCount[i] = 1  #dictCount[i] is either 0 or 1
            diff += dictCount[i]
        if (diff > 0.0):
            diff = (diff - 1)/(nEntry-1)
    else:
        tmpEntry = list(arrEntry)
        tmpEntry.append(objEntry)
        diff = (max(tmpEntry)-min(tmpEntry))/(Uj-Lj)
    return diff

def GCP(L, U, arrCat, tuple1, arrTuple2):
    nTup = len(tuple1)
    nList = len(arrTuple2)
    gcpsum = 0
    if nList > 0:
        for i in range(nTup):
            arrTuplei = [arrTuple2[j][i] for j in range(nList)]
            gcpsum += NCP(L[i], U[i], arrCat[i],tuple1[i], arrTuplei)
    return gcpsum

def makeTuple(arrEntry,arrCat,dictCat):
    nCat = len(arrCat)
    listEntry = []
    for i in range(nCat):
        if arrCat[i]:
            tmpEntry = [0 for j in range(len(dictCat[i]))]
            tmpEntry[dictCat[i].index(arrEntry[i])] = 1
            listEntry.append(tmpEntry)
        else:
            listEntry.append(arrEntry[i])
    return tuple(listEntry)

def reduceWeight(dictEdges, currWeight, arrVertices, i):
    strPrefix = "S"+str(i)+"_T"
    vert = i
    arrTuple = [StaticData.records[int(stri)] for stri in arrVertices]
    if (arrTuple):
        for strEdge in dictEdges.keys():
            if strPrefix in strEdge:
                lenEdge = len(strEdge)
                lenPrefix = len(strPrefix)
                vert = int(strEdge[lenPrefix:lenEdge])
                dictEdges[strEdge] = GCP(StaticData.L, StaticData.U, StaticData.arrCat, StaticData.records[vert], arrTuple) - currWeight

def transformResult(arrVertices):
    strRes = ""
    arrEntry = [StaticData.records[int(stri)] for stri in arrVertices]
    nVert = len(arrVertices)
    nCat = len(StaticData.arrCat)
    for i in range(nCat):
        if (StaticData.arrCat[i]):      
            for j in range(nVert):
                nTup = len(arrEntry[j][i])
                for l in range(nTup):
                    if arrEntry[j][i][l] > 0:
                        strRes += str(StaticData.dictCat[i][l]) + "|"
            strRes += ","
        else:
            tmp = [arrEntry[j][i] for j in range(nVert)]
            mx = max(tmp)
            mn = min(tmp)
            strRes += "["+str(mn)+" - "+str(mx)+"],"
    return strRes       

                
            

def AnonymizeByGreedyYMILP(strFile, strFileAtt, K, strFileOut):
    start_time = time.time()
    #clear data from previous run
    StaticData.records = []
    StaticData.L = []
    StaticData.U = []
    StaticData.arrCat = []
    StaticData.dictCat = dict()

    objData = GetProblemData(strFile)
    arrData = objData.data
    arrHeaders = objData.names    
    n = len(arrData) #number of records
    nHeaders = len(arrHeaders)
    arrCard = [len(set(arrData[header])) for header in arrHeaders]

    matSorted = [[i] for i in range(n)]
    for i in range(n):
        for header in arrHeaders:
            matSorted[i].append(float(arrData[header][i]))
    dtype = [('index', 'int')]
    for header in arrHeaders:
        dtype.append((header,'float'))
    for i in range(n):
        matSorted[i] = tuple(matSorted[i])
    matSorted = np.array(matSorted,dtype=dtype)

    attData = GetFeatureAttributes(strFileAtt).data
    arrVarType = [str(attData[header][0]) for header in arrHeaders]
    L = [min(arrData[header]) for header in arrHeaders]
    U = [max(arrData[header]) for header in arrHeaders]
    for j in range(nHeaders):
        if (L[j] == U[j]):
            L[j] = float(attData[arrHeaders[j]][1])
            U[j] = float(attData[arrHeaders[j]][2])

    StaticData.arrCat = [GetBoolCategorical(strVarType) for strVarType in arrVarType]
    StaticData.L = [L[j] for j in range(nHeaders)]
    StaticData.U = [U[j] for j in range(nHeaders)]


    for i in range(nHeaders):
        if StaticData.arrCat[i]:
            setEntry = set(matSorted[arrHeaders[i]])
            if len(setEntry) < 2:
                setEntry.add(StaticData.L[i])
                setEntry.add(StaticData.U[i])
            StaticData.dictCat[i] = list(setEntry)
            StaticData.dictCat[i].sort()

    CurrW = dict()
    A = dict()

    for i in range(n):
        A[i] = [i]
        t1 = makeTuple([float(matSorted[header][i]) for header in arrHeaders],StaticData.arrCat,StaticData.dictCat)
        StaticData.records.append(t1)
        CurrW[i] = 0.0
    #for i in S:
    #    for j in T:
    #        E["S"+str(i)+"_T"+str(j)] = GCP(StaticData.L,StaticData.U,StaticData.arrCat,StaticData.records[j],[StaticData.records[i]])
    arrVar = [np.var(arrData[arrHeaders[i]]) for i in range(nHeaders)]
    arrOrder = np.argsort(arrVar)#arrCard)
    headerSorted = [arrHeaders[j] for j in arrOrder]
    matSorted = np.sort(matSorted,order=headerSorted)
    nSets = int(n/K)
    nRemain = n - nSets*K
    CurrE= dict()
    CurrT = dict()
    S = [matSorted['index'][i] for i in range(n)]
    T = S[:]
    tmpNotAssigned =[]
    tmpRemain = []
    lenT = len(T)
    
    f = open(strFileOut, 'a')
    for i in S:
        if lenT <= nRemain:
            tmpRemain = T[:]
            break
        elif i in T:
            T.remove(i)
            lenT = lenT - 1
            prevW = 0.0
            currASet = [StaticData.records[i]]
            for k in range(K-1):
                tmpEdgeW = dict()
                for j in T:
                    tmpEdgeW[j] = GCP(StaticData.L,StaticData.U,StaticData.arrCat,StaticData.records[j],currASet)
                tmpj_min = min(tmpEdgeW, key=tmpEdgeW.get)
                prevW = tmpEdgeW[tmpj_min]
                T.remove(tmpj_min)
                lenT = lenT - 1
                A[i].append(tmpj_min)
                currASet.append(StaticData.records[tmpj_min])
            CurrW[i] = prevW
        #else:
        #    tmpNotAssigned.append(i)

    tmpKeys = list(A.keys())
    
    for i in tmpKeys:
        if len(A[i]) < K:
            del A[i]
            tmpNotAssigned.append(i)

    tmpKeys = list(A.keys())
    if T:       
        for j in tmpRemain:
            tmpEdgeW = dict()
            for i in tmpKeys:
                currASet = [StaticData.records[tmpi] for tmpi in A[i]]
                tmpEdgeW[i] = GCP(StaticData.L,StaticData.U,StaticData.arrCat,StaticData.records[j],currASet)
            tmpj_min = min(tmpEdgeW, key=tmpEdgeW.get)
            A[tmpj_min].append(j)
            A[j] = A[tmpj_min]
            for newj in A[tmpj_min]:
                CurrW[newj] = tmpEdgeW[tmpj_min] #update weight of current set
            T.remove(j)
    tmpKeys = list(A.keys())
    for j in tmpNotAssigned:
        for i in tmpKeys:
            if j in A[i]:
                A[j] = A[i]
                CurrW[j] = CurrW[i]


    
    #print("iteration ", k)
    #f.write("iteration "+str(k)+"\n")

    #for i in S:
    #    strRes = ""
    #    print(i, [StaticData.records[j] for j in A[i]])
    #    tmpStr = "["
    #    for a in A[i]:
    #        tmpStr += str(StaticData.records[a])+" "
    #    tmpStr += "]"
    #    f.write(str(i)+":"+tmpStr+"\n")
    for i in S:
        print(i,A[i])
        tmpStr = "["
        for a in A[i]:
            tmpStr += str(a) + " "
        tmpStr += "]"
        f.write(str(i)+":"+tmpStr+"\n")
    for i in S:
        print(i, "= "+transformResult(A[i]))
        f.write(str(i)+"="+transformResult(A[i])+"\n")
    for i in S:
        print(i, CurrW[i])
        f.write(str(i)+"=["+str(CurrW[i])+"]\n")
    print("Final weight: ", sum(CurrW.values()))
    elapsed_time = time.time() - start_time
    f.write("Final solution found in: "+str(elapsed_time)+"\n")
    f.write("Final weight: "+str(sum(CurrW.values()))+"\n\n")
    f.close()





