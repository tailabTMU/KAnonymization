from Anonymization_MainAlgo import AnonymizeByLpGurobiMain


strFile = '../../example/person20_4attr.csv'
strFileAtt = '../../example/person_4att_sup.txt'
strFileOut = '../../example/person20_4attr_k3s3.csv'
AnonymizeByLpGurobiMain(strFile,strFileAtt,3,3, strFileOut)