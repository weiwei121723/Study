import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
from pythonStudy.mlchat1.KNN import *
from analysis.structA.Byaes import *
from analysis.structA.SVM import *
cardFilePath = 'F:\数据\精准资助\精准资助\\train\card_train.txt'
subsidyFilePath = 'F:\数据\精准资助\精准资助\\train\subsidy_train.txt'
borrowFilePath = 'F:\数据\精准资助\精准资助\\train\\borrow_train.txt'
libraryFilePath = 'F:\数据\精准资助\精准资助\\train\library_train.txt'
scoreFilePath = 'F:\数据\精准资助\精准资助\\train\score_train.txt'
test_cardFilePath = 'F:\数据\精准资助\精准资助\\test\card_test.txt'
test_subsidyFilePath = 'F:\数据\精准资助\精准资助\\test\studentID_test.txt'
test_borrowFilePath = 'F:\数据\精准资助\精准资助\\test\\borrow_test.txt'
test_libraryFilePath = 'F:\数据\精准资助\精准资助\\test\\library_test.txt'
test_scoreFilePath = 'F:\数据\精准资助\精准资助\\test\score_test.txt'
def cardStatis(filePath):
    fr = open(filePath,encoding='utf-8');
    cardCount = {};
    for line in fr.readlines():
        line = line.strip()#
        line = line.replace('\'','')
        line = line.replace('\"','')
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in cardCount:
            cardCount[keyTemp] = {};
            cardCount[keyTemp]['card']=float(0);
        temp = float(listFromLine[len(listFromLine) - 2]);
        cardCount[keyTemp]['card'] +=temp;

    return cardCount;
def borrowStatis(filePath):
    fr = open(filePath,encoding='utf-8');
    borrowCount = {};
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in borrowCount:
            borrowCount[keyTemp] = {};
            borrowCount[keyTemp]['borrow']=int(0);
        borrowCount[keyTemp]['borrow'] +=1;
    return borrowCount;
def libraryStatis(filePath):
    fr = open(filePath,encoding='utf-8');
    libraryCount = {};
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in libraryCount:
            libraryCount[keyTemp] = {};
            libraryCount[keyTemp]['library']=int(0);
        libraryCount[keyTemp]['library'] +=1;
    return libraryCount;
def scoreStatis(filePath):
    fr = open(filePath,encoding='utf-8');
    scoreCount = {};
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in scoreCount:
            scoreCount[keyTemp] = {};
            scoreCount[keyTemp]['score']=int(listFromLine[1]);
    return scoreCount;
def subsidyStatis(filePath1,filePath2,filePath3,filePath4,filePath5):
    cardCount = cardStatis(filePath2);
    borrowCount = borrowStatis(filePath3)
    libraryCount = libraryStatis(filePath4)
    #scoreCount = scoreStatis(filePath5)
    fr = open(filePath1,encoding='utf-8');
    subsidyCount = {};
    for line in fr.readlines():
        line = line.strip()#去掉所有回车
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in subsidyCount:
            subsidyCount[keyTemp] = {};
            subsidyCount[keyTemp]['subsidy']=str(0);
        if len(listFromLine)==1:
            subsidyCount[keyTemp]['subsidy']=str(0);
        else:
            temp = int(listFromLine[len(listFromLine) - 1]);
            subsidyCount[keyTemp]['subsidy'] =str(temp);
        if keyTemp not in cardCount:
            subsidyCount[keyTemp]['card'] =float(0);
        else:
            subsidyCount[keyTemp]['card'] =cardCount[keyTemp]['card'];
        if keyTemp not in borrowCount:
            subsidyCount[keyTemp]['borrow'] =int(0);
        else:
            subsidyCount[keyTemp]['borrow'] =borrowCount[keyTemp]['borrow'];
        if keyTemp not in libraryCount:
            subsidyCount[keyTemp]['library'] =int(0);
        else:
            subsidyCount[keyTemp]['library'] =libraryCount[keyTemp]['library'];
    return  subsidyCount
def trainDataSet(train_Data):
    matrix = [[0 for i in range(2)] for i in range(len(train_Data))]
    index = 0;
    for k in train_Data:
        matrix[index][0] = float(train_Data[k]['card']);
        matrix[index][1] = str(train_Data[k]['subsidy']);
        index +=1;
    return  matrix
def runKNN():
    train_Data = subsidyStatis(subsidyFilePath,cardFilePath);
    trainDataSetList = trainDataSet(train_Data);#训练数据
    group = array(trainDataSetList)[:,0:1]
    labels = array(trainDataSetList)[:,1:2]
    labels2=[];
    for i in labels:
        labels2.append(str(i[0]));
    test_Data = subsidyStatis(test_subsidyFilePath,test_cardFilePath);
    fw = open('F:\数据\精准资助\精准资助\\train\\result.csv','w');
    fw.write("studentid,subsidy\n")
    for k in sorted(test_Data.keys()):
        testList = [];
        testList.append(test_Data[k]['card'])
        resTemp = classify0(testList, group , labels2 , 2);
        strTemp = str(k)+","+str(resTemp)+'\n'
        fw.write(strTemp)
def dataClean():
    subsidyCount = subsidyStatis(test_subsidyFilePath,test_cardFilePath,test_borrowFilePath,test_libraryFilePath,test_scoreFilePath)
    fw = open('F:\数据\精准资助\精准资助\\test\\testDataSet.txt','w');
    fw.write("studentid,card,borrow,library,subsidy\n")
    for k in sorted(subsidyCount.keys()):
        strtemp = str(k)+","+str(subsidyCount[k]['card'])+","+str(subsidyCount[k]['borrow'])+","+str(subsidyCount[k]['library'])+","+str(subsidyCount[k]['subsidy'])
        fw.write(strtemp+'\n')
def readData(filePath):
    fr = open(filePath,encoding='utf-8');
    lenTemp = len(fr.readlines()) - 1;
    #matrixT = [[0 for i in range(4)] for i in range(lenTemp)]
    matrixT = list()
    labelT = list();
    fr = open(filePath,encoding='utf-8');
    indexT = -1;
    for line in fr.readlines():
        if indexT==-1:
            indexT +=1;
            continue;#条过第一行
        line = line.strip()#去掉所有回车
        dataLine = list();
        listFromLine = line.split(',')
        dataLine.append(float(listFromLine[1]))
        dataLine.append(int(listFromLine[2]))
        dataLine.append(int(listFromLine[3]))
        matrixT.append(dataLine)

        labelT.append(int(listFromLine[0]))
    return matrixT,labelT
def readDataTest(filePath):
    fr = open(filePath,encoding='utf-8');
    lenTemp = len(fr.readlines()) - 1;
    #matrixT = [[0 for i in range(4)] for i in range(lenTemp)]
    matrixT = list()
    labelT = list();
    fr = open(filePath,encoding='utf-8');
    indexT = -1;
    for line in fr.readlines():
        if indexT==-1:
            indexT +=1;
            continue;#条过第一行
        line = line.strip()#去掉所有回车
        dataLine = list();
        listFromLine = line.split(',')
        labelT.append(int(listFromLine[0]))
        dataLine.append(float(listFromLine[1]))
        dataLine.append(int(listFromLine[2]))
        dataLine.append(int(listFromLine[3]))
        matrixT.append(dataLine)
        if len(listFromLine)==5:
            labelT.append(int(listFromLine[4]))
    return matrixT,labelT
def readDataNum(matrixT,labelT,espiteNum):
    matrix_return = list();
    labelT_return = list();
    labelT_next = list();
    indexT = -1
    for row in labelT:
        indexT+=1;
        if row==0:
            continue;
        if row==espiteNum:
            labelT_return.append(1) ;

        else :
            labelT_return.append(-1) ;
            labelT_next.append(row)
        matrix_return.append(matrixT[indexT])
    return matrix_return,labelT_return,labelT_next

def runSVM():
    matrixT, labelT = readData('F:\数据\精准资助\精准资助\\train\\trainDataSet.txt');
    test_matrixT, test_labelT = readData('F:\数据\精准资助\精准资助\\test\\testDataSet.txt');
    labelT_0 = [];
    for i in labelT:
        if i == 0:
            labelT_0.append(-1)
        else:
            labelT_0.append(1);
    b, alphas = smoP(matrixT, labelT_0, 0.6, 0.0001, 40)
    ws = calcWs(alphas, matrixT, labelT_0)
    fw = open('F:\数据\精准资助\精准资助\\train\svmResult.txt','w');
    fw.write(str(ws[0][0])+","+str(ws[1][0])+","+str(ws[2][0])+","+str(b)+'\n')
    print("ws = \n", ws)
    print("b = \n", b)
    matrixT, labelT,labelT_next= readDataNum(matrixT, labelT,1000)
    b, alphas = smoP(matrixT, labelT, 0.6, 0.0001, 40)
    ws = calcWs(alphas, matrixT, labelT)
    print("ws = \n", ws)
    print("b = \n", b)
    fw.write(str(ws[0][0])+","+str(ws[1][0])+","+str(ws[2][0])+","+str(b)+'\n')
    matrixT, labelT,labelT_next = readDataNum(matrixT, labelT_next,1500)
    b, alphas = smoP(matrixT, labelT, 0.6, 0.0001, 40)
    ws = calcWs(alphas, matrixT, labelT)
    print("ws = \n", ws)
    print("b = \n", b)
    fw.write(str(ws[0][0])+","+str(ws[1][0])+","+str(ws[2][0])+","+str(b)+'\n')
    print()
def SVMR(testVector):
    svmrWS = [
        [0.0125548886455,-2.26005920595e-06,-3.35767163981e-05,0],
        [-0.0187018051255,-0.00835825289196,-0.00454962034393,-1.0934485],
        [0.013194698584,-0.000483308963715,0.000800301051988,107.78348509]
    ]
    t1 = svmrWS[0][0] * testVector[0] + svmrWS[0][1] * testVector[1] + svmrWS[0][2] * testVector[2] +svmrWS[0][3]
    t2 = svmrWS[1][0] * testVector[0] + svmrWS[1][1] * testVector[1] + svmrWS[1][2] * testVector[2] +svmrWS[1][3]
    t3 = svmrWS[2][0] * testVector[0] + svmrWS[2][1] * testVector[1] + svmrWS[2][2] * testVector[2] +svmrWS[2][3]
    if svmrWS[0][0] * testVector[0] + svmrWS[0][1] * testVector[1] + svmrWS[0][2] * testVector[2] +svmrWS[0][3] >=50:
        return 0;
    elif svmrWS[1][0] * testVector[0] + svmrWS[1][1] * testVector[1] + svmrWS[1][2] * testVector[2] +svmrWS[1][3] >=-70:
        return 1000;
    elif svmrWS[2][0] * testVector[0] + svmrWS[2][1] * testVector[1] + svmrWS[2][2] * testVector[2] +svmrWS[2][3] >=200:
        return 1500;
    else :
        return 2000;

def SVMtestResult():
    test_matrixT, test_labelT = readData('F:\数据\精准资助\精准资助\\test\\testDataSet.txt');
    fw = open('F:\数据\精准资助\精准资助\\train\svmResult.csv','w');
    fw.write("studentid,subsidy\n")
    indexT = 0;
    for i in test_labelT:
        subsidy = SVMR(test_matrixT[indexT])
        fw.write(str(i)+","+str(subsidy)+'\n')
        indexT+=1;

SVMtestResult()