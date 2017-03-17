import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
from pythonStudy.mlchat1.KNN import *
from analysis.structA.Byaes import *
from analysis.structA.SVM import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree import export_graphviz
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import svm
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
def tool(typeConsume):
    valuseS = ["食堂","开水","淋浴","洗衣机","文艺中心"]
    indexT = 0;
    for varS in valuseS:
        if varS == typeConsume:
            break;
        indexT+=1;
    return  indexT
def tool2(typeSubsidy):
    valuseS = [0,1000,1500,2000]
    indexT = 0;
    for varS in valuseS:
        if varS == typeSubsidy:
            break;
        indexT+=1;
    return  indexT
def junzhi(labelArray,numArray):
    resultT = [[0,0],[0,0],[0,0],[0,0]];
    indexT = 0;
    for i in labelArray:
        resultT[i-1][0] += numArray[indexT][0]
        resultT[i-1][1]+=1;
        indexT+=1;
    returnT = []
    returnT.append(float(resultT[0][0]) /resultT[0][1] )
    returnT.append(float(resultT[1][0]) /resultT[1][1] )
    returnT.append(float(resultT[2][0]) /resultT[2][1] )
    returnT.append(float(resultT[3][0]) /resultT[3][1] )
    return returnT
def junzhi2(numArray):
    indexT = 0;
    returTTTT = 0.0;
    for i in numArray:
        returTTTT += i
        indexT+=1;
    return returTTTT / indexT
'''card'''
def readTest():
    fr = open(subsidyFilePath,encoding='utf-8');
    subidyCount = {};
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in subidyCount:
            subidyCount[keyTemp] = {};
        subidyCount[keyTemp]['subidy'] =int(listFromLine[1]);

    fr = open(cardFilePath,encoding='utf-8');
    cardCount = {};
    for line in fr.readlines():
        line = line.strip()#
        line = line.replace('\'','')
        line = line.replace('\"','')
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        consumeType = listFromLine[3]
        consumeMoney = float(listFromLine[5])
        consumeYue = float(listFromLine[6])
        if keyTemp not in cardCount:
            cardCount[keyTemp] = [0.0,0.0,0.0,0.0,100000,0.0];
        indexT = tool(consumeType);
        cardCount[keyTemp][3] = max(cardCount[keyTemp][3] , consumeYue)
        cardCount[keyTemp][4] = min(cardCount[keyTemp][4] , consumeYue)
        if (not indexT==3) and (not indexT==4):
            cardCount[keyTemp][indexT] +=consumeMoney;

    resultT = [[],[]];
    zhuxuej = [0,0,0,0,0];
    for keyT in subidyCount:
        if keyT in cardCount:
            currentLine = cardCount[keyT];
            indTTT = tool2(subidyCount[keyT]['subidy'])
            zhuxuej[indTTT]+=1;
            resultT[0].append(indTTT+1);
            resultT[1].append(currentLine);
    print();
    plt.subplot(231)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=7000,ymin=0)
    plt.xlabel("shitang")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,:1]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(232)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=7000,ymin=0)
    plt.xlabel("kaishui")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,1:2]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(233)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=3000,ymin=0)
    plt.xlabel("linyu")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,2:3]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(234)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=(array(resultT[1])[:,3:4]).max(),ymin=0)
    plt.xlabel("yuemax")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,3:4]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(235)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=(array(resultT[1])[:,4:5]).max(),ymin=0)
    plt.xlabel("yuemin")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,4:5]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(236)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=10000,ymin=0)
    plt.xlabel("qita")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,5:6]),marker='x',color = 'm',s=30,alpha=0.5)
    print(zhuxuej)
    plt.show()
def readScore():
    fr = open(subsidyFilePath,encoding='utf-8');
    subidyCount = {};
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp not in subidyCount:
            subidyCount[keyTemp] = {};
        subidyCount[keyTemp]['subidy'] =int(listFromLine[1]);

    fr = open(scoreFilePath,encoding='utf-8');
    scoreResult = [[],[],[],[]];
    for line in fr.readlines():
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        if keyTemp in subidyCount:
            scoreResult[tool2(subidyCount[keyTemp]['subidy'])].append([int(listFromLine[1]),int(listFromLine[2])])
    scoreResult = array(scoreResult);
    plt.scatter(array([1]).repeat(len(array((scoreResult[0]))[:,1:2])), array((scoreResult[0]))[:,1:2], marker = 'x', color = 'm', label='1', s = 1,alpha=0.5)
    plt.scatter(array([2]).repeat(len(array((scoreResult[1]))[:,1:2])), array((scoreResult[1]))[:,1:2], marker = '+', color = 'black', label='2', s = 1,alpha=0.5)
    plt.scatter(array([3]).repeat(len(array((scoreResult[2]))[:,1:2])), array((scoreResult[2]))[:,1:2], marker = 'o', color = 'r', label='3', s = 1,alpha=0.5)
    plt.scatter(array([4]).repeat(len(array((scoreResult[3]))[:,1:2])), array((scoreResult[3]))[:,1:2], marker = '4', color = 'b', label='4', s = 1,alpha=0.5)
    plt.show()
def readTest2():
    fr = open('F:\数据\精准资助\精准资助\\train\\trainDataSet.txt',encoding='utf-8');
    resultT = [[],[]];
    indexT = 0
    for line in fr.readlines():
        if indexT==0:
            indexT+=1;
            continue;
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        resultT[1].append([int(listFromLine[2]),int(listFromLine[3])])
        resultT[0].append(1+tool2(int(listFromLine[4])))
    plt.subplot(211)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=(array(resultT[1])[:,0:1]).max(),ymin=0)
    plt.xlabel("borrowCount")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,0:1]),marker='x',color = 'm',s=30,alpha=0.5)

    plt.subplot(212)
    plt.xlim(xmax=7,xmin=0)
    plt.ylim(ymax=(array(resultT[1])[:,0:1]).max(),ymin=0)
    plt.xlabel("libraryCount")
    plt.ylabel("y")
    plt.scatter([1,2,3,4],junzhi(resultT[0],array(resultT[1])[:,1:2]),marker='x',color = 'm',s=30,alpha=0.5)
    plt.show()
def cardStatis(filePath):
    fr = open(filePath,encoding='utf-8');
    cardCount = {};
    for line in fr.readlines():
        line = line.strip()#
        line = line.replace('\'','')
        line = line.replace('\"','')
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        consumeType = listFromLine[3]
        consumeMoney = float(listFromLine[5])
        if keyTemp not in cardCount:
            cardCount[keyTemp] = [0.0,0.0,0.0];
        cardCount[keyTemp][0] =max(cardCount[keyTemp][0],float(listFromLine[6]));
        if consumeType=='食堂':
            cardCount[keyTemp][1] += consumeMoney;
        elif consumeType=='开水':
            cardCount[keyTemp][2] += consumeMoney;
    return cardCount;
def readAllData(file1,file2,file3,file4):
    cardCount = cardStatis(file1)
    fr = open(file2,encoding='utf-8');
    indexT = 0;
    resultT = [];
    for line in fr.readlines():
        if indexT == 0:
            indexT+=1;
            continue;
        current = [];
        line = line.strip()#
        listFromLine = line.split(',')
        keyTemp = int(listFromLine[0]);
        current.append(keyTemp)
        if keyTemp not in cardCount:
            current.extend([0.0,0.0,0.0])
        else :
            current.extend(cardCount[keyTemp])
        current.extend([int(listFromLine[2]),int(listFromLine[3])])
        if len(listFromLine) ==5:
            current.append(int(listFromLine[4]))
        resultT.append(current)
    return resultT;
def findMax(arrayT):
    valuseS = [0,1000,1500,2000]
    maxT = 0.0
    indexT = 0;
    for i in arrayT:
        if i>maxT:
            return valuseS[indexT]
        indexT+=1;

def designTree():
    trainData =readAllData(cardFilePath,'F:\数据\精准资助\精准资助\\train\\trainDataSet.txt',borrowFilePath,subsidyFilePath);
    trainData =array(trainData)
    testData =array(readAllData(test_cardFilePath,'F:\数据\精准资助\精准资助\\test\\testDataSet.txt',borrowFilePath,subsidyFilePath));

    clf = tree.DecisionTreeClassifier()
    trainT = trainData[:,2:6];
    trainLabel = trainData[:,6:7];
    clf.fit(trainData[:,2:6], trainLabel)
    #print(clf.feature_importances_)
    fw = open('F:\数据\精准资助\精准资助\\train\designTreeResult3.csv','w');
    fw.write("studentid,subsidy\n")
    countAll = [0,0,0,0];

    for testLine in testData:
        #r_ = clf.predict_proba(testLine[1:6]);
        r_ = clf.predict(testLine[2:6]);
        resT = int(r_[0]);
        countAll[tool2(resT)]+=1;
        fw.write(str(int(testLine[0]))+","+str(resT)+'\n');
    print(countAll);
designTree();