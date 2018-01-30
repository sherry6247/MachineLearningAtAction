import numpy
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import operator #operator是运算符模块

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
#使用KNN将每组数据划分到某个类中
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffmat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffmat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistIndexs = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndexs[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 将classCount分解成元组列表
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
# 处理输入格式问题
def file2matrix(filename):
    with open(filename)as f:
        arrayOLines = f.readlines()
        numberOfLines = len(arrayOLines)
        # zeros返回一个给定形状和类型的用0填充的数组
        returnMat = zeros((numberOfLines,3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector
# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" %(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is : %f" %(errorCount/numTestVecs))
# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percenTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream  = float(input("Liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percenTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person:", resultList[classifierResult - 1])

if __name__ =='__main__' :
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 111的意思是画布分成一行一列，然后图像画在从左到右从上到下第一块
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    print(classifyPerson())