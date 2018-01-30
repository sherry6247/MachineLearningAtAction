import numpy
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
c = []
with open('./test1.txt','r')as f:
    lines = f.readlines()
    numberOfLines = len(lines)
    returnMat = zeros((numberOfLines, 3))
    c = []
    index = 0
    for line in lines:
        line = line.strip()
        listF = line.split('\t')
        returnMat[index,:] = listF[0: 3]
        c.append(int(listF[-1]))
        index += 1
    # print(returnMat)
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
if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('./test1.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    plt.show()
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(342)
    # plt.show()
