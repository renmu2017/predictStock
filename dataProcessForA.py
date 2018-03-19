#!/usr/bin/env python
# encoding: utf-8
'''
predictStock.dataProcess -- shortdesc

predictStock.dataProcess is a processing of stock data from excel-file

It defines classes_and_methods

@author:     renmu

@copyright:  2018 renmu. All rights reserved.

@license:    license

@contact:    renmu@126.com
@deffield    updated: Updated
'''
import numpy as np
import copy
import random
from numpy import rank

def readData():
    '''
            数据读取，从csv文件中读取数据，转成列表形式
    '''
    data = []
    date = []
    with open('E:\AI\data\stock_data\szzhzs.CSV') as file:
        file.readline()
        line = file.readline()
        while line:
            line = file.readline().strip('\n')
            data_i = line.split(',')
            i0 = data_i[0]
            data_i =[float(i) for i in data_i[1:]]
            #data_i.insert(0,i0)
            data.append(data_i)
            date.append(i0)
        data = data[:-1]
        date = date[:-1]
    return data,date

def normalization(data):
    '''
            归一化处理，将所有数值参数与其百日均值比值作为当前数据，比例数据不变
    '''
    #dataNumpy = np.asanyarray(data)
    newData = []
    labels = []
    for i in range(len(data)):
        labels.append(data[i][-1])#最后一列为涨跌幅
        row = copy.deepcopy(data[i])
        forwardData = data[i-100 if i>100 else 0:i+1]#求前100日的数据矩阵
        fDataNumpy = np.array(forwardData)
        rankAvg = np.mean(fDataNumpy,axis=0)#前100日数据均值
        rankAvg[-1] = 1
        rowNumpy = np.array(row)
        rowNumpy = rowNumpy/rankAvg
        
#         print(rowNumpy)
        row = rowNumpy.tolist()
#         row.extend(rankAvg.tolist())#将前100日数据加入当日数据行
#         print (row)
        newData.append(row)
        
    for i in range(100,len(newData)-1):
        for j in range(100):
            newData[i].append(newData[j][0])#当日数据加入前100日收盘价
        labels[i]=labels[i+1]#第二日涨跌幅为当日标签
    newData=newData[100:]#删掉前100日数据
    #labels变为one_hot形式
    newlabels = []
    for f in labels:
        f = round(f)
        one_hot = []
        for i in range(21):
            if i-10==f:
                one_hot.append(1)
            elif i==20 and f>10:
                one_hot.append(1)
            else:
                one_hot.append(0)
        newlabels.append(one_hot)
#     print (newlabels)        
    newlabels=newlabels[100:]
    testLine = int(len(newData)*4/5)#4/5的数据为训练数据，其余为测试数据
    #print (newData)
    train = newData[:testLine]
    trainLabels = newlabels[:testLine]
    test = newData[testLine:-1]
    testLabels = newlabels[testLine:-1]
    return train,test,trainLabels,testLabels
def random_batch(train,trainLabels,num):
    ranList = random.sample([i for i in range(len(train))],num)
    train_batch = []
    trainLabels_batch = []
    for index in ranList:
        train_batch.append(train[index])
        trainLabels_batch.append(trainLabels[index])
    return train_batch,trainLabels_batch

def next_batch(train,trainLabels,num):
    pass


if __name__ == '__main__':
    data,date= readData()
    date = date[100:-1]#删掉日期前100日
    train,test,trainLabels,testLabels= normalization(data)
    filename = 'E:\AI\data\stock_data\write_data.txt'
    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(str(train))
        f.write(str(test))
