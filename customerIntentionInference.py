#!/usr/bin/env python
# encoding: utf-8
'''
Created on 2018��3��16��
@author: WQ
'''
import numpy as np
import tensorflow as tf
import os
import matplotlib as plt
import pandas as pd
import random
import sys
sys.path.append('d:/anaconda3/lib/site-packages')
from sklearn import preprocessing as prep

LOG_DIR='./logs'

file_path='data_encoder.csv'

data=pd.read_csv(file_path)
# print(data)
label =np.array(pd.get_dummies( data.iloc[:,0])).tolist()
print(len(label[0]))
sample=data.iloc[:,1:]
sample = np.array(prep.scale(sample)).tolist()
testNum=int(len(label)*2/3)

trainSample=sample[:testNum]
trainLabel=label[:testNum]
testSample=sample[testNum:]
testLabel=label[testNum:]

x = tf.placeholder("float",[None,23])    
y_ = tf.placeholder("float",[None,61])

n_stocks = 23

n_neurons_1 = 70
    
n_neurons_2 = 70
    
    
n_neurons_4 = 70
    
n_target = 61
# Layer 1: Variables for hidden weights and biases
    
W_hidden_1 = tf.Variable(tf.random_normal([n_stocks, n_neurons_1]))
    
bias_hidden_1 = tf.Variable(tf.random_normal([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases

W_hidden_2 = tf.Variable(tf.random_normal([n_neurons_1, n_neurons_2]))

bias_hidden_2 = tf.Variable(tf.random_normal([n_neurons_2]))

# Layer 4: Variables for hidden weights and biases

W_hidden_4 = tf.Variable(tf.random_normal([n_neurons_2, n_neurons_4]))

bias_hidden_4 = tf.Variable(tf.random_normal([n_neurons_4]))



# Output layer: Variables for output weights and biases

W_out = tf.Variable(tf.random_normal([n_neurons_4, n_target]))

bias_out = tf.Variable(tf.random_normal([n_target]))


# Hidden layer




hidden_1 = tf.nn.tanh(tf.add(tf.matmul(x, W_hidden_1), bias_hidden_1))

hidden_2 = tf.nn.tanh(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

hidden_4 = tf.nn.tanh(tf.add(tf.matmul(hidden_2, W_hidden_4), bias_hidden_4))



# Output layer (must be transposed)

out =tf.add(tf.matmul(hidden_4, W_out), bias_out)    
y=tf.nn.softmax(out)

#损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-09))
# cross_entropy = tf.reduce_mean(tf.squared_difference(y, y_))
#设定学习率随时间递减
global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.1 #初始学习率

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=500,decay_rate=0.8)
#训练过程梯度下降法最小化损失函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

#初始化session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
def random_batch(sam,lab,n):
    ranList = random.sample([i for i in range(len(sam))],n)
    train_batch = []
    trainLabels_batch = []
    for index in ranList:
        train_batch.append(sam[index])
        trainLabels_batch.append(lab[index])
    return np.array(train_batch).tolist(),np.array(trainLabels_batch).tolist()


def next_batch(sam,lab,startNum,n):
    if startNum<len(sam)-n:
        train_batch_1 = sam[startNum:startNum+n]
        trainLabels_batch_1 = lab[startNum:startNum+n]
    return np.array(train_batch_1).tolist(),np.array(trainLabels_batch_1).tolist()

startNum=0
for i in range(100000):
    #取随机100个训练样本
    
#     batch_xs,batch_ys = trainSample,trainLabel
    batch_xs,batch_ys = random_batch(trainSample,trainLabel,30000)
    startNum+=100
    optt,cross,msee = sess.run([train_step,cross_entropy,W_hidden_1],feed_dict = {x:batch_xs,y_:batch_ys})
    #每100步进行测试
    if i%100==0 or i == 99999:
        print(optt,':',cross,':',msee[0][0])
        checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=i)
        
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        batch_test_xs,batch_test_ys = random_batch(np.array(testSample).tolist(),np.array(testLabel).tolist(),10000)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        accuracy,argmaxN =sess.run([accuracy,tf.argmax(y,1)],feed_dict = {x:batch_test_xs,y_:batch_test_ys})
        print ("正确率为：",accuracy,'预测值为',argmaxN[:20])
