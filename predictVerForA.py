#!/usr/bin/env python
# encoding: utf-8
'''
Created on 2018��3��16��
@author: WQ
'''
import dataProcess as dp
import numpy as np
import tensorflow as tf
import os

LOG_DIR = './logs'

def main(_):
  while tf.gfile.Exists(LOG_DIR):
    tf.gfile.DeleteRecursively(LOG_DIR)
  tf.gfile.MakeDirs(LOG_DIR)
  run_training()

def run_training():
    #读取数据
    data,date= dp.readData()
    train,test,trainLables,testLabels= dp.normalization(data)
    
    #添加参数及变量
    x = tf.placeholder("float",[None,109])
    
    w = tf.Variable(tf.random_normal([109,21]))
    b = tf.Variable(tf.random_normal([21]))
    
    y = tf.nn.softmax(tf.matmul(x,w)+b)
    
    y_ = tf.placeholder("float",[None,21])
    
    #损失函数
    cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-9))
    
    #设定学习率随时间递减
    global_step = tf.Variable(0, trainable=False)
    
    initial_learning_rate = 0.0001 #初始学习率
    
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10,decay_rate=0.9)
    #训练过程梯度下降法最小化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    #初始化session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(5000):
        #取随机100个训练样本
        batch_xs,batch_ys = dp.random_batch(train,trainLables,100)
        sess.run(train_step,feed_dict = {x:batch_xs,y_:batch_ys})
        #每10步进行测试
        if i%100==0 or i == 4999:
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=i)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            batch_test_xs,batch_test_ys = dp.random_batch(test,testLabels,100)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            accuracy,argmaxN =sess.run([accuracy,tf.argmax(y,1)],feed_dict = {x:batch_test_xs,y_:batch_test_ys})
            print ("正确率为：",accuracy,"测试结果为：",argmaxN)
def predict():
    data,date= dp.readData()
    train,test,trainLables,testLabels= dp.normalization(data)
    
    with tf.Session() as sess:
        
        # Restore variables from disk.
        predict_xs=test[-50:-1]
        predict_ys = testLabels[-50:-1]
        
        
        x = tf.placeholder("float",[None,109])
    
        w = tf.Variable(tf.random_normal([109,21]))
        b = tf.Variable(tf.random_normal([21]))
        
        y = tf.nn.softmax(tf.matmul(x,w)+b)
        
        y_ = tf.placeholder("float",[None,21])
        
        #损失函数
        cross_entropy = -tf.reduce_sum(y_*tf.log(y+1e-9))
        
        #设定学习率随时间递减
        global_step = tf.Variable(0, trainable=False)
        
        initial_learning_rate = 0.0001 #初始学习率
        
        learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=10,decay_rate=0.9)
        #训练过程梯度下降法最小化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(LOG_DIR, ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        print(sess.run(tf.argmax(y,1),feed_dict = {x:predict_xs,y_:predict_ys}))
    
    



if __name__ == '__main__':
    tf.app.run(main=main)
