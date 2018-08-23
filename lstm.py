# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:48:58 2018

@author: yaradong
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#定义常量
rnn_unit=10       #hidden layer units
input_size=6
output_size=1
lr=0.000005 

raw_data = pd.read_table('C:/Users/yaradong/Desktop/bitcoin/huobikline.txt',encoding='gbk', sep=',')
raw_data = raw_data.sort_index(ascending=False)

raw_data_y =raw_data['收盘价']

clm_size = raw_data.columns.size
data=raw_data.iloc[:,1:clm_size].values  #取第3-10列
data_y = raw_data_y.iloc[:].values
data_y = data_y[1:]
data_y = np.append(data_y,data_y[-1])

#获取训练集
def get_train_data(batch_size=40,time_step=20,train_begin=0,train_end=1500):
    batch_index=[]
    data_train=data[train_begin:train_end]
    y_train=data_y[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    
    
    normalized_train_y = (y_train-np.mean(y_train, axis=0))/np.std(y_train, axis=0)
    
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:6]
       #y=normalized_train_data[i:i+time_step,7,np.newaxis]
       y=normalized_train_y[i:i+time_step,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(time_step=20,test_begin=1500):
    data_test=data[test_begin:]
    y_test=data_y[test_begin:]
    
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    normalized_test_y = (y_test-np.mean(y_test, axis=0))/np.std(y_test, axis=0)
    
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:6]
       y=normalized_test_y[i*time_step:(i+1)*time_step]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:6]).tolist())
    test_y.extend((normalized_test_y[(i+1)*time_step:]).tolist())
    return mean,std,test_x,test_y



#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input_x=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input_x,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=40,time_step=20,train_begin=0,train_end=1500):
    
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    #tf.reset_default_graph()
    #with tf.Graph().as_default():
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    
    print("yyyy")
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    checkpoint_dir = 'C:/Users/yaradong/Desktop/bitcoin/save1/'
    module_file = tf.train.latest_checkpoint(checkpoint_dir) 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #saver.restore(sess, "C:/Users/yaradong/Desktop/bitcoin/save/mode.mod")
        #重复训练10000次
        for i in range(8000):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==199:
                print("保存模型：",saver.save(sess,"./save1/mode.mod",global_step=i))


#train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=20):
    
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    checkpoint_dir = 'C:/Users/yaradong/Desktop/bitcoin/save1/'
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_predict=np.array(test_predict)*std[3]+mean[3]
        
        #test_y=np.array(test_y)*std[6]+mean[6]
        test_y = data_y[1500:]
        
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        print("acc:",acc)
        #以折线图表示结果
        #plt.figure()
        plt.figure(figsize=(10, 5))
        plt.plot(list(range(len(test_predict))), test_predict, color='b', label='predict')
        plt.plot(list(range(len(test_y))), test_y,  color='r', label='origin')
        plt.xlabel('time_step')
        plt.ylabel('values')
        plt.legend(loc=4)
        plt.show()

prediction() 

