# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:31:02 2019

@author: falcon1
"""
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Input, Dense, Conv2D, Concatenate, Bidirectional, LSTM
from keras.layers import Dropout, Reshape, Permute
from keras.regularizers import l2
from keras.models import Model
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import math
import numpy as np

HIGH = 21
WIDTH = 1000
rampup_length = 80
rampdown_length = 50
num_epochs = 300
BATCH_SIZE = 32
learning_rate_max = 0.003
scaled_unsup_weight_max = 100

def rampup(epoch):
    if epoch < rampup_length:
        p = 1.0 - float(epoch)/rampup_length
        return math.exp(-p*p*5.0)
    else:
        return 1.0
    
def rampdown(epoch):
    if epoch >= num_epochs - rampdown_length:
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep*ep) / rampdown_length)
    return 1.0

def unsupWeight(epoch):
    return rampup(epoch) * scaled_unsup_weight_max

# learning rate
def learningRate(epoch):
    return rampup(epoch) * rampdown(epoch) * learning_rate_max

def pred_result_accuracy(y_pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
            count = count + 1
    print(count/len(y_true))


data = np.load('testModelDataset_3.npz')
x_train, x_test, y_train, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
nb_train = len(x_train)
total_batch = int( np.ceil( nb_train / BATCH_SIZE))

sess = tf.InteractiveSession()
ep = tf.Variable(0.0)  
rate = tf.Variable(0.0)
weight = tf.Variable(0.0)
# input
with tf.name_scope("input"):
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,1000,21,1))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None,2))
    
# first convolutionary layers
with tf.name_scope("cnn_1"):    
    #x1_1 = Conv2D(20,(1,21), padding="same")(inputs)
    x1_2 = Conv2D(20,(3,21), padding="same")(inputs)
    #x1_3 = Conv2D(20,(5,21), padding="same")(inputs)
    #x1_4 = Conv2D(20,(9,21), padding="same")(inputs)
    #x1_5 = Conv2D(20,(15,21), padding="same")(inputs)
    #x1_6 = Conv2D(20,(21,21), padding="same")(inputs)

    # Concatenate all CNN layers
    #x1 = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])
    #x1 = Concatenate()([x1_2, x1_3])
# second CNN layer
with tf.name_scope("cnn_2"):
    #x2 = Conv2D(128,(3,120), padding="same")(x1)
    x2 = Conv2D(128,(3,20), padding="same")(x1_2)
# Dropout
with tf.name_scope("drop_1"):
    l_drop = Dropout(rate=0.5)(x2)

# Reshape
with tf.name_scope("reshape"):
    l_reshape = Reshape((128,-1))(l_drop)
    l_reshape = Permute((2,1))(l_reshape)

# BiLSTM  layer
with tf.name_scope("bilstm"):    
    l_bilstm = Bidirectional(LSTM(128))(l_reshape)

with tf.name_scope("drop_2"):
    l_drop_2 = Dropout(rate=0.5)(l_bilstm)

# output layer
with tf.name_scope("out"):    
    y_pred_1 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)
    y_pred_2 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)

# loss
with tf.name_scope("loss"):
    nb_label = tf.reduce_sum(y_true)
    loss_label = tf.reduce_sum(-tf.reduce_sum(y_true * tf.log(y_pred_1), axis=-1) * tf.reduce_sum(y_true, axis=-1)) / nb_label
    loss_unlabel = tf.reduce_mean( tf.square( y_pred_2 - y_pred_1))
    loss = loss_label + weight * loss_unlabel

# train
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(rate).minimize(loss)  


sess.run(tf.global_variables_initializer())


for epoch in range(0, num_epochs):
    sess.run(tf.assign(ep, epoch))
    sess.run(tf.assign(weight, unsupWeight(epoch)))
    sess.run(tf.assign(rate, learningRate(epoch)))
    if (epoch % 50) == 0 and epoch > 0:
        print("epoch======>", epoch)   
    for i in range(total_batch): # for each miniBatch
        start = (i * BATCH_SIZE) % nb_train
        end = min(start + BATCH_SIZE, nb_train)
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]
        
        sess.run(train_step, feed_dict={inputs: batch_x, y_true: batch_y})

y_pred = sess.run(y_pred_1, feed_dict={inputs: x_test, y_true: y_test})   
pred_result_accuracy(y_pred, y_test)       
sess.close()                
            
            
            
    


