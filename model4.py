# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:14:35 2019

@author: falcon1
"""

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Conv1D, Concatenate, Bidirectional, LSTM
from keras.layers import Dropout, Reshape, Permute, Embedding, CuDNNLSTM
from keras.regularizers import l2
import random, math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import time
from datetime import timedelta

HIGH = 21
WIDTH = 1000
rampup_length = 80
rampdown_length = 50
num_epochs = 3
BATCH_SIZE = 32
learning_rate_max = 0.003
scaled_unsup_weight_max = 100
l2value = 0.01

def load_data():
    with open('cytoplasm_pos_neg.txt') as fr:
        lns = fr.readlines()
        
    labels=[]
    seqs=[]
    for line in lns:
        ls = line.split()
        if int(ls[0]) == 0:
            labels.append([0,1])
        else:
            labels.append([1,0])
        seqs.append(ls[2])
    
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x = token.texts_to_sequences(seqs)
    x = pad_sequences(x, maxlen=1000, padding='post', truncating='post')
    y = np.array(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    
    for i in range(len(x_train)):
        if random.random() < 0.5:
            y_train[i] = [0,0]
            
    np.savez('padSequences.npz',x_train, x_test, y_train, y_test)
    
    
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

#load_data() # just run in the program firstly run
data = np.load('padSequences.npz')
x_train, x_test, y_train, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
nb_train = len(x_train)
total_batch = int( np.ceil( nb_train / BATCH_SIZE))

sess = tf.InteractiveSession()
epoch = tf.Variable(0, name='ep', trainable=False)  
rate = tf.Variable(0.0)
weight = tf.Variable(0.0)
# input
with tf.name_scope("input"):
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,1000))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None,2))
    y_true_cls = tf.argmax(y_true, dimension=1)

# Embedding layer
with tf.name_scope("embedding"):
    x = Embedding(output_dim=100, input_dim=23, input_length=1000)(inputs)    
# first convolutionary layers
with tf.name_scope("cnn_1"):    
    x1_1 = Conv1D(20,1, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_2 = Conv1D(20,3, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_3 = Conv1D(20,5, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_4 = Conv1D(20,9, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_5 = Conv1D(20,15, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_6 = Conv1D(20,21, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)

    # Concatenate all CNN layers
    x1 = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])

# second CNN layer
with tf.name_scope("cnn_2"):
    #x2 = Conv2D(128,(3,120), padding="same")(x1)
    x2 = Conv1D(128,3, padding="same")(x1)
# Dropout
with tf.name_scope("drop_1"):
    l_drop = Dropout(rate=0.5)(x2)

# Reshape
#with tf.name_scope("reshape"):
#    l_reshape = Reshape((128,-1))(l_drop)
#    l_reshape = Permute((2,1))(l_reshape)

# BiLSTM  layer
with tf.name_scope("bilstm"):    
    lstm = CuDNNLSTM(128)
    l_bilstm = Bidirectional(lstm)(l_drop)

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
# acc
with tf.name_scope("acc"):
    y_pred_cls = tf.argmax(y_pred_1, dimension=1)
    accuracy = tf.reduce_mean(tf.cast( tf.equal( y_pred_cls, y_true_cls), dtype=tf.float32))
# train
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(rate).minimize(loss)  


sess.run(tf.global_variables_initializer()) # initial variables

ckpt_dir = "log/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(max_to_keep=2)
summary_wirter = tf.summary.FileWriter(ckpt_dir, sess.graph)

# if the checkpoint file is existed, read the lastest checkpoint file and restore variable value.
ckpt = tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
    saver.restore(sess, ckpt)
else:
    print("Training from scratch")

start_epoch = sess.run(epoch)    
print("Training starts from {} epoch".format(start_epoch+1))

best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 2 # required number of iterations in which the improvement is found

start_time = time.time()
for ep in range(0, num_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    sess.run(tf.assign(epoch, ep+1))
    sess.run(tf.assign(weight, unsupWeight(ep)))
    sess.run(tf.assign(rate, learningRate(ep)))
    
    for i in range(total_batch): # for each miniBatch
        start = (i * BATCH_SIZE) % nb_train
        end = min(start + BATCH_SIZE, nb_train)
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]
        
        sess.run(train_step, feed_dict={inputs: batch_x, y_true: batch_y})
               
    
    print("epoch {} finished".format(ep)) 
    '''
    acc_validation = sess.run(accuracy, feed_dict={inputs:x_test, y_true:y_test})
    if acc_validation > best_validation_accuracy:
        best_validation_accuracy = acc_validation
        last_improvement = ep
        # save checkpoint
        saver.save(sess, ckpt+"model.cpkt", global_step=ep+1)
    if ep - last_improvement > require_improvement:
        print("No imporvement found in a while, stopping optimization")
        # break out from the for-loop
        break
    '''
    saver.save(sess, ckpt+"model.cpkt", global_step=ep+1)
end_time = time.time()
print("Time usage: {}".format( timedelta(seconds=int(round(start_time-end_time)))))
        
y_pred = sess.run(y_pred_1, feed_dict={inputs: x_test, y_true: y_test})   
pred_result_accuracy(y_pred, y_test)  
sess.close()                
            
            
            
    


