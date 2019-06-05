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
import re
from datetime import timedelta
from Bio import SeqIO

HIGH = 21
WIDTH = 1000
rampup_length = 80
rampdown_length = 50
num_epochs = 200
BATCH_SIZE = 32
TOTAL_TRAIN_BATCH = 0
TOTAL_TEST_BATCH = 0
nb_train = 0
nb_test = 0
learning_rate_max = 0.003
scaled_unsup_weight_max = 100
l2value = 0.01
unlabeled_rate = 0.5

'''
read sequences from cytoplasm_pos_60.fa and cytoplasm_neg_60.fa respectively,
and cut out 1000 amino acids by pad_sequences.
Finally, save sequences to cytoplasm_pos_padsequence_60.npy and cytoplasm_neg_padsequences_60.npy
'''
def save_data():
    seq_records = SeqIO.parse('cytoplasm_pos_60.fa', 'fasta')
    seqs=[]
    for record in seq_records:
        seq = str(record.seq)
        seq = re.sub('[ZUB]',"",seq)
        seqs.append(seq)
    
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x_pos = token.texts_to_sequences(seqs)
    x_pos = pad_sequences(x_pos, maxlen=1000, padding='post', truncating='post')
    np.save('cytoplasm_pos_padsequence_60.npy', x_pos)
    
    seq_records = SeqIO.parse('cytoplasm_neg_60.fa', 'fasta')
    seqs=[]
    for record in seq_records:
        seq = str(record.seq)
        seq = re.sub('[ZUB]',"",seq)
        seqs.append(seq)
    
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x_neg = token.texts_to_sequences(seqs)
    x_neg = pad_sequences(x_neg, maxlen=1000, padding='post', truncating='post')
    np.save('cytoplasm_neg_padsequence_60.npy', x_neg)

def split_train_test(test_rate=0.2):
    x_train_pos = []
    x_test_pos = []
    x_pos = np.load('cytoplasm_pos_padsequence_60.npy')
    x_neg = np.load('cytoplasm_neg_padsequence_60.npy')
    
    x_train_pos, x_test_pos = train_test_split(x_pos, test_size=test_rate)
    x_train_neg, x_test_neg = train_test_split(x_neg, test_size=test_rate)
    
    np.savez('cytoplasm_pos_neg_padsequence_60.npz', x_train_pos, x_test_pos, x_train_neg, x_test_neg)

def buildBenchmarkDataset(unlabeled_rate=0.5):
    data = np.load('cytoplasm_pos_neg_padsequence_60.npz')
    x_train_pos, x_test_pos, x_train_neg, x_test_neg = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    y_train_pos = np.zeros(shape=(len(x_train_pos), 2), dtype=int)
    y_test_pos = np.zeros(shape=(len(x_test_pos), 2), dtype=int)
    y_train_pos[:,0] = 1
    y_test_pos[:,0] = 1
    
    y_train_neg = np.zeros(shape=(len(x_train_neg), 2), dtype=int)
    y_test_neg = np.zeros(shape=(len(x_test_neg), 2), dtype=int)
    y_train_neg[:,1] = 1
    y_test_neg[:,1] = 1
        
    for i in range(len(x_train_pos)):
        if random.random() < unlabeled_rate:
            y_train_pos[i] = [0,0]
    
    for i in range(len(x_train_neg)):
        if random.random() < unlabeled_rate:
            y_train_neg[i] = [0,0]

    x_train = np.concatenate((x_train_pos, x_train_neg))
    y_train = np.concatenate((y_train_pos, y_train_neg))
    
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))
    
    np.savez("".join(['ctplsm60_padSequences_train', str(int(unlabeled_rate*100)), '.npz']),x_train, y_train)
    np.savez("".join(['ctplsm60_padSequences_test', str(int(unlabeled_rate*100)), '.npz']), x_test, y_test)

def get_global_number(trainFile, testFile):
    data = np.load(trainFile) # 'ctplsm60_padSequences_train.npz'
    x = data['arr_0']
    nb_train = len(x)
    total_train_batch = int(np.ceil(nb_train / BATCH_SIZE))
    
    data = np.load(testFile) # 'ctplsm60_padSequences_test.npz'
    x = data['arr_0']
    nb_test = len(x)
    total_test_batch = int(np.ceil(nb_test / BATCH_SIZE))
    return nb_train, total_train_batch, nb_test, total_test_batch

def load_validate_data(validateFile):
    data = np.load(validateFile) # 'ctplsm60_padSequences_test.npz'
    x, y = data['arr_0'], data['arr_1']

    indx = np.random.choice(len(x), 100)
    return x[indx,:], y[indx,:]

def load_train_batch_data(trainFile, n_epoch, n_batch ):
    data = np.load(trainFile) # 'ctplsm60_padSequences_train.npz'
    x, y = data['arr_0'], data['arr_1']
    x, y = shuffle(x, y, random_state=n_epoch )
    
    start = (n_batch * BATCH_SIZE) % nb_train
    end = min(start + BATCH_SIZE, nb_train)
    batch_x = x[start:end]
    batch_y = y[start:end]
    
    return batch_x, batch_y

def load_test_batch_data(testFile, n_batch):
    global TOTAL_TEST_BATCH
    global nb_test
    data = np.load(testFile) # 'ctplsm60_padSequences_test.npz'
    x, y = data['arr_0'], data['arr_1']
    
    start = (n_batch * BATCH_SIZE) % nb_test
    end = min(start + BATCH_SIZE, nb_test)
    return x[start:end], y[start:end]

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


# load data
# save_data() read sequences from fasta and cut out 1000 amino acids.It just is run firstly
if not os.path.exists('cytoplasm_neg_60.npy'):
    save_data() 
# split_train_test() split train and test dataset. It just is run firstly   
if not os.path.exists('cytoplasm_pos_neg_padsequence_60.npz'):
    split_train_test() 
    
trainFile = "".join(['ctplsm60_padSequences_train', str(int(unlabeled_rate * 100)), '.npz'])
testFile = "".join(['ctplsm60_padSequences_test', str(int(unlabeled_rate * 100)), '.npz'])
buildBenchmarkDataset(unlabeled_rate)
    
nb_train, TOTAL_TRAIN_BATCH, nb_test, TOTAL_TEST_BATCH = get_global_number(trainFile, testFile)

sess = tf.InteractiveSession()
epoch = tf.Variable(0, name='ep', trainable=False)  
rate = tf.Variable(0.0)
weight = tf.Variable(0.0)
# input
with tf.name_scope("input"):
    inputs = tf.placeholder(dtype=tf.float32, shape=(None,1000))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None,2))
    y_true_cls = tf.argmax(y_true, axis=1)

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

# BiLSTM  layer
with tf.name_scope("bilstm"):    
    lstm = CuDNNLSTM(128)
    #lstm = LSTM(10)
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
    y_pred_cls = tf.argmax(y_pred_1, axis=1)
    accuracy = tf.reduce_mean(tf.cast( tf.equal( y_pred_cls, y_true_cls), dtype=tf.float32))
# train
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(rate).minimize(loss)  


sess.run(tf.global_variables_initializer()) # initial variables

ckpt_dir = "".join(["log/ctplsm60/", str(int(unlabeled_rate * 100)), "/"])
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver(max_to_keep=3)
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
require_improvement = 10 # required number of iterations in which the improvement is found

start_time = time.time()
for ep in range(start_epoch, num_epochs):

    sess.run(tf.assign(epoch, ep+1))
    sess.run(tf.assign(weight, unsupWeight(ep)))
    sess.run(tf.assign(rate, learningRate(ep)))
    
    for i in range(TOTAL_TRAIN_BATCH): # for each miniBatch
        batch_x, batch_y = load_train_batch_data(trainFile, ep, i)
        sess.run(train_step, feed_dict={inputs: batch_x, y_true: batch_y})
    
    print("epoch {} finished".format(ep)) 
    x_validate, y_validate = load_validate_data(testFile)
    acc_validation = sess.run(accuracy, feed_dict={inputs:x_validate, y_true:y_validate})
    print("{} step validation accuracy {}".format(ep, acc_validation))
    
    if acc_validation > best_validation_accuracy:
        best_validation_accuracy = acc_validation
        last_improvement = ep
        # save checkpoint
        saver.save(sess, ckpt_dir+"model.cpkt", global_step=ep+1)
    if ep - last_improvement > require_improvement:
        print("No imporvement found in a while, stopping optimization")
        # break out from the for-loop
        break
   
    saver.save(sess, ckpt_dir+"model.cpkt", global_step=ep+1)
end_time = time.time()
print("Time usage: {}".format( timedelta(seconds=int(round(start_time-end_time)))))

flag = 0
for i in range(TOTAL_TEST_BATCH):
    x_test_batch, y_test_batch = load_test_batch_data(testFile, i)
    y_pred_batch = sess.run(y_pred_1, feed_dict={inputs: x_test_batch, y_true: y_test_batch})  
    if flag == 0:
        y_pred, y_test = y_pred_batch, y_test_batch
        flag = 1
    else:
        y_pred, y_test= np.concatenate((y_pred, y_pred_batch)), np.concatenate((y_test,y_test_batch))

from sklearn.metrics import matthews_corrcoef, precision_score, recall_score,accuracy_score 
from sklearn.metrics import confusion_matrix 
y_p = np.array(y_pred> 0.5).astype(int)
print("Accuracy=%f"%(accuracy_score(y_test[:,0],y_p[:,0])))
print("precision=%f"%(precision_score(y_test[:,0], y_p[:,0])))
print("recall=%f"%(recall_score(y_test[:,0],y_p[:,0])))
print("MCC=%f"%(matthews_corrcoef(y_test[:,0],y_p[:,0]))) 
print(confusion_matrix(y_test[:,0], y_p[:,0]))  
sess.close()                        


