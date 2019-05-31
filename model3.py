# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:22:00 2019

@author: falcon1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:10:32 2019

@author: Weizhong Lin
"""

import sys
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Conv1D, Concatenate, Bidirectional, LSTM
from keras.layers import Dropout, TimeDistributed, Reshape, Embedding
from keras.regularizers import l2
from keras.models import Model
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import math
import numpy as np
from sklearn.models import train_test_split

HIGH = 21
WIDTH = 1000
rampup_length = 80
rampdown_length = 50
num_epochs = 300
BATCH_SIZE = 100
learning_rate_max = 0.003
scaled_unsup_weight_max = 100
l2value = 0.001

def load_Data():
    with open('cytoplasm_pos_neg.txt') as fr:
        lns = fr.readlines()
        
    labels=[]
    seqs=[]
    for line in lns:
        ls = line.split()
        labels.append(int(ls[0]))
        seqs.append(ls[2])
    
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x=token.texts_to_sequences(seqs)
    x=pad_sequences(x,maxlen=1000,padding='post',truncating='post')
    
    n = len(seqs)
    y = np.ndarray(shape=[n,2])
    for i in range(n):
        if labels[i] == 0:
            y[i] = [1,0]
        else:
            y[i] = [0,1]
    
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    return x,y

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
      

def buildNet(x_train,y_train):
    # built net
    main_input = Input(shape=(1000,), dtype='int32', name='main_input')
    x = Embedding(output_dim=20, input_dim=23, input_length=1000)(main_input)
    
    # first convolutionary layers
    x1_1 = Conv1D(20,1, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_2 = Conv1D(20,3, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_3 = Conv1D(20,5, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_4 = Conv1D(20,9, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_5 = Conv1D(20,15, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_6 = Conv1D(20,21, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    
    # Concatenate all CNN layers
    x1 = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])
    
    # second CNN layer
    x2 = Conv1D(128,3, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x1)
    
    # Dropout
    l_drop = Dropout(rate=0.5)(x2)
      
    # BiLSTM  layer
    l_bilstm = Bidirectional(LSTM(256))(l_drop)
    
    
    l_drop_2 = Dropout(rate=0.5)(l_bilstm)
    
    # output layer
    y_pred_1 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)
    y_pred_2 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)
    
    model = Model(inputs=main_input, outputs=[y_pred_1, y_pred_2])
    
    def loss_1(y_true, y_pred):  
        num = K.int_shape(y_true)[0]
        return K.sum(K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true, axis=-1))/num
    
    def loss_2(y_true, y_pred):
        return mean_squared_error(y_pred_1, y_pred)
    
    alpha = K.variable(1.0)
    beta = K.variable(0.0)
    model.compile(optimizer='adam', loss=[loss_1, loss_2], 
                  loss_weights=[alpha, beta], metrics=['accuracy'])
    
    print(model.summary())
    
    class CustomValidationLoss(Callback):
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta
            
        def on_epoch_end(self, epoch, logs={}):
            self.alpha = 1.0
            self.beta = unsupWeight(epoch)
            print(epoch, K.get_value(self.alpha), K.get_value(self.beta))
            optim = Adam(lr=learningRate(epoch))
            model.compile(optimizer=optim,
                          loss=[loss_1, loss_2],
                          loss_weights=[self.alpha, self.beta],
                          metrics=['accuracy'])
            sys.stdout.flush()
            
    custom_validation_loss = CustomValidationLoss(alpha, beta)
                          
    model.fit(x_train, [y_train, y_train], epochs=10, batch_size=BATCH_SIZE,
              verbose=2, callbacks=[custom_validation_loss])
    
    
    
    return model


def pred_result_accuracy(y_pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
            count = count + 1
    print(count/len(y_true))


x,y = load_Data()
x_train, x_test, y_train, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        
model = buildNet(x_train, y_train)
[y_pred_1, y_pred_2] = model.predict(x_test)
pred_result_accuracy(y_pred_1, y_test)

