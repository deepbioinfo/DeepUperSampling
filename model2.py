# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:10:32 2019

@author: Weizhong Lin
"""

import sys
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
BATCH_SIZE = 100
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

def loss_1(y_true, y_pred):  
    #num = K.int_shape(y_true)[0]
    return K.mean(K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true, axis=-1))

def loss_2(other_pred):
    def mseloss(y_treu, y_pred):
        return mean_squared_error(other_pred, y_pred)
    return mseloss      


def pred_result_accuracy(y_pred, y_true):
    count = 0
    for i in range(len(y_true)):
        if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
            count = count + 1
    print(count/len(y_true))


data = np.load('DataSubset.npz')
x_train, x_test, y_train, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
# built net
inputs = Input(shape=(1000,21,1,))

# first convolutionary layers
x1_1 = Conv2D(20,(1,21), padding="same")(inputs)
x1_2 = Conv2D(20,(3,21), padding="same")(inputs)
x1_3 = Conv2D(20,(5,21), padding="same")(inputs)
x1_4 = Conv2D(20,(9,21), padding="same")(inputs)
x1_5 = Conv2D(20,(15,21), padding="same")(inputs)
x1_6 = Conv2D(20,(21,21), padding="same")(inputs)

# Concatenate all CNN layers
x1 = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])

# second CNN layer
x2 = Conv2D(128,(3,120), padding="same")(x1)

# Dropout
l_drop = Dropout(rate=0.5)(x2)

# Reshape
l_reshape = Reshape((128,-1))(l_drop)
l_reshape = Permute((2,1))(l_reshape)

# BiLSTM  layer
l_bilstm = Bidirectional(LSTM(256))(l_reshape)


l_drop_2 = Dropout(rate=0.5)(l_bilstm)

# output layer
y_pred_1 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)
y_pred_2 = Dense(2, activation='softmax', kernel_regularizer=l2(0.001))(l_drop_2)

model = Model(inputs=inputs, outputs=[y_pred_1, y_pred_2])

alpha = K.variable(1.0)
beta = K.variable(0.0)

model.compile(optimizer='adam', loss=[loss_1, loss_2(other_pred=y_pred_1)], 
              loss_weights=[alpha, beta], metrics=['accuracy'])

print(model.summary())

class CustomValidationLoss(Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, 1.0)
        K.set_value(self.beta, unsupWeight(epoch))
        print(epoch, K.get_value(self.alpha), K.get_value(self.beta))
        
custom_validation_loss = CustomValidationLoss(alpha, beta)
                      
model.fit(x_train, [y_train, y_train], epochs=10, batch_size=BATCH_SIZE,
          verbose=2, callbacks=[custom_validation_loss])

[y_pred_1, y_pred_2] = model.predict( x_test)

pred_result_accuracy(y_pred_1, y_test)

