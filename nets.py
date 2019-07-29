# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:50:27 2019

@author: falcon1
"""

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers import Conv1D, Bidirectional, Embedding, LSTM
from keras.regularizers import l2
from keras import optimizers
def semisup_net():
    # varibale
    global weight
    #l2value = 0.001
    maxlen = 1000
    ## Constructe model
    x_input = Input(shape=(1000,), dtype='int32', name='main_input')
    x = Embedding(output_dim=20, input_dim=23, input_length=maxlen)(x_input)
    
    # first convolutionary layers
    #x1_1 = Conv1D(20,1, activation='relu', padding="same")(x)
    #x1_2 = Conv1D(20,3, activation='relu', padding="same")(x)
    #x1_3 = Conv1D(20,5, activation='relu', padding="same")(x)
    #x1_4 = Conv1D(20,9, activation='relu', padding="same")(x)
    #x1_5 = Conv1D(20,15, activation='relu', padding="same")(x)
    #x1_6 = Conv1D(20,21, activation='relu', padding="same")(x)
    
    # Concatenate all CNN layers
    #x = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])
    
    # second CNN layer
    x = Conv1D(128,3, activation='relu',  padding="same")(x)
    
    # Dropout
    drop_a = Dropout(rate=0.25)
    x_a = drop_a(x)
    x_b = drop_a(x)
    
    # bidirectional lstm
    #blstm = Bidirectional(LSTM(5))
    blstm = LSTM(5)
    x_a = blstm(x_a)
    x_b = blstm(x_b)
    
    # Dropout
    drop_b = Dropout(0.25, name="unsupLayer")
    x_a = drop_b(x_a)
    x_b = drop_b(x_b)
    
    # output
    out = Dense(2, activation='softmax')(x_a)
    
    model = Model(inputs=x_input, outputs=[out, x_b])
    model.summary()
    
    #loss2 = semisup_loss(x_a, x_b)
    #model.compile(loss=[sup_loss, loss2], loss_weights=[1, weight],
    #              optimizer=optimizers.Adam(lr=learning_rate),  metrics=['accuracy'])  
    
    return model

def sup_net():
    l2value = 0.001
    maxlen = 1000
    x_input = Input(shape=(1000,), dtype='int32', name='main_input')
    x = Embedding(output_dim=20, input_dim=23, input_length=maxlen)(x_input)
    
    # first convolutionary layers
    x1_1 = Conv1D(20,1, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_2 = Conv1D(20,3, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_3 = Conv1D(20,5, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_4 = Conv1D(20,9, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_5 = Conv1D(20,15, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    x1_6 = Conv1D(20,21, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    
    # Concatenate all CNN layers
    x = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6])
    
    # second CNN layer
    x = Conv1D(128,3, activation='relu', kernel_regularizer=l2(l2value), padding="same")(x)
    
    # Dropout
    x = Dropout(rate=0.25)(x)
    
    # bidirectional lstm
    x = Bidirectional(LSTM(10))(x)
    
    # Dropout
    x = Dropout(0.25)(x)
        
    # output
    out = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=x_input, outputs=out)
    model.summary()
    
    # equal to: model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(lr=0.01), metrics=['accuracy'])
    #model.compile(loss=sup_loss, optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    return model