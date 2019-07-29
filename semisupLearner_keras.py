# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:49:08 2019

@author: falcon1
"""

import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
from semisupHelper import semisupCallback
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.layers import Conv1D, Bidirectional, Embedding, LSTM
from keras.regularizers import l2
from keras import optimizers

# Global varibale
weight = K.variable(0.)

## Define loss functions
def sup_loss(y_true, y_pred):  
    m = K.sum(y_true, axis=-1)
    return  K.switch(K.equal(K.sum(y_true), 0), 0., K.sum(K.categorical_crossentropy(K.tf.boolean_mask(y_true,m), K.tf.boolean_mask(y_pred,m), from_logits=True)) / K.sum(y_true))

def semisup_loss(o1, o2):
    def los(y_true, y_pred):
        return keras.losses.mean_squared_error(o1, o2)
    return los
def semisup_net(learning_rate):
    # varibale
    global weight
    l2value = 0.001
    maxlen = 1000
    ## Constructe model
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
    drop_a = Dropout(rate=0.25)
    x_a = drop_a(x)
    x_b = drop_a(x)
    
    # bidirectional lstm
    blstm = Bidirectional(LSTM(10))
    x_a = blstm(x_a)
    x_b = blstm(x_b)
    
    # Dropout
    drop_b = Dropout(0.25)
    x_a = drop_b(x_a)
    x_b = drop_b(x_b)
    
    # output
    out = Dense(2, activation='softmax')(x_a)
    
    model = Model(inputs=x_input, outputs=[out, x_b])
    model.summary()
    
    loss2 = semisup_loss(x_a, x_b)
    model.compile(loss=[sup_loss, loss2], loss_weights=[1, weight],
                  optimizer=optimizers.Adam(lr=learning_rate),  metrics=['accuracy'])  
    
    return model

def sup_net(learning_rate):
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
    model.compile(loss=sup_loss, optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    return model

## print and save metrics of model
## Params:
##    y_true: true label
##    predicted_Probability: predicted probability of label
##    noteinfo: the note messages what are wrote into file
##    metricsFile: a string of the file name to saving predicting metrics
def displayMetrics(y_true, predicted_Probability, noteinfo, metricsFile):
        prediction = np.array(predicted_Probability> 0.5).astype(int)
        prediction = prediction[:,0]
        labels = y_true[:,0]
        print('Showing the confusion matrix')
        cm=confusion_matrix(labels,prediction)
        print(cm)
        print("ACC: %f "%accuracy_score(labels,prediction))
        print("F1: %f "%f1_score(labels,prediction))
        print("Recall: %f "%recall_score(labels,prediction))
        print("Pre: %f "%precision_score(labels,prediction))
        print("MCC: %f "%matthews_corrcoef(labels,prediction))
        print("AUC: %f "%roc_auc_score(labels,predicted_Probability[:,0]))
        with open(metricsFile,'a') as fw:
            fw.write(noteinfo + '\n')
            for i in range(2):
                fw.write(str(cm[i,0]) + "\t" +  str(cm[i,1]) + "\n" )
            fw.write("ACC: %f "%accuracy_score(labels,prediction))
            fw.write("\nF1: %f "%f1_score(labels,prediction))
            fw.write("\nRecall: %f "%recall_score(labels,prediction))
            fw.write("\nPre: %f "%precision_score(labels,prediction))
            fw.write("\nMCC: %f "%matthews_corrcoef(labels,prediction))
            fw.write("\nAUC: %f\n "%roc_auc_score(labels,predicted_Probability[:,0]))
        #print('Plotting the ROC curve...')
        #plotROC(labels,predicted_Probability[:,0])
        
class SemisupLearner:
    def __init__(self, batch_size, epochs, patience, rampup_len, rampdown_len,
                 lr_max, scal_unsup_wm, gammer, beita, modelFile, model, **data):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.modelFile = modelFile
        self.model = model
        self.data = data
        self.rampup_len = rampup_len
        self.rampdown_len = rampdown_len
        self.lr_max = lr_max
        self.scal_unsup_wm = scal_unsup_wm
        self.gammer = gammer
        self.beita = beita
    def train(self):
        print('Training...')
        global weight
        ssCallback = semisupCallback(weight, self.epochs, self.rampup_len, self.rampdown_len,
                 self.lr_max, self.scal_unsup_wm, self.gammer, self.beita)
        x_train = self.data['x_train']
        y_train = self.data['y_train']
        if 'x_vldt' in self.data.keys():
            self.model.fit(x_train, y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              validation_data=[self.data['x_vldt'], self.data['y_vldt']],
              callbacks=[ssCallback,
                         EarlyStopping(patience=5),
                         ModelCheckpoint(filepath=self.modelFile,
                                         save_weights_only=True,
                                         save_best_only=False)]
            )
        else:
            self.model.fit(x_train, y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              validation_split=0.1,
              callbacks=[ssCallback,
                         EarlyStopping(patience=self.patience),
                         ModelCheckpoint(filepath=self.modelFile,
                                         save_weights_only=True,
                                         save_best_only=False)]
            )
                         
    def predict(self, x_test):
        print('Evaluating the model')
        predictedList = self.model.predict(x_test)
        return predictedList[0]   
    
    # predict sample's label probability by loading pretraining model
    def load_and_predict(self, x_test):
        self.model.load_weights(self.modelFile)
        return self.predict(x_test)
    

