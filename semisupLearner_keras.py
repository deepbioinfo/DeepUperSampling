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
from SemisupCallback import SemisupCallback
from keras import optimizers

## Define loss functions
def sup_loss(y_true, y_pred):  
    m = K.sum(y_true, axis=-1)
    return  K.switch(K.equal(K.sum(y_true), 0), 0., K.sum(K.categorical_crossentropy(K.tf.boolean_mask(y_true,m), K.tf.boolean_mask(y_pred,m), from_logits=True)) / K.sum(y_true))

def semisup_loss(o1, o2):
    def los(y_true, y_pred):
        return keras.losses.mean_squared_error(o1, o2)
    return los

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
    def __init__(self, modelFile, model, **ssparam):
        self.weight = K.variable(0.)
        self.modelFile = modelFile       
        self.ssparam = ssparam
        layer = model.get_layer('unsupLayer')
        loss2 = semisup_loss(layer.get_output_at(0), layer.get_output_at(1))
        model.compile(loss=[sup_loss, loss2], loss_weights=[1, self.weight],
                     optimizer=optimizers.Adam(lr=ssparam['learning_rate']),  metrics=['accuracy']) 
        self.model = model
    def train(self):
        print('Training...')
        ssCallback = SemisupCallback(self.weight,self.ssparam['rampup_length'], self.ssparam['rampdown_length'], 
                                     self.ssparam['epochs'], self.ssparam['learning_rate_max'], 
                                     self.ssparam['scaled_unsup_weight_max'], self.ssparam['gammer'], 
                                     self.ssparam['beita'])
        if 'x_vldt' in self.ssparam.keys():
            self.model.fit(self.ssparam['x_train'], self.ssparam['y_train'],
                              batch_size=self.ssparam['batch_size'],
                              epochs=self.ssparam['epochs'],
                              validation_data=[self.ssparam['x_vldt'], self.ssparam['y_vldt']],
                              callbacks=[ssCallback,
                                         EarlyStopping(patience=self.ssparam['patience']),
                                         ModelCheckpoint(filepath=self.modelFile,
                                                         save_weights_only=True,
                                                         save_best_only=False)]
            )
        else:
            self.model.fit(self.ssparam['x_train'], self.ssparam['y_train'],
                              batch_size=self.ssparam['batch_size'],
                              epochs=self.ssparam['epochs'],
                              validation_split=0.1,
                              callbacks=[ssCallback,
                                         EarlyStopping(patience=self.ssparam['patience']),
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
    

