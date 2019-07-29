# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:04:40 2019

@author: falcon1
"""
import math
import keras
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

'''
@param  "ssparam" is a dict, for example:
ssparam = {"rampup_length": 30,
                  "rampdown_length": 20,
                  "num_epochs": 100,
                  "learning_rate_max": 0.001,
                  "scaled_unsup_weight_max": 30,
                  "gammer": 5.0,
                  "beita": 0.5} 
'''

class SemiSuperviserController:
    def __init__(self, **ssparam):
        self.rampup_length = ssparam["rampup_length"]
        self.rampdown_length = ssparam["rampdown_length"]
        self.num_epochs = ssparam["num_epochs"]
        self.learning_rate_max = ssparam["learning_rate_max"]
        self.scaled_unsup_weight_max = ssparam["scaled_unsup_weight_max"]
        self.gammer=ssparam["gammer"]
        self.beita=ssparam["beita"]
    def rampup(self, epoch):
        if epoch < self.rampup_length:
            p = 1.0 - float(epoch)/self.rampup_length
            return math.exp(-p * p * self.gammer)
        else:
            return 1.0
    
    def rampdown(self, epoch):
        if epoch >= self.num_epochs - self.rampdown_length:
            ep = (epoch - (self.num_epochs - self.rampdown_length)) * self.beita
            return math.exp(-(ep*ep) / self.rampdown_length)
        return 1.0
    
    def unsupWeight(self, epoch):
        return self.rampup(epoch) * self.scaled_unsup_weight_max
    
    # learning rate
    def learningRate(self, epoch):
        return self.rampup(epoch) * self.rampdown(epoch) * self.learning_rate_max
    
class semisupCallback(keras.callbacks.Callback):
    def __init__(self, weight, epochs, rampup_len, rampdown_len,
                 lr_max, scal_unsup_wm, gammer, beita):
        self.weight = weight
        ssparam = {"rampup_length": rampup_len,
                        "rampdown_length": rampdown_len,
                        "num_epochs": epochs,
                        "learning_rate_max": lr_max,
                        "scaled_unsup_weight_max": scal_unsup_wm,
                        "gammer": gammer,
                        "beita": beita} 
        self.ssc = SemiSuperviserController(**ssparam)
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch': [], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        
    def loss_1(y_true, y_pred):  
        m = K.sum(y_true, axis=-1)
        return  K.switch(K.equal(K.sum(y_true), 0), 0., K.sum(K.categorical_crossentropy(K.tf.boolean_mask(y_true,m), K.tf.boolean_mask(y_pred,m), from_logits=True)) / K.sum(y_true))

    def on_train_begain(self, logs={}):
        self.weight = 0.
        
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc')) 
       
    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc')) 
        
    def on_batch_begin(self, batch, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        K.set_value(self.weight, self.ssc.unsupWeight(epoch))
    
    def  loss_plot(self,loss_type):
        iters=range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters,self.accuracy[loss_type],'r',label='train acc')
        plt.plot(iters,self.losses[loss_type],'g',label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type],'b',label='val acc')
            plt.plot(iters, self.val_loss[loss_type],'k',label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('./figure/acc-loss.png')
        
def plotROC(test,score):
    fpr,tpr,threshold = roc_curve(test, score)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig('./figure/ROC.png')
    
