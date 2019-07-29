# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:30:05 2019

@author: falcon1
"""
from SemiSuperviserController import SemiSuperviserController
import keras
from keras import backend as K
import matplotlib.pyplot as plt
__metaclass__ = type
class SemisupCallback(keras.callbacks.Callback):
    def __init__(self, weight,rampup_length, rampdown_length, epochs, learning_rate_max, 
                 scaled_unsup_weight_max, gammer, beita):
        self.weight = weight
        self.ssc = SemiSuperviserController(rampup_length, rampdown_length, epochs, learning_rate_max, 
                 scaled_unsup_weight_max, gammer, beita)
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch': [], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

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
        print("weight:", K.get_value(self.weight))
    
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