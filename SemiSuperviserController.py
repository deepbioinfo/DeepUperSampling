# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 13:28:52 2019

@author: falcon1
"""

import math

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
__metaclass__ = type
class SemiSuperviserController:
    def __init__(self, rampup_length, rampdown_length, epochs, learning_rate_max, 
                 scaled_unsup_weight_max, gammer, beita):
        self.rampup_length = rampup_length
        self.rampdown_length = rampdown_length
        self.num_epochs = epochs
        self.learning_rate_max = learning_rate_max
        self.scaled_unsup_weight_max = scaled_unsup_weight_max
        self.gammer=gammer
        self.beita=beita
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