# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:13:39 2019

@author: Weizhong Lin
"""

import numpy as np

amino_acid = 'PQRYWTMNVELHSFCIKADG'
def seq2mat(seq):
    m = np.zeros(shape=[21,1000])
    i, N = 0, len(seq)
    c = min(1000, N)
    while i < c:
        try:
            k = amino_acid.index(seq[i])
        except ValueError:
            i = i + 1
            continue
        else:
            m[k][i] = 1
            i = i + 1
    return m

def seq2mat_2(seq):
    m = np.zeros(shape=[1000,21])
    i, N = 0, len(seq)
    c = min(1000, N)
    while i < c:
        try:
            k = amino_acid.index(seq[i])
        except ValueError:
            i = i + 1
            continue
        else:
            m[i][k] = 1
            i = i + 1
    return m

def benckmarkDataset(labels, seqs, n):
    data = np.ndarray(shape=[n,21,1000])
    targ = np.ndarray(shape=[n,2,1])
    for i in range(n):
        data[i] = seq2mat(seqs[i])
        
        if labels[i] == 0:
            targ[i] = [[1],[0]]
        else:
            targ[i] = [[0],[1]]
    return data, targ

def benckmarkDataset_2(labels, seqs, n):
    data = np.ndarray(shape=[n,1000,21])
    targ = np.ndarray(shape=[n,2])
    for i in range(n):
        data[i] = seq2mat_2(seqs[i])
        
        if labels[i] == 0:
            targ[i] = [1,0]
        else:
            targ[i] = [0,1]
    return data, targ


def seqData():
    with open('cytoplasm_pos_neg.txt') as fr:
        lns = fr.readlines()
        
    labels=[]
    seqs=[]
    for line in lns:
        ls = line.split()
        labels.append(int(ls[0]))
        seqs.append(ls[2])
    
    return seqs, labels

from sklearn.model_selection import train_test_split
import random

seqs, labels = seqData()
data, targ = benckmarkDataset_2(labels,seqs,10000) # lns.length=74484
x_train, x_test, y_train, y_test = train_test_split(data, targ)
for i in range(len(x_train)):
    if random.random() < 0.3:
        y_train[i] = [0,0]
        
sp = x_train.shape
x_train = np.reshape(x_train,(sp[0],sp[1], sp[2], 1))
np.savez('DataSubset',x_train, x_test, y_train, y_test) 
   
#data, targ = benckmarkDataset(labels,seqs,74484) # lns.length=74484
#np.savez('dataset',data,targ)    