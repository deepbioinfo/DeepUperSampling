# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:34:50 2019

@author: falcon1
"""

import numpy as np
from Bio import SeqIO
import os

seqs_test = list(SeqIO.parse('data/DeepLoc/processed_deeploc_test_seq','fasta'))
seqs_train = list(SeqIO.parse('data/DeepLoc/processed_deeploc_train_seq','fasta'))


labels_test = np.zeros(shape=(len(seqs_test), 10))
with open('data/DeepLoc/processed_deeploc_test_label','r') as fr:
    i = 0
    for line in fr.readlines():
        line = line.replace("\n", "")
        j = int(float(line))
        labels_test[i][j] = 1
        i = i + 1
    
labels_train = np.zeros((len(seqs_train), 10))
with open('data/DeepLoc/processed_deeploc_train_label','r') as fr:
    i = 0
    for line in fr.readlines():
        line = line.replace("\n","")
        p = line.split(";")
        if len(p) == 1:
            j = int(float(p[0]))
            labels_train[i][j] = 1
        else:
            for s in p:
                j = int(float(s))
                labels_train[i][j] = 1
        i = i + 1

locs = ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell-membrane','Endoplasmic-reticulum',
        'Plastid', 'Golgi-apparatus','Lysosome-Vacuole', 'Peroxisome'] 
           
num_trains = len(labels_train)
T = [[] for i in range(11)]  
for k in range(num_trains):
    if labels_train[k,0] == 1 and labels_train[k,1] == 0:
        T[0].append(seqs_train[k])
    elif labels_train[k,0] == 0 and labels_train[k,1] == 1:
        T[1].append(seqs_train[k])
    elif labels_train[k,0] == 1 and labels_train[k,1] == 1:
        T[10].append(seqs_train[k])
    for i in range(2,10):
        if labels_train[k,i] == 1:
            T[i].append(seqs_train[k])
# node 1
s1_pos = []
for i in [2,4,8,5,7]:
    s1_pos.extend(T[i])
s1_neg = []
for i in [0,1,3,6,9,10]:
    s1_neg.extend(T[i])    
SeqIO.write(s1_pos, os.path.join('data', 'DeepLoc', 's1-train-pos'), 'fasta')
SeqIO.write(s1_neg, os.path.join('data', 'DeepLoc', 's1-train-neg'), 'fasta')    

# node 2
s2_pos = []
for i in [4,8,5,7]:
    s2_pos.extend(T[i])
s2_neg = T[2]
SeqIO.write(s2_pos, os.path.join('data', 'DeepLoc', 's2-train-pos'), 'fasta')
SeqIO.write(s2_neg, os.path.join('data', 'DeepLoc', 's2-train-neg'), 'fasta') 

# node 3
s3_pos = []
s3_pos.extend(T[4])
s3_pos.extend(T[8])
s3_neg = [] 
s3_neg.extend(T[5])
s3_neg.extend(T[7])  
SeqIO.write(s3_pos, os.path.join('data', 'DeepLoc', 's3-train-pos'), 'fasta')
SeqIO.write(s3_neg, os.path.join('data', 'DeepLoc', 's3-train-neg'), 'fasta') 

# node 4
SeqIO.write(T[4], os.path.join('data', 'DeepLoc', 's4-train-pos'), 'fasta')
SeqIO.write(T[8], os.path.join('data', 'DeepLoc', 's4-train-neg'), 'fasta') 

# node 5
SeqIO.write(T[5], os.path.join('data', 'DeepLoc', 's5-train-pos'), 'fasta')
SeqIO.write(T[7], os.path.join('data', 'DeepLoc', 's5-train-neg'), 'fasta') 

# node 6
s6_pos = []
for i in [0,1,9,10]:
    s6_pos.extend(T[i])
s6_neg = []
s6_neg.extend(T[3])
s6_neg.extend(T[6])
SeqIO.write(s6_pos, os.path.join('data', 'DeepLoc', 's6-train-pos'), 'fasta')
SeqIO.write(s6_neg, os.path.join('data', 'DeepLoc', 's6-train-neg'), 'fasta') 

# node 7
s7_neg = []
for i in [0,1,10]:
    s7_neg.extend(T[i])
SeqIO.write(T[9], os.path.join('data', 'DeepLoc', 's7-train-pos'), 'fasta')
SeqIO.write(s7_neg, os.path.join('data', 'DeepLoc', 's7-train-neg'), 'fasta') 

# node 8
SeqIO.write(T[3], os.path.join('data', 'DeepLoc', 's8-train-pos'), 'fasta')
SeqIO.write(T[6], os.path.join('data', 'DeepLoc', 's8-train-neg'), 'fasta') 

# node 9
SeqIO.write(T[0], os.path.join('data', 'DeepLoc', 's9-train-pos'), 'fasta')
SeqIO.write(T[1], os.path.join('data', 'DeepLoc', 's9-train-neg'), 'fasta') 






  