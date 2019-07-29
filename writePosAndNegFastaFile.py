# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:35:37 2019

@author: Weizhong Lin
"""
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

with open('Secreted_pos_neg.txt') as fr:
        lns = fr.readlines()
        
pos_seqrecords = []

neg_seqrecords = []

for line in lns:
    ls = line.split()
    s = ls[2]
    if len(s) > 50 and s.count('X') < 5:
        head = ls[1].split(";")
        seq = Seq(s, IUPAC.ExtendedIUPACProtein)
        seqrecord = SeqRecord(seq, id=head[0])
        if int(ls[0]) == 0:
            neg_seqrecords.append(seqrecord)
        else:
            pos_seqrecords.append(seqrecord) 

SeqIO.write(neg_seqrecords, './data/secreted_neg.fa', 'fasta')
SeqIO.write(pos_seqrecords, './data/secreted_pos.fa', 'fasta')

    
    
