# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:30:02 2019

@author: falcon1
"""
import numpy as np
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import re
import os
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from hmmer import getHomoProteinsByHMMER
######
# To build balanced benchmark dataset which includes same positive and negative protein sequences.
# Because the number of  negative samples is far more than thtat of positive samples, 
# we just use parts of negative samples.
# the benchmark dataset is split three sets: training set, testing set, validating set
# training set includes 70% of all positive and negative sequences respectively
# testing set includes 20% of all positive and negative sequences respectively
# validating set includes 10% of all positive and negative sequences respectively
######
def buildBenchmarkData_vldt(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest, fposvldt, fnegvldt):
    pos_recorders = list(SeqIO.parse(posFile, 'fasta'))
    neg_recorders = list(SeqIO.parse(negFile, 'fasta'))
    
    kf = KFold(n_splits=5, shuffle=True)
    pos_train_ls = []
    pos_test_ls = []
    neg_train_ls = []
    neg_test_ls = []
    for train_indx, test_indx in kf.split(pos_recorders):
        pos_train_ls.append(np.asarray(pos_recorders)[train_indx])
        pos_test_ls.append(np.asarray(pos_recorders)[test_indx])
        
    for train_indx, test_indx in kf.split(neg_recorders):
        neg_train_ls.append(np.asarray(neg_recorders)[train_indx])
        neg_test_ls.append(np.asarray(neg_recorders)[test_indx])
    
    for i in range(5):
        pos_train_recorders = pos_train_ls[i]
        pos_test_recorders = pos_test_ls[i]
        neg_train_recorders = neg_train_ls[i]
        neg_test_recorders = neg_test_ls[i]
        
        pos_train_recorders, pos_vldt_recorders = train_test_split(pos_train_recorders, test_size=0.15)
        neg_train_recorders, neg_vldt_recorders = train_test_split(neg_train_recorders, test_size=0.15)  
        
        SeqIO.write(list(pos_train_recorders), ''.join([fpostrain,'_',str(i)]), 'fasta')
        SeqIO.write(list(neg_train_recorders), ''.join([fnegtrain,'_',str(i)]), 'fasta')
        SeqIO.write(list(pos_test_recorders), ''.join([fpostest,'_',str(i)]), 'fasta')
        SeqIO.write(list(neg_test_recorders), ''.join([fnegtest,'_',str(i)]), 'fasta')
        SeqIO.write(list(pos_vldt_recorders), ''.join([fposvldt,'_',str(i)]), 'fasta')
        SeqIO.write(list(neg_vldt_recorders), ''.join([fnegvldt,'_',str(i)]), 'fasta')

######
# To build balanced benchmark dataset which includes same positive and negative protein sequences.
# Because the number of  negative samples is far more than thtat of positive samples, 
# we just use parts of negative samples.
# the benchmark dataset is split three sets: training set, testing set, validating set
# training set includes 80% of all positive and negative sequences respectively
# testing set includes 20% of all positive and negative sequences respectively
######
def buildBenchmarkData(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest):
    print('generate benchmark data .........##########')
    pos_recorders = list(SeqIO.parse(posFile, 'fasta'))
    neg_recorders = list(SeqIO.parse(negFile, 'fasta'))
    
    kf = KFold(n_splits=5, shuffle=True)
    pos_train_ls = []
    pos_test_ls = []
    neg_train_ls = []
    neg_test_ls = []
    for train_indx, test_indx in kf.split(pos_recorders):
        pos_train_ls.append(np.asarray(pos_recorders)[train_indx])
        pos_test_ls.append(np.asarray(pos_recorders)[test_indx])
        
    for train_indx, test_indx in kf.split(neg_recorders):
        neg_train_ls.append(np.asarray(neg_recorders)[train_indx])
        neg_test_ls.append(np.asarray(neg_recorders)[test_indx])
    
    for i in range(5):
        pos_train_recorders = pos_train_ls[i]
        pos_test_recorders = pos_test_ls[i]
        neg_train_recorders = neg_train_ls[i]
        neg_test_recorders = neg_test_ls[i]
        
        
        SeqIO.write(list(pos_train_recorders), ''.join([fpostrain,'_',str(i)]), 'fasta')
        SeqIO.write(list(neg_train_recorders), ''.join([fnegtrain,'_',str(i)]), 'fasta')
        SeqIO.write(list(pos_test_recorders), ''.join([fpostest,'_',str(i)]), 'fasta')
        SeqIO.write(list(neg_test_recorders), ''.join([fnegtest,'_',str(i)]), 'fasta')
        
def loadBenchmarkData_vldt(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest, fposvldt, fnegvldt, hasBenchmarkData=False, maxlen=1000, k=0):   
    if not hasBenchmarkData:
        buildBenchmarkData_vldt(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest, fposvldt, fnegvldt)
    fpostrain = ''.join([fpostrain,'_',str(k)])
    fnegtrain = ''.join([fnegtrain,'_',str(k)])
    fpostest = ''.join([fpostest,'_',str(k)])
    fnegtest = ''.join([fnegtest,'_',str(k)])
    fposvldt = ''.join([fposvldt,'_',str(k)])
    fnegvldt = ''.join([fnegvldt,'_',str(k)])
    
    x_pos_train = padSequences(fpostrain, maxlen)
    x_neg_train = padSequences(fnegtrain, maxlen)
    x_pos_test = padSequences(fpostest, maxlen)
    x_neg_test = padSequences(fnegtest, maxlen)
    x_pos_vldt = padSequences(fposvldt, maxlen)
    x_neg_vldt = padSequences(fnegvldt, maxlen)
    
    x_train = np.concatenate((x_pos_train, x_neg_train))
    x_test = np.concatenate((x_pos_test, x_neg_test))
    x_vldt = np.concatenate((x_pos_vldt, x_neg_vldt))
    
    y_train = np.zeros((len(x_pos_train) + len(x_neg_train), 2))
    y_train[:len(x_pos_train), 0] = 1
    y_train[len(x_pos_train):, 1] = 1
    
    y_test = np.zeros((len(x_pos_test) + len(x_neg_test), 2))
    y_test[:len(x_pos_test), 0] = 1
    y_test[len(x_pos_test):, 1] = 1
    
    y_vldt = np.zeros((len(x_pos_vldt) + len(x_neg_vldt), 2))
    y_vldt[:len(x_pos_vldt), 0] = 1
    y_vldt[len(x_pos_vldt):, 1] = 1
    
    return (x_train, y_train), (x_test, y_test), (x_vldt, y_vldt)

def loadBenchmarkData(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest, hasBenchmarkData=False, maxlen=1000, k=0):   
    if not hasBenchmarkData:
        buildBenchmarkData(posFile, negFile, fpostrain, fnegtrain, fpostest, fnegtest)
    fpostrain = ''.join([fpostrain,'_',str(k)])
    fnegtrain = ''.join([fnegtrain,'_',str(k)])
    fpostest = ''.join([fpostest,'_',str(k)])
    fnegtest = ''.join([fnegtest,'_',str(k)])
    
    x_pos_train = padSequences(fpostrain, maxlen)
    x_neg_train = padSequences(fnegtrain, maxlen)
    x_pos_test = padSequences(fpostest, maxlen)
    x_neg_test = padSequences(fnegtest, maxlen)
    
    x_train = np.concatenate((x_pos_train, x_neg_train))
    x_test = np.concatenate((x_pos_test, x_neg_test))
    
    y_train = np.zeros((len(x_pos_train) + len(x_neg_train), 2))
    y_train[:len(x_pos_train), 0] = 1
    y_train[len(x_pos_train):, 1] = 1
    
    y_test = np.zeros((len(x_pos_test) + len(x_neg_test), 2))
    y_test[:len(x_pos_test), 0] = 1
    y_test[len(x_pos_test):, 1] = 1
        
    return (x_train, y_train), (x_test, y_test)

def padSequences(protfile, maxlen):
    seqs = []
    for recorder in SeqIO.parse(protfile, 'fasta'):
        seq = str(recorder.seq)
        seq = re.sub('[ZUB]', "", seq)
        seqs.append(seq)
        
    token = Tokenizer(char_level=True)
    token.fit_on_texts(seqs)
    x = token.texts_to_sequences(seqs)
    x = pad_sequences(x, maxlen=maxlen, padding='post', truncating='post')
    return x
        
def prots2onehot(profile, maxlen):
    amino_acid = 'PQRYWTMNVELHSFCIKADG'
    prots = []
    for recorder in SeqIO.parse(profile, 'fasta'):
        seq = str(recorder.seq)
        seq = re.sub('[ZUBX]', "", seq)
        c = len(seq)
        x = np.zeros((maxlen, 20))
        for i in range(c):
            if i == maxlen:
                break
            k = amino_acid.index(seq[i])
            x[i,k] = 1
        prots.append(x)
    return np.array(prots)

def onehot2prot(prot):
    amino_acid = 'PQRYWTMNVELHSFCIKADG'
    seq = []
    for i in range(len(prot)):
        if np.sum(prot[i]) == 0:
            break
        seq.append(amino_acid[np.argmax(prot[i])])
    return "".join(seq)
   
#######
## up sampling.
## search protein by hmmer
#######
def upSampling_seqmatch(num_upsamp, upsamp_file, evalue, fpos, fneg, db_fname):
    swissprot_dict={}
    for record in SeqIO.parse(db_fname, 'fasta'):
        swissprot_dict[record.id] = str(record.seq)    
    
    pos_records = list(SeqIO.parse(fpos, 'fasta'))
    neg_records = list(SeqIO.parse(fneg, 'fasta'))
    records = pos_records + neg_records
    records = shuffle(records)
    
    proteins = []
    i = 0
    for record in records:
        protID = getHomoProteinsByHMMER(record, evalue)
        for p in protID:
            if record.id not in p:
                seq = Seq(swissprot_dict[p], IUPAC.ExtendedIUPACProtein)
                seqrecord = SeqRecord(seq, id=p)
                proteins.append(seqrecord)
                i = i + 1
                if i == num_upsamp:
                    break
        if i == num_upsamp:
            break
    SeqIO.write(proteins, upsamp_file, 'fasta')

## Parameters:
##     num_upsamp: the number of up sampling
##     upsamp_file: the name of the file to save the up-sampling seqeuences
##     fpos: the name of the file from which the raw sequences are token out 
##     vrate: the mutation rate   
def upSampling_pssm(num_upsamp, upsamp_file, fpos, vrate, pssmdir):
    seq_records = list(SeqIO.parse(fpos, 'fasta'))
    N = len(seq_records)
    proteins = []
    i = 0
    inputfile = 'input.fasta'
    if not os.path.exists(pssmdir):
        os.mkdir(pssmdir)   
    while True:
        k = np.random.randint(0, N)
        seq_record = seq_records[k]
        
        pssmfile = os.path.join(pssmdir, seq_record.id + "_pssm.txt")
        # psi-blast output file
        if not os.path.exists(pssmfile):
            # psi-blast input file
            if os.path.exists(inputfile):
                os.remove( inputfile)
            SeqIO.write( seq_record, inputfile, 'fasta')
            # psi-blast    
            psiblast_cline = NcbipsiblastCommandline( query = inputfile, db='Swissprot', evalue=0.001,
                                                     num_iterations=3, out_ascii_pssm=pssmfile)
            stdout,stderr=psiblast_cline() 
            
            # If psi-blast didn't constructe pssm
            if not os.path.exists(pssmfile):
                 continue
        
        pssm = readPSSM(pssmfile)
        prot = genSeq(str(seq_record.seq), pssm, vrate)
        fake_seq = Seq(prot, IUPAC.ExtendedIUPACProtein)
        fake_record = SeqRecord(fake_seq, id='fake'+str(i))
        proteins.append(fake_record)
        
        i = i + 1
        print("{:.2f}% ====> {}/{} finished".format(i/num_upsamp * 100, i, num_upsamp))
        if i == num_upsamp:
            break
    SeqIO.write(proteins, upsamp_file, 'fasta')
        
def readPSSM(pssmfile):
    pssm = []
    with open(pssmfile, 'r') as f:
        count = 0
        for eachline in f:
            count += 1
            if count <= 3:
                continue
            if not len(eachline.strip()):
                break
            line = eachline.split()
            pssm.append(line[22: 42])
    return np.array(pssm)

def genSeq(seq, pssm, vrate):
    amino_acid = 'ARNDCQEGHILKMFPSTWYV'
    n_len_seq = len(seq)
    n_variation = np.random.random()*vrate 
    n_variation = np.ceil(n_len_seq * n_variation)
    vary_position = np.random.choice(n_len_seq, int(n_variation))
    gp = [seq[i] for i in range(n_len_seq)]
    for i in vary_position:
        b = np.zeros(shape=(20))
        b[0] = int(pssm[i,0])
        for j in range(1,20):
            b[j] = b[j-1] + int(pssm[i,j])
        
        p = -1    
        r = np.random.randint(0,100)
        for j in range(20):
            if r <= b[j]:
                p = j
                break
        if p == -1:
            gp[i] = ''
        else:
            gp[i] = amino_acid[p]
        
    return ''.join(gp)
    

def duplicateSample(x, m):
    i = 0
    dx = x
    for i in range(int(m-1)):
        dx = np.concatenate((dx, x))
    if m - int(m) > 0:
        pindx = random.sample(range(len(x)), int((m-int(m)) * len(x)))
        pindx.sort()
        dx = np.concatenate((dx, x[pindx])) 
    dx = shuffle(dx)
    return dx

# generate mulrate*len(fpos) Psu-seqenuce samples
def gen_fake_seq(fpos, mulrate):
    #fpos = 'cytoplasm30_benchmark_pos_train.fa'
    pos_recorders = list(SeqIO.parse(fpos, 'fasta'))
    num_pos = len(pos_recorders)
    vrate = 0.2
    num_upsamp = int(mulrate*num_pos)
    upsamp_file = './data/cytoplasm/fake_pos_{}.fa'.format(num_upsamp)
    upSampling_pssm(num_upsamp, upsamp_file, fpos, vrate, './cytoplasm_pssm')

# generate pssm for each sequence in fname and save in dir, named pssm_dir
def gen_pssm(fname, pssm_dir):    
    #fname = './data/secreted_pos_30_train_0'
    seq_records = list(SeqIO.parse(fname, 'fasta'))
    i = 0
    for seq_record in seq_records:
        inputfile = 'input.fasta'
        if not os.path.exists(pssm_dir):
            os.mkdir(pssm_dir)   
                
        pssmfile = pssm_dir + "/" + seq_record.id + "_pssm.txt"
        # psi-blast output file
        if not os.path.exists(pssmfile):
            # psi-blast input file
            if os.path.exists(inputfile):
                os.remove( inputfile)
            SeqIO.write( seq_record, inputfile, 'fasta')
            # psi-blast    
            psiblast_cline = NcbipsiblastCommandline( query = inputfile, db='Swissprot', evalue=0.001,
                                                     num_iterations=3, out_ascii_pssm=pssmfile)
            stdout,stderr=psiblast_cline() 
            
            i = i + 1
            print("{:.2f}% ====> {}/{} finished".format(i/len(seq_records) * 100, i, len(seq_records)))
        