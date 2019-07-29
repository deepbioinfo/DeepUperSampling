# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:13:40 2019

@author: falcon1
"""
from configparser import ConfigParser
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from semisupLearner_keras import displayMetrics
from semisupLearner_keras import sup_loss, SemisupLearner
from dataHelper import loadBenchmarkData,padSequences
from nets import semisup_net, sup_net
from keras import optimizers

def readConfParam(loc):
    conf = ConfigParser()
    conf.read('conf.ini')
    confParam = {}
    confParam['posFile'] = conf.get(loc, 'posFile')
    confParam['negFile'] = conf.get(loc, 'negFile')
    confParam['fpostrain'] = conf.get(loc, 'fpostrain')
    confParam['fnegtrain'] = conf.get(loc, 'fnegtrain')
    confParam['fpostest'] = conf.get(loc, 'fpostest')
    confParam['fnegtest'] = conf.get(loc, 'fnegtest')
    confParam['fposvldt'] = conf.get(loc, 'fposvldt')
    confParam['fnegvldt'] = conf.get(loc, 'fnegvldt')
    
    confParam['batch_size'] = int(conf.get('netparam', 'batch_size'))
    confParam['epochs'] = int(conf.get('netparam', 'epochs'))
    confParam['patience'] = int(conf.get('netparam', 'patience'))
    confParam['learning_rate'] = float(conf.get('netparam', 'learning_rate'))
    
    confParam['rampup_length'] = int(conf.get('semisupparam', 'rampup_length'))
    confParam['rampdown_length'] = int(conf.get('semisupparam', 'rampdown_length'))
    confParam['learning_rate_max'] = float(conf.get('semisupparam', 'learning_rate_max'))
    confParam['scaled_unsup_weight_max'] = int(conf.get('semisupparam', 'scaled_unsup_weight_max'))
    confParam['gammer'] = float(conf.get('semisupparam', 'gammer'))
    confParam['beita'] = float(conf.get('semisupparam', 'beita'))
    
    confParam['hasBenchmarkData'] = bool(int(conf.get('otherparam', 'hasBenchmarkData')))
    confParam['maxlen'] = int(conf.get('otherparam', 'maxlen'))
 
    return confParam

def supLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam):
    # supervised learning
    model = sup_net()
    model.compile(loss=sup_loss, optimizer=optimizers.Adam(lr=confParam['learning_rate']), metrics=['accuracy'])
    model.fit(x_train, y_train, 
              batch_size=confParam['batch_size'], 
              epochs=confParam['epochs'], 
              validation_split=0.1,
              callbacks=[EarlyStopping(patience=confParam['patience']),
                         ModelCheckpoint(filepath=modelFile,
                                 save_weights_only=True,
                                 save_best_only=False)])
    pred_prob = model.predict(x_test)
    # print predicting metrics
    displayMetrics(y_test, pred_prob, noteInfo, metricsFile) 
    
def semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam):
    # semi-supervised learning
    model = semisup_net() 
    ssparam={}
    ssparam['x_train'] = x_test
    ssparam['y_train'] = [y_test, y_test]
    ssparam['batch_size'] = confParam['batch_size']
    ssparam['epochs'] = confParam['epochs']
    ssparam['patience'] = confParam['patience'] 
    ssparam['rampup_length'] = confParam['rampup_length']
    ssparam['rampdown_length'] = confParam['rampdown_length']
    ssparam['learning_rate_max'] = confParam['learning_rate_max']
    ssparam['scaled_unsup_weight_max'] = confParam['scaled_unsup_weight_max']
    ssparam['gammer'] = confParam['gammer']
    ssparam['beita'] = confParam['beita'],
    ssparam['learning_rate'] = confParam['learning_rate']
    ssl = SemisupLearner(modelFile, model, **ssparam)
    # Train net
    ssl.train()
    # predict
    pred_prob = ssl.predict(x_test)
    # print predicting metrics
    displayMetrics(y_test, pred_prob, noteInfo, metricsFile)   

### main ...
def main(loc):
    #num_upsamp = 0
    is_on_bechmark = True
    is_on_upsamp = False
    is_supervised = False
    is_semisup = True
    confParam = readConfParam(loc)
    print('Generating labels and features...')
    (x_train, y_train), (x_test, y_test)=loadBenchmarkData(confParam['posFile'], confParam['negFile'],
                                                                                 confParam['fpostrain'], confParam['fnegtrain'],
                                                                                 confParam['fpostest'], confParam['fnegtest'],
                                                                                 hasBenchmarkData=confParam['hasBenchmarkData'])
    x_train, y_train = shuffle(x_train, y_train)
    ## learning on bechmark data 
    if is_on_bechmark:                
        if is_supervised:
            noteInfo = '\nOn bechmark dataset, supervised learning predicting result'
            metricsFile = 'supervised_info.txt'
            modelFile = './modelFile/cyto_superModel_benchmark.hdf5'
            supLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam)  
        if is_semisup:
            noteInfo = '\nOn bechmark dataset, semi-supervised learning predicting result'
            metricsFile = 'semisup_info.txt'
            modelFile = './modelFile/cyto_semiSuperModel_benchmark.hdf5'
            semisupLearn(x_train, y_train, x_test, y_test, modelFile, noteInfo, metricsFile, **confParam)
    ## learning on up-sampling 
    if is_on_upsamp:
        for mulrate in [2,3,4,5,6]:
            upsampleFile = './data/cytoplasm/fake_pos_{}.fa'.format(mulrate)
            semisup_modelFile = './modelFile/cyto_semiSupModel_upsample_{}.hdf5'.format(mulrate)
            sup_modelFile = './modelFile/cyto_supModel_upsample_{}.hdf5'.format(mulrate)
            
            x_fake = padSequences(upsampleFile, confParam['maxlen'])
            y_fake = np.zeros((len(x_fake), 2))
            y_fake[:,0] = 1
            x_train_upsamp = np.concatenate((x_train, x_fake))
            y_train_upsamp = np.concatenate((y_train, y_fake))
            x_train_upsamp, y_train_upsamp = shuffle(x_train_upsamp, y_train_upsamp)
            
            if is_supervised:
                noteInfo = '\ngenerate positive sample {}  multiple samples by pssm'.format(mulrate)
                metricsFile = 'supervised_info.txt'
                supLearn(x_train_upsamp, y_train_upsamp, x_test, y_test, sup_modelFile, noteInfo, metricsFile, **confParam)
            if is_semisup:
                noteInfo = '\ngenerate positive sample {}  multiple samples by pssm'.format(mulrate)
                metricsFile = 'semisup_info.txt' 
                semisupLearn(x_train, y_train, x_test, y_test, semisup_modelFile, noteInfo, metricsFile, **confParam)
                
main('cytoplasm')