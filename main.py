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
from semisupLearner_keras import sup_net, semisup_net,SemisupLearner
from dataHelper import loadBenchmarkData,padSequences
#from dataHelper import duplicateSample

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

### main ...
def main(loc):
    num_upsamp = 0
    is_on_bechmark = False
    is_on_upsamp = True
    is_supervised = False
    is_semisup = True
    confParam = readConfParam(loc)
    
    (x_train, y_train), (x_test, y_test)=loadBenchmarkData(confParam['posFile'], confParam['negFile'],
                                                                                 confParam['fpostrain'], confParam['fnegtrain'],
                                                                                 confParam['fpostest'], confParam['fnegtest'],
                                                                                 hasBenchmarkData=confParam['hasBenchmarkData'])
    x_train, y_train = shuffle(x_train, y_train)
    ## learning on bechmark data 
    if is_on_bechmark:
        print('Generating labels and features...')
                
        if is_supervised:
            # supervised learning
            model = sup_net(confParam['learning_rate'])
            model.fit(x_train, y_train, batch_size=confParam['batch_size'], epochs=confParam['epochs'], 
                      callbacks=[EarlyStopping(patience=confParam['patience']),
                                 ModelCheckpoint(filepath='./modelFile/cyto_superModel_benchmark.hdf5',
                                         save_weights_only=True,
                                         save_best_only=False)])
            pred_prob = model.predict(x_test)
            # print predicting metrics
            noteInfo = '\nOn bechmark dataset, supervised learning predicting result'
            metricsFile = 'supervised_info.txt'
            displayMetrics(y_test, pred_prob, noteInfo, metricsFile)    
        if is_semisup:
            # semi-supervised learning
            model_file = './modelFile/cyto_semiSuperModel_benchmark.hdf5'
            model = semisup_net(confParam['learning_rate']) 
            data={'x_train': x_train, 'y_train': [y_train, y_train]}
            ssl = SemisupLearner(confParam['batch_size'], confParam['epochs'], confParam['patience'], 
                                 confParam['rampup_length'], confParam['rampdown_length'], confParam['learning_rate_max'],
                                 confParam['scaled_unsup_weight_max'], confParam['gammer'], confParam['beita'],
                                 model_file, model, **data)
            # Train net
            ssl.train()
            # predict
            pred_prob = ssl.predict(x_test)
            # print predicting metrics
            noteInfo = '\nOn bechmark dataset, semi-supervised learning predicting result'
            metricsFile = 'semisup_info.txt'
            displayMetrics(y_test, pred_prob, noteInfo, metricsFile)   
    ## learning on up-sampling 
    if is_on_upsamp:
        for mulrate in [2,3,4,5,6]:
            upsampleFile = './data/cytoplasm/fake_pos_{}.fa'.format(mulrate)
            semisup_model_file = './modelFile/cyto_semiSupModel_upsample_{}.hdf5'.format(mulrate)
            sup_model_file = './modelFile/cyto_supModel_upsample_{}.hdf5'.format(mulrate)
            
            x_fake = padSequences(upsampleFile, confParam['maxlen'])
            '''
            x_pos = padSequences(confParam['posFile'], confParam['maxlen'])
            x_fake = duplicateSample(x_pos, num_upsamp)
            '''
            y_fake = np.zeros((len(x_fake), 2))
            y_fake[:,0] = 1
            x_train_upsamp = np.concatenate((x_train, x_fake))
            y_train_upsamp = np.concatenate((y_train, y_fake))
            x_train_upsamp, y_train_upsamp = shuffle(x_train_upsamp, y_train_upsamp)
            
            if is_supervised:
                # supervised learning
                model = sup_net(confParam['learning_rate'])
                model.fit(x_train_upsamp, y_train_upsamp, batch_size=confParam['batch_size'], epochs=confParam['epochs'], 
                          callbacks=[EarlyStopping(patience=confParam['patience']),
                                     ModelCheckpoint(filepath=sup_model_file,
                                         save_weights_only=True,
                                         save_best_only=False)])
                pred_prob = model.predict(x_test)
                # print predicting metrics
                noteInfo = '\nsupervised learning predicting result. generate postive sample {} multiple by pssm'.format(mulrate)
                metricsFile = 'supervised_info.txt'
                displayMetrics(y_test, pred_prob, noteInfo, metricsFile)    
            if is_semisup:
                # semi-supervised learning
                model = semisup_net(confParam['learning_rate']) 
                data={'x_train': x_train_upsamp, 'y_train': [y_train_upsamp, y_train_upsamp]}
                ssl = SemisupLearner(confParam['batch_size'], confParam['epochs'], confParam['patience'], 
                                     confParam['rampup_length'], confParam['rampdown_length'], confParam['learning_rate_max'],
                                     confParam['scaled_unsup_weight_max'], confParam['gammer'], confParam['beita'],
                                     semisup_model_file, model, **data)
                # Train net
                ssl.train()
                # predict
                pred_prob = ssl.predict(x_test)
                # print predicting metrics
                noteInfo = '\ngenerate positive sample {}  multiple samples by pssm'.format(mulrate)
                metricsFile = 'semisup_info.txt'
                displayMetrics(y_test, pred_prob, noteInfo, metricsFile)    
        
                
main('cytoplasm')