import numpy as np
import matplotlib.pyplot as plt
import glob
import energyflow as ef
from energyflow.archs import DNN, PFN
#from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from keras.models import Sequential
from keras.layers import Dense 
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from eventHelper import *
from datetime import datetime
from ROOT import *
import math

#-----------------------------------------------------------------------------------
#def prep_and_shufflesplit_data(anomaly_ratio,train_set,test_set, size_each = 76000, shuffle_seed = 69,
#                               train = 0.8, val = 0.2, test_size_each = 5000):
def prep_and_shufflesplit_data(X_selected, X_sideband, X_sig, anomaly_ratio,train_set,test_set, size_each = 76000, shuffle_seed = 69,
                               train = 0.8, val = 0.2, test = 0.1):
    
    #how much bg and signal data to take?
    anom_size = int(round(anomaly_ratio * size_each)) #amount of sig contamination
    bgsig_size = int(size_each - anom_size) #remaining background to get to 100%


    # select sideband datapoints
    if train_set=='CWoLa':
      this_X_sb = X_sideband[:size_each]
      this_y_sb = np.zeros(size_each) # 0 for bg in SB
      
      # select bg in SR datapoints
      this_X_bgsig = X_selected[:bgsig_size]
      this_y_bgsig = np.ones(bgsig_size) # 1 for bg in SR
      
      # select anomaly datapoints
      this_X_sig = X_sig[:anom_size]
      this_y_sig = np.ones(anom_size) # 1 for signal in SR
  
    # 0128 benchmark
    # select bg in SR datapoints
    elif train_set == 'benchmark': #train bg vs. bg+sig in SR 
      #print('# inputs of X: ', len(X_selected[0]))
      this_X_sb= X_selected[:size_each]
      this_y_sb = np.zeros(size_each) # 0 for bg in SR
      
      # select anomaly datapoints
      this_X_sig = X_sig[:anom_size]
      this_y_sig = np.ones(anom_size) # 1 for signal in SR
   
      # select bg in SR datapoints
      this_X_bgsig = X_selected[size_each:size_each+bgsig_size]
      this_y_bgsig = np.ones(bgsig_size) #1 for other bg in SR
   
    #import ipdb
    #ipdb.set_trace()

 
    """
    Shuffle + Train-Val-Test Split (not test set) """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_sb, this_X_bgsig, this_X_sig])
    this_y = np.concatenate([this_y_sb, this_y_bgsig, this_y_sig])
    
    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    (this_X_tr, this_X_v, _,this_y_tr, this_y_v, _) = data_split(this_X, this_y, val=val, test=0)
  
    if 'benchmark' in train_set:      
      print('Size of bkg #1 in SR (0s):',this_X_sb.shape)
      print('Size of bkg #2 in SR (1s):',this_X_bgsig.shape)
      print('Size of sig in SR (1s):',this_X_sig.shape)
    elif 'CWoLa' in train_set:      
      print('Size of bg in SB (0s):',this_X_sb.shape)
      print('Size of bg in SR (1s):',this_X_bgsig.shape)
      print('Size of sig in SR (1s):',this_X_sig.shape)
    
    
      
    """
    Get the test set  """ 
    #---  test = truth S vs truth B in SR only 
    #test_size_each = test*size_each
    test_size_each = int(bgsig_size * test)   

    if train_set=='CWoLa' and test_set == 'SvsB':
      this_X_test_P = X_sig[anom_size:anom_size+test_size_each] #truth sig 
      this_X_test_N = X_selected[bgsig_size:bgsig_size+test_size_each] #truth bkg in SR
    #---  test = mixed sig + bkg in sr vs. bkg sb
    #this_X_test_P = np.concatenate([X_sig[anom_size:anom_size+test_size_each/2], X_selected[bgsig_size:bgsig_size+test_size_each/2]]) #sig and bkg in SR
    #this_X_test_N = X_sideband[size_each:size_each+test_size_each] #sb 
    #---  test = bkg sr vs. bkg sb
    elif test_set == 'BvsB':
      this_X_test_P = X_selected[bgsig_size:bgsig_size+test_size_each] #truth bkg in SR
      this_X_test_N = X_sideband[size_each:size_each+test_size_each] #sb 
    #---  test = truth S vs truth B in SR only, benchmark training
    elif train_set=='benchmark' and test_set == 'SvsB':
      this_X_test_P = X_sig[anom_size:anom_size+test_size_each] #truth sig 
      this_X_test_N = X_selected[size_each+bgsig_size:size_each+bgsig_size+test_size_each] #truth bkg in SR

    #labels
    this_y_test_P = np.ones(test_size_each)
    this_y_test_N = np.zeros(test_size_each)
        
    # Shuffle the combination    
    this_X_te = np.concatenate([this_X_test_P, this_X_test_N])
    this_y_te = np.concatenate([this_y_test_P, this_y_test_N])
   
    #ipdb.set_trace() 
    this_X_te, this_y_te = shuffle(this_X_te, this_y_te, random_state = shuffle_seed)
    print('Size of test set:',this_X_te.shape)
    print('Test set distribution:',np.unique(this_y_te,return_counts = True))
       
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
    


    """
    Data processing """
    #from sklearn import preprocessing
    #X_train = preprocessing.scale(X_train)
    #X_val = preprocessing.scale(X_val)
    #X_test = preprocessing.scale(X_test)

    # --------------> PFN 
    # Centre and normalize all the Xs
    for x in X_train:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    for x in X_val:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    for x in X_test:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    # Centre and normalize all the Xs
    #for x in X_train:
    #    #print(x)
    #    #mask = x[:,0] > 0
    #    yphi_avg = np.average(x, axis=0)
    #    x -= yphi_avg
    #    x /= x.sum()
    #for x in X_val:
    #    yphi_avg = np.average(x, axis=0)
    #    x -= yphi_avg
    #    x /= x.sum()
    #for x in X_test:
    #    yphi_avg = np.average(x, axis=0)
    #    x -= yphi_avg
    #    x /= x.sum()


     #remap PIDs for all the Xs
    remap_pids(X_train, pid_i=3)
    remap_pids(X_val, pid_i=3)
    remap_pids(X_test, pid_i=3)
    
    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('Training set size, distribution:',X_train.shape)
    #print(np.unique(y_train,return_counts = True))
    print('Validations set size, distribution:',X_val.shape)
    #print(np.unique(y_val,return_counts = True))
    print('Test set size, distribution:',X_test.shape)
    #print(np.unique(y_test,return_counts = True))
    
    return X_train, X_val, X_test, Y_train,Y_val,Y_test

