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
import math
import random

#-----------------------------------------------------------------------------------
def get_sig_in_SB(X_sig,X_sig_sr ,X_sig_sb,anom_size):
    print('total sig: ', len(X_sig))
    print('total sig in SR: ', len(X_sig_sr))
    print('total sig in SB: ', len(X_sig_sb))
    print('total sig that I will put in the SR: ', anom_size)
    sigNormalization = np.divide(anom_size,len(X_sig))
    print('which is '+str(sigNormalization)+" of the total signal.")
    print('size of X_sig in the sb: ', len(X_sig_sb))
    sigSB_size = int(sigNormalization * len(X_sig_sb))
    print('so I want to include '+str(sigSB_size)+" signal events in the SB")
    return sigSB_size

#-----------------------------------------------------------------------------------
def print_debug(X_train_b,this_X,title):
      print(title)
      print('X_train_b: ', X_train_b[0:1])
      print('X_train unshuffed = this_X_tr: ', this_X[0:1])

#-----------------------------------------------------------------------------------
def normalize(X_train):
    for x in X_train:
        mask = x[:,0] > 0
        if len(x[mask,0]) < 1: continue
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()

    return X_train

#-----------------------------------------------------------------------------------
def get_sig(X_sig,n_sig,doRandom):
      print("number of signal events :", n_sig, ', random injection? ', doRandom)
      if doRandom: 
        if n_sig == 0: this_X_sr_sig = np.zeros((0, 15, 10))
        else: this_X_sr_sig = np.array(random.sample(list(X_sig), n_sig))
      else: this_X_sr_sig = X_sig[:n_sig]
      
      return this_X_sr_sig 

#-----------------------------------------------------------------------------------
def get_datasets(n_bkg_sb,n_bkg_sr,n_sig_sb,n_sig_sr,X_sideband,X_selected,X_sig_sb,X_sig_sr,train_set,doRandom):

    ###################  CWoLa
    if train_set=='CWoLa': #bg+sig in SB vs. bg+sig in SR
      # SB = 0s 
      this_X_sb_bg =X_sideband[:n_bkg_sb]
      this_y_sb_bg = np.zeros(n_bkg_sb)
      this_X_sb_sig = get_sig(X_sig_sb, n_sig_sb, doRandom)
      this_y_sb_sig = np.zeros(n_sig_sb) 
      
      # SR = 1s
      this_X_sr_bg = X_selected[:n_bkg_sr]
      this_y_sr_bg = np.ones(n_bkg_sr) 
      # select anomaly datapoints = 1s
      this_X_sr_sig = get_sig(X_sig_sr, n_sig_sr, doRandom)
      this_y_sr_sig = np.ones(n_sig_sr) 


    ###################  benchmark
    elif train_set == 'benchmark': #train bg vs. bg+sig in SR 
      # Bkg in SR = 0s
      this_X_sb_bg = X_selected[:n_bkg_sr]
      this_y_sb_bg = np.zeros(n_bkg_sr)
      this_X_sb_sig = []
      this_y_sb_sig = []

      # Bkg in SR = 1s
      this_X_sr_bg = X_selected[n_bkg_sr:2*n_bkg_sr]
      this_y_sr_bg = np.ones(n_bkg_sr) # 1 for bg in SR
      # select anomaly datapoints = 1s
      this_X_sr_sig = get_sig(X_sig_sr, n_sig_sr, doRandom)
      this_y_sr_sig = np.ones(n_sig_sr) 
      

    return this_X_sb_bg,this_y_sb_bg,this_X_sr_bg,this_y_sr_bg,    this_X_sb_sig,this_y_sb_sig,this_X_sr_sig,this_y_sr_sig 



#-----------------------------------------------------------------------------------
def prep_and_shufflesplit_data(X_selected, X_sideband, X_sig_sr, X_sig_sb, X_sig, anomaly_ratio,train_set,test_set, size_each = 25000, shuffle_seed = 69,train = 0.8, val = 0.2, test = 0.1, doRandom=False,debug=False):
   
    """
    Get the number of events and training sets """
    if doRandom: print('--------------------> IMPORTANT: RANDOM = TRUE !') 
    #how much bg and signal data to take?
    #anom_size = int(round(anomaly_ratio * size_each)) #amount of sig contamination
    #bgsig_size = int(size_each - anom_size) #remaining background to get to 100%
    #compute SF: amount to scale bkg hist down 
    n_bkg_sb = size_each
    sf =np.divide(n_bkg_sb, len(X_sideband))
    n_bkg_sr = int(sf*len(X_selected))
    n_sig_sr = int(anomaly_ratio*n_bkg_sr) 
    n_sig_sb = int(sf*get_sig_in_SB(X_sig, X_sig_sr,X_sig_sb,n_sig_sr))
    test_size_each = int(size_each * test)   


    print('SCALED Yields!') 
    print('Bkg in SB: ', n_bkg_sb)
    print('Scale factor: ', sf)
    print('Bkg in SR: ', n_bkg_sr)
    print('Sig in SB: ', n_sig_sb)
    print('Sig in SR: ', n_sig_sr) 
   

    #      get   datasets
    this_X_sb_bg,this_y_sb_bg,this_X_sr_bg,this_y_sr_bg,this_X_sb_sig,this_y_sb_sig,this_X_sr_sig,this_y_sr_sig  = get_datasets(n_bkg_sb,n_bkg_sr,n_sig_sb,n_sig_sr,X_sideband,X_selected,X_sig_sb,X_sig_sr,train_set,doRandom)
    


    #----- just naming conventions
    #this_X_sb = sig +bkg in SB 
    #this_X_bgsig = bkg in SR 
    #this_X_sig_sr = sig in SR
    if n_sig_sb> 0: this_X_sb = np.concatenate([this_X_sb_bg,this_X_sb_sig])
    else: this_X_sb = this_X_sb_bg
    this_y_sb = np.zeros(n_bkg_sb + n_sig_sb) # 0 for bg+sig in SB
    #                             SR contribution,         SB contribution
    if n_sig_sb > 0: X_train_s = np.concatenate([this_X_sb_sig, this_X_sr_sig]) 
    else: X_train_s = this_X_sr_sig
    X_train_b = np.concatenate([this_X_sb_bg, this_X_sr_bg]) 


    #------------- check duplicates
    if debug:
      jets = []
      for e in X_train_b: 
        for j in e:
          if j[0]==0 and j[1] == 0: continue 
          else: 
            jets.append(j)
      vals,inds,cts = np.unique(jets,axis=0,return_index= True, return_counts=True)
      print('###################    duplicate check???? X_train_b: ',len(X_train_b)*15,len(jets),', unique events: ', len(vals)) 


 
    """
    Shuffle + Train-Val-Test Split (not test set) """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_sb, this_X_sr_bg, this_X_sr_sig])
    this_y = np.concatenate([this_y_sb, this_y_sr_bg, this_y_sr_sig])

    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    (this_X_tr, this_X_v, _,this_y_tr, this_y_v, _) = data_split(this_X, this_y, val=val, test=0)
  
    if 'benchmark' in train_set:      
      print('Size of bkg #1 in SR (0s):',this_X_sb.shape)
      print('Size of bkg #2 in SR (1s):',this_X_sr_bg.shape)
      print('Size of sig in SR (1s):',this_X_sr_sig.shape)
    elif 'CWoLa' in train_set:      
      print('Size of bg+sig in SB (0s):',this_X_sb.shape)
      print('Size of bg in SR (1s):',this_X_sr_bg.shape)
      print('Size of sig in SR (1s):',this_X_sr_sig.shape)


    
      
    """
    Get the test set  """ 
    #---  test = truth S vs truth B in SR only 
    if train_set=='CWoLa' and test_set == 'SvsB':
      if doRandom: this_X_test_P = random.sample(list(X_sig_sr), test_size_each)
      else: this_X_test_P = X_sig_sr[n_sig_sr:n_sig_sr+test_size_each] #truth sig 
      this_X_test_N = X_selected[n_bkg_sr:n_bkg_sr+test_size_each] #truth bkg in SR
    #---  test = truth S vs truth B in SR only, benchmark training
    elif train_set=='benchmark' and test_set == 'SvsB':
      if doRandom: this_X_test_P = random.sample(list(X_sig_sr), test_size_each)
      else: this_X_test_P = X_sig_sr[n_sig_sr:n_sig_sr+test_size_each] #truth sig 
      this_X_test_N = X_selected[n_bkg_sb+n_bkg_sr:n_bkg_sb+n_bkg_sr+test_size_each] #truth bkg in SR
    #---  test = bkg sr vs. bkg sb
    elif test_set == 'BvsB':
      this_X_test_P = X_selected[n_bkg_sr:n_bkg_sr+test_size_each] #truth bkg in SR
      this_X_test_N = X_sideband[n_bkg_sb:n_bkg_sb+test_size_each] #sb 

    #labels
    this_y_test_P = np.ones(test_size_each)
    this_y_test_N = np.zeros(test_size_each)
        
    # Shuffle the combination    
    this_X_te = np.concatenate([this_X_test_P, this_X_test_N])
    this_y_te = np.concatenate([this_y_test_P, this_y_test_N])
   
    #ipdb.set_trace() 
    this_X_te, this_y_te = shuffle(this_X_te, this_y_te, random_state = shuffle_seed)
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
   


    """
    Data processing """
    # --------------> PFN 
    # Centre and normalize all the Xs
    X_train = normalize(X_train)
    X_train_b = normalize(X_train_b)
    X_train_s = normalize(X_train_s)
    X_val = normalize(X_val)
    X_test = normalize(X_test)

    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('number of inputs :', X_train.shape[-1])
    print('Training set size, distribution:',X_train.shape)
    print('Validations set size, distribution:',X_val.shape)
    print('Test set size, distribution:',X_test.shape)


    return X_train,X_train_b,X_train_s,X_val,X_test, Y_train,Y_val,Y_test

