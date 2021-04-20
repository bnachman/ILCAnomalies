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
def prep_and_shufflesplit_data_jerry(X_selected, X_sideband, X_sig, anomaly_ratio, size_each = 50, shuffle_seed = 69,
                               train = 0.7, val = 0.2, test = 0.1, special_test = False):
    
    assert (size_each <= min(X_sideband.shape[0], X_sig.shape[0]))
    
    
    
    
    #how much bg and signal data to take?
    anom_size = int(anomaly_ratio * size_each)
    bg_sig_size = size_each - anom_size
    
    
    # select sideband datapoints
    this_X_sideband = X_sideband[:size_each]
    this_y_sideband = np.zeros(size_each)
    
    # duplicate bgsignal datapoints
    this_X_bgsignal = shuffle(np.copy(X_selected))
    this_y_bgsignal = np.ones(this_X_bgsignal.shape[0])
        
    (this_X_bgsignal, this_X_bgsignal_v, this_X_bgsignal_t,
     this_y_bgsignal, this_y_bgsignal_v, this_y_bgsignal_t) = data_split(this_X_bgsignal, this_y_bgsignal, val=val, test=test)
    
    bg_sig_size_tr = int(bg_sig_size * train)
    
        
    multiplier = math.ceil(bg_sig_size_tr/this_X_bgsignal.shape[0])
    print('multiplier;', multiplier)

    #this_X_bgsignal = np.concatenate([this_X_bgsignal] * float(multiplier))[:bg_sig_size_tr]
    this_X_bgsignal = this_X_bgsignal[:bg_sig_size_tr]
    this_y_bgsignal = np.ones(bg_sig_size_tr)

    #this_X_bgsignal_v = np.concatenate([this_X_bgsignal_v] * float(multiplier))[:round(bg_sig_size * val)]
    this_X_bgsignal_v = this_X_bgsignal_v[:int(bg_sig_size * val)]
    this_y_bgsignal_v = np.ones(int(bg_sig_size * val))
    
    #if special_test:
    #    test_size_each = int(test * size_each)
    #    multiplier = math.ceil(test_size_each/len(this_X_bgsig_t))
    #    this_X_bgsignal_t = np.concatenate([this_X_bgsignal_t] * multiplier)[:test_size_each]
    #    this_y_bgsignal_t = None
    #    
    #else:
    test_size_each = None
    this_X_bgsignal_t = this_X_bgsignal_t[:int(bg_sig_size * test)]
    this_y_bgsignal_t = np.ones(int(bg_sig_size * test))
        
    print('Size of sigbg (Train, Val, Test):')
    print(this_X_bgsignal.shape, this_X_bgsignal_v.shape, this_X_bgsignal_t.shape)
        
    
#     #select bgsignal datapoints
#     this_X_bgsignal = this_X_bgsignal[:bg_sig_size_tr]
    
#     this_X_bgsignal_v = this_X_bgsignal_v[:round(bg_sig_size * val)]
    
#     this_X_bgsignal_t = this_X_bgsignal_t[:round(bg_sig_size * test)]

    
    #select anomaly datapoints
    this_X_anom = X_sig[:anom_size]
    this_y_anom = np.ones(anom_size)
    
    print('Size of anomalies (pre-splitting):')
    print(this_X_anom.shape)

    
    # only bg_sig has been split. Now, we have to shuffle then split the others.
    this_X = np.concatenate([this_X_sideband, this_X_anom])
    this_y = np.concatenate([this_y_sideband, this_y_anom])
    
    assert this_X.shape[0] == this_y.shape[0]
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    (this_X_train, this_X_val, this_X_test,
     this_y_train, this_y_val, this_y_test) = data_split(this_X, this_y, val=val, test=test)
    
    # make sure there is enough real anomalies to do the special test
    if special_test:
        assert len(X_sig) > anom_size + test_size_each
        X_special_negative = this_X_bgsignal_t
        X_special_positive = X_sig[anom_size:anom_size + test_size_each]
        assert X_special_negative.shape == X_special_positive.shape
        y_special_negative = np.zeros(test_size_each)
        y_special_positive = np.ones(test_size_each)
        
    
    # now, we can add the bg_sig to the rest of the data and shuffle again
    X_train, y_train = shuffle(np.concatenate([this_X_train, this_X_bgsignal]),
                               np.concatenate([this_y_train, this_y_bgsignal]),
                              random_state = shuffle_seed)
    X_val, y_val = shuffle(np.concatenate([this_X_val, this_X_bgsignal_v]),
                               np.concatenate([this_y_val, this_y_bgsignal_v]),
                              random_state = shuffle_seed)
    
    if special_test:
        X_test, y_test = shuffle(np.concatenate([X_special_negative, X_special_positive]),
                                   np.concatenate([y_special_negative, y_special_positive]),
                                  random_state = shuffle_seed)
    else:    
        X_test, y_test = shuffle(np.concatenate([this_X_test, this_X_bgsignal_t]),
                                   np.concatenate([this_y_test, this_y_bgsignal_t]),
                                  random_state = shuffle_seed)
    
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
    
    # remap PIDs for all the Xs
    remap_pids(X_train, pid_i=3)
    remap_pids(X_val, pid_i=3)
    remap_pids(X_test, pid_i=3)
    
    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('Training set size, distribution:')
    print(X_train.shape)
    print(np.unique(y_train,return_counts = True))
    print('Validations set size, distribution:')
    print(X_val.shape)
    print(np.unique(y_val,return_counts = True))
    print('Test set size, distribution:')
    print(X_test.shape)
    print(np.unique(y_test,return_counts = True))
    
    return X_train, X_val, X_test, Y_train,Y_val,Y_test


