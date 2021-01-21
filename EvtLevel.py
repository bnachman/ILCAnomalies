#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IO: do this:
#source activate fullenv
#python -m ipykernel install --user --name fullenv --display-name "fullenv"
# also see this https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
import numpy as np
import matplotlib.pyplot as plt
import glob
import energyflow as ef
from energyflow.archs import DNN
#from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from keras.models import Sequential
from keras.layers import Dense 
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from eventHelper import *

#-----------------------------------------------------------------------------------
def binary_side_band(y_thing):
      if y_thing >= sr_left and y_thing < sr_right:
          return 1
      elif y_thing >= sb_left and y_thing < sb_right:
          return 0
      else:
          return -1

#-----------------------------------------------------------------------------------
def train_models(X_train, X_val, X_test, Y_train,Y_val,Y_test):
    model = PFN(input_dim=X_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)
    history = model.fit(X_train, Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=(X_val, Y_val),
          verbose=1)
    Y_predict = model.predict(X_test)
    
    return (history, Y_test, Y_predict)

#-----------------------------------------------------------------------------------
def make_evt_arrays(these_records):
    padded_evt_arrays =[]
    for i,record in enumerate(these_records):
        #print(i, record)
        # convert to np array
        #these_jets = np.array(record['jets']).astype('float')
        #if len(these_jets) == 0:
        #    these_jets = np.zeros(11).reshape([1,11])
        #these_jets = these_jets[:,6:11] # only want nsubjettiness

        ## determine how many zero values to pad
        #pad_length = max_njets - these_jets.shape[0]
        ##pad_length = 2#max_njets - these_jets.shape[0]
        ##pad
        #padded_jets = np.pad(these_jets, ((0,pad_length),(0,0)))
        ##print(i,pad_length, these_jets.shape[0], padded_jets.shape)
        ## check padding
        #assert padded_jets.shape == (max_njets, 5)
        ## add to list
        #padded_jet_arrays.append(padded_jets)
        evt_vars = [record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['total_jet_mass'],record['thrust_major'],record['thrust_minor']]
        #evt_vars = [record['total_jet_mass']]
        padded_evt_arrays.append(np.array(evt_vars).real)
    return np.array(padded_evt_arrays)

#-----------------------------------------------------------------------------------
def prep_and_shufflesplit_data(anomaly_ratio, size_each = 76000, shuffle_seed = 69,
                               train = 0.8, val = 0.2, test_size_each = 5000):
    
    """
    Pre-Data Selection"""
    #how much bg and signal data to take?
    anom_size = int(round(anomaly_ratio * size_each)) #amount of sig contamination
    bgsig_size = int(size_each - anom_size) #remaining background to get to 100%

    # make sure we have enough data.
    print('Anom size: ', anom_size, ', bgsig size: ', bgsig_size,', size each: ',size_each,', test size each: ', test_size_each) 
    print('Bg in sideband: ', X_sideband.shape) #amount of bkg in SB
    print('Bg in SR: ',X_selected.shape) #amount of bkg in SR
    print('Total sig: ',X_sig.shape) #total signal events
    assert (size_each <= X_sideband.shape[0]) # size each = total data to train in SB
    assert (anom_size + test_size_each <= X_sig.shape[0]) #test_size each = data to train in SR 
    assert (bgsig_size + test_size_each <= X_selected.shape[0]) #test_size each = data to train in SR  
   

 
    """
    Data Selection"""
    # training to separate SB from SR: 0 for all SB events, 1 for all SR events 
    # select sideband datapoints
    this_X_sb = X_sideband[:size_each]
    this_y_sb = np.zeros(size_each) # 0 for bg in SB
    
    # select bg in SR datapoints
    this_X_bgsig = X_selected[:bgsig_size]
    this_y_bgsig = np.ones(bgsig_size) # 1 for bg in SR
    
    # select anomaly datapoints
    this_X_sig = X_sig[:anom_size]
    this_y_sig = np.ones(anom_size) # 1 for signal in SR
   

 
    """
    Shuffle + Train-Val-Test Split (not test set) """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_sb, this_X_bgsig, this_X_sig])
    this_y = np.concatenate([this_y_sb, this_y_bgsig, this_y_sig])
    
    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    (this_X_tr, this_X_v, _,this_y_tr, this_y_v, _) = data_split(this_X, this_y, val=val, test=0)
        
    print('Size of sb (0s):',this_X_sb.shape)
    print('Size of bg in SR (1s):',this_X_bgsig.shape)
    print('Size of sig in SR (1s):',this_X_sig.shape)
    
    
      
    """
    Get the test set  """ 
    #---  test = truth S vs truth B 
    #this_X_test_P = X_sig[anom_size:anom_size+test_size_each] #truth sig 
    #this_X_test_N = X_selected[bgsig_size:bgsig_size+test_size_each] #truth bkg in SR
    #---  test = sr vs. sb
    print(X_sig[anom_size:anom_size+test_size_each])
    print(X_selected[bgsig_size:bgsig_size+test_size_each])
    this_X_test_P = np.concatenate([X_sig[anom_size:anom_size+test_size_each/2], X_selected[bgsig_size:bgsig_size+test_size_each/2]]) #sig and bkg in SR
    this_X_test_N = X_sideband[size_each:size_each+test_size_each] #sb 
    #labels
    this_y_test_P = np.ones(test_size_each)
    this_y_test_N = np.zeros(test_size_each)
        
    # Shuffle the combination    
    this_X_te = np.concatenate([this_X_test_P, this_X_test_N])
    this_y_te = np.concatenate([this_y_test_P, this_y_test_N])
    
    this_X_te, this_y_te = shuffle(this_X_te, this_y_te, random_state = shuffle_seed)
    print('Size of test set:',this_X_te.shape)
    print('Test set distribution:',np.unique(this_y_te,return_counts = True))
       
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
    


    """
    Data processing """
    from sklearn import preprocessing
    X_train = preprocessing.scale(X_train)
    X_val = preprocessing.scale(X_val)
    X_test = preprocessing.scale(X_test)
    # Centre and normalize all the Xs
    '''
    for x in X_train:
        #print(x)
        #mask = x[:,0] > 0
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    for x in X_val:
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    for x in X_test:
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    '''
    # remap PIDs for all the Xs
    #remap_pids(X_train, pid_i=3)
    #remap_pids(X_val, pid_i=3)
    #remap_pids(X_test, pid_i=3)
    
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



#-------------------------------------------------------------------------
if __name__ == "__main__":

  print('hello!')
  dataDir = '/data/users/jgonski/Snowmass/training_npy/'
  sig_records = np.ndarray.tolist(np.load(dataDir+"1202_sig_records.npy",allow_pickle=True))
  bg_records = np.ndarray.tolist(np.load(dataDir+"1202_bg_records_smaller.npy",allow_pickle=True))

  print('Running over '+str(len(bg_records))+' background events and '+str(len(sig_records))+' signal events....')

  #for i in sig_records:
  #    i['from_anomaly_data'] = True
  #for i in bg_records:
  #    i['from_anomaly_data'] = False

  all_records = sig_records[:79999] + bg_records

  # Make some plots 
  make_var_plots(sig_records,bg_records)
  


  #----------------- ----------
  # # NN training
  #----------------- ----------
  X = make_evt_arrays(all_records)
  X_bg = make_evt_arrays(bg_records)
  X_sig = make_evt_arrays(sig_records)
  y_bg = np.array([i['truthsqrtshat'] for i in bg_records])
  y_sig = np.array([i['truthsqrtshat'] for i in sig_records])

  # Identify signal and side band 
  sb_left = 225
  sb_right = 475
  sr_left = 320
  sr_right = 380

  y_bg_binary = np.vectorize(binary_side_band)(y_bg)
  np.unique(y_bg_binary,return_counts = True)

  side_band_indicator = (y_bg_binary == 0)
  within_bounds_indicator = (y_bg_binary == 1)
  # This is the background data in the SB
  X_sideband = X_bg[side_band_indicator]
  y_sideband = y_bg_binary[side_band_indicator]
  # This is the background data in the SR
  X_selected = X_bg[within_bounds_indicator]
  y_selected = y_bg_binary[within_bounds_indicator]





  # ---------------------------- Building the model 

  # network architecture parameters
  dense_sizes = (100, 100)
  # network training parameters
  num_epoch = 100
  batch_size = 100
 
  # dim of however many features we give in X  
  #dnn = DNN(input_dim=int(len(X[0])), dense_sizes=dense_sizes, summary=(i==0),dropouts=0.001,l2_regs=0.005)
  #dnn = DNN(input_dim=int(len(X[0])), dropouts=0.2, dense_sizes=dense_sizes, summary=True)
  #dnn = DNN(input_dim=int(len(X[0])), dense_sizes=dense_sizes, summary=True)

  # by hand 
  #base_model = Sequential()
  #base_model.add(Dense(64, activation='relu', input_dim=len(X[0])))
  #base_model.add(Dense(64, activation='relu'))
  #base_model.add(Dense(3, activation='softmax'))
  #base_model.name = 'Baseline model'
  #base_model.compile(optimizer='adam'
  #                , loss='categorical_crossentropy'
  #                , metrics=['accuracy'])

  aucs = []
  rocs = []
  anomalyRatios = [0.0,0.01, 0.05, 0.1, 0.15, 0.2, 0.4,1.0]
  anomalyRatios = [0.0, 0.05, 0.4, 1.0]
  anomalyRatios = [1.0]
  for r in anomalyRatios:

      dnn = DNN(input_dim=int(len(X[0])), dropouts=0.2, dense_sizes=dense_sizes, summary=True)
      # try skinnier SR
      #X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=r, size_each = 24000, shuffle_seed = 69,train = 0.8, val = 0.2, test_size_each = 2400)
      X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=r, size_each = 1000, shuffle_seed = 69,train = 0.5, val = 0.5, test_size_each = 200)
      print('number of inputs :', len(X[0]))
      print('training input shape: ', np.shape(X_train))
      
      h = dnn.fit(X_train, Y_train,
      epochs=num_epoch,
      batch_size=batch_size,
      validation_data=(X_val, Y_val),
      verbose=0)
      #h = base_model.fit(X_train,Y_train,
      #epochs=num_epoch,
      #batch_size=batch_size,
      #validation_data=(X_val, Y_val),
      #verbose=0)
 
      plot_loss(h) 
       
      # ROCs for SB vs. SR  
      Y_predict = dnn.predict(X_test)
      auc = roc_auc_score(Y_test[:,1], Y_predict[:,1]) #Y_test = true labels, Y_predict = net determined positive rate
      roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1]) #[fpr,tpr]
      rocs.append(roc_curve)
      aucs.append(auc)


  print(aucs)
  for i,r in enumerate(anomalyRatios):
      plt.plot(rocs[i][0],rocs[i][1],label=str(r)+", AUC="+str(np.round(aucs[i],2)))
  plt.xlabel('fpr')
  plt.ylabel('tpr')
  plt.title('ROC curve: truth SR vs. truth SB')
  plt.legend()
  plt.savefig('plots/0120_roc_aucs_srVsSb.pdf')
  #plt.show()


