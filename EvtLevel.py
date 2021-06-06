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
from datetime import datetime
from ROOT import *

#-----------------------------------------------------------------------------------
def get_sqrts_type(saveTag):
  iden = saveTag.split("_")[0]
  print(iden)
  if '041' in iden or 'tru' in iden: return 'truth $\sqrt{\hat{s}}$'
  if '513' in iden: return 'measued $\sqrt{\hat{s}}$ (all hadrons)'
  if '531' in iden: return 'measued $\sqrt{\hat{s}}$ (outgoing photon)'

#-----------------------------------------------------------------------------------
def get_ars(sigmas,sizeeach):
  ars = []
  for sigma in sigmas: 
    sigNum1 = 0.5*(-sigma**2 - np.sqrt(sigma**4 + 4*sigma**2*sizeeach) )
    sigNum2 = 0.5*(np.sqrt(sigma**4 + 4*sigma**2*sizeeach) - sigma**2 )
    if sigNum2/sizeeach > 1.0: ars.append(1.0)
    else: ars.append(sigNum2/sizeeach)
  if 1.0 not in ars: ars.append(1.0)
  return ars

#-----------------------------------------------------------------------------------
def load_arrs(typee,savee):
  print('getting files of form ', "training_data/"+savee+"*X*"+typee+"*.npy")
  X_arr = []
  y_arr=[]
  for s in glob.glob("training_data/"+savee+"*X*"+typee+"*.npy"):
    X_arr.append(np.load(s))
  for s in glob.glob("training_data/"+savee+"*y*"+typee+"*.npy"):
    y_arr.append(np.load(s))
  return X_arr, y_arr

#-----------------------------------------------------------------------------------
def get_sigma_rs(size_each=900):
  goal_sigs = [0.5,1,2,3]
  returnVals = []
  sTry = 0.5
  rTry = 1.0
  for num in goal_sigs:
    print('Goal sig: ', num)
    sigYield = rTry*size_each
    bkgYield = size_each-sigYield
    sigTry=  RooStats.NumberCountingUtils.BinomialExpZ(sigYield,bkgYield,0.3)
    print('test sig: ', sigTry)
    if rTry> num-0.1*num and rTry < num+0.1*num: 
      print('found it! ', rTry)
      returnVals.append(rTry)
      continue
    else: 
      if sigTry > num: rTry -= sTry
      elif sigTry < num: rTry += sTry  
      sTry = sTry  /2 #make interval smaller

  return returnVals

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
def prep_and_shufflesplit_data(anomaly_ratio,train_set,test_set, size_each = 76000, shuffle_seed = 69,
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

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--savename", default = '', type=str, nargs='+',
                     help="savename")
  parser.add_argument("-s", "--sizeeach", default = 15000, type=int, nargs='+',
                     help="sizeeach")
  parser.add_argument("-te", "--testset", default = '', type=str, nargs='+',
                     help="testset")
  parser.add_argument("-tr", "--trainset", default = '', type=str, nargs='+',
                     help="trainset")
  parser.add_argument("-sig", "--signal", default = '350', type=str,
                     help="type of signal run")
  args = parser.parse_args()
  sizeeach = int(args.sizeeach[0])
  savename = args.savename[0]
  testset = args.testset[0]
  trainset = args.trainset[0]
  signal = args.signal
  saveTag = savename+"_"+testset+"_"+trainset

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))
  print('arguments: signal: ', signal, ', sizeeach: ', sizeeach, ', saveTag: ', saveTag, ', testSet: ', testset, ", training: ", trainset)


  # -- Get input files 
  #X_bg_arr, y_bg_arr = load_arrs("bg",savename.split("_")[0])
  #X_sig_arr, y_sig_arr = load_arrs("sig",savename.split("_")[0])
  # -- Get input files
  X_bg_arr, y_bg_arr = load_arrs("bg",savename.split("_")[0])
  if '350' in signal: X_sig_arr, y_sig_arr = load_arrs("signal",savename.split("_")[0])
  elif '700' in signal: X_sig_arr, y_sig_arr = load_arrs("s700",savename.split("_")[0])
  #X_sig_arr, y_sig_arr = load_arrs("s700",savename.split("_")[0])

  X_bg = np.vstack(X_bg_arr)#[:,0:14]
  X_sig = np.vstack(X_sig_arr)#[:,0:14] 
  y_bg = np.concatenate(y_bg_arr)
  y_sig = np.concatenate(y_sig_arr)
  print(np.shape(X_bg))
  print(np.shape(X_sig))
  print('Running over '+str(len(X_bg))+' background events and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')
  
  #-- rmove nans 
  #for a,b in zip(X_bg_arr, y_bg_arr): #each file
  #  for y in range(len(a)): # each event 
  #    for z in a[y]: #each var 
  #      if z!=z:
  #        print('found one!')
  #        print(a[y])
  #        np.delete(a, y)
  #        np.delete(b, y)
  #print('AFTER NANS: running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')


  #make_var_plots(X_sig,X_bg,saveTag+"_npy")

  # --  Identify signal and side band 
  # 0126 harmonized Ines
  #sb_left = 275
  #sb_right = 425
  #sr_left = 325
  #sr_right = 375
  # --  Identify signal and side band
  if '350' in signal:
    sb_left = 275
    sb_right = 425
    sr_left = 325
    sr_right = 375
    print('350::::: SB=',sb_left,sb_right,", SR=",sr_left,sr_right)
  elif '700' in signal:
    sb_left = 625
    sb_right = 775
    sr_left = 675
    sr_right = 725
    print('700::::: SB=',sb_left,sb_right,", SR=",sr_left,sr_right)

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
  # This is the signal yield in the SR
  y_sig_binary = np.vectorize(binary_side_band)(y_sig)
  np.unique(y_sig_binary,return_counts = True)
  s_side_band_indicator = (y_sig_binary == 0)
  s_within_bounds_indicator = (y_sig_binary == 1)
  X_sig_sr = X_sig[s_within_bounds_indicator]
  X_sig_sb = X_sig[s_side_band_indicator]


  print('Yields!') 
  print('Bkg in SR: ', len(X_selected))
  print('Bkg in SB: ', len(X_sideband))
  print('Sig in SR: ', len(X_sig_sr))
  print('Sig in SB: ', len(X_sig_sb))




  # ---------------------------- Building the model 

  # network architecture parameters
  dense_sizes = (100, 100)
  # network training parameters
  num_epoch = 500
  batch_size = 500
 
  aucs = []
  rocs = []
  sigs=[]
  sigmas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
  anomalyRatios = get_ars(sigmas,sizeeach)
  sigmas.append('inf')
 
  for r in range(len(anomalyRatios)):
      anom_size = int(round(anomalyRatios[r]* sizeeach)) #amount of sig contamination
      bgsig_size = int(sizeeach - anom_size) #remaining background to get to 100%
      sigs.append(np.round(anom_size/np.sqrt(bgsig_size),3))
      print('S labelled 1s:', anom_size, ", B labelled 1s: ", bgsig_size, ", sig: ", anom_size/np.sqrt(bgsig_size))

      print('-------------- Anomaly Ratio = '+str(anomalyRatios[r]))
      dnn = DNN(input_dim=int(len(X_sig[0])), dropouts=0.2, dense_sizes=dense_sizes, summary=True)
      # try skinnier SR
      X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.5, val = 0.5, test_size_each = int(np.divide(sizeeach,2)))
      #X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=r, train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.5, val = 0.5, test_size_each = 200)
      print('number of inputs :', len(X_sig[0]))
      print('training input shape: ', np.shape(X_train))
      
      h = dnn.fit(X_train, Y_train,
      epochs=num_epoch,
      batch_size=batch_size,
      validation_data=(X_val, Y_val),
      verbose=0)
      filename = 'evt_models/'+saveTag+'_'+str(sigmas[r])+'_model.h5'
      dnn.save(filename)
      print('>Saved %s' % filename)
 
      plot_loss(h,sigmas[r],saveTag) 
       
      # ROCs for SB vs. SR  
      Y_predict = dnn.predict(X_test)
      auc = roc_auc_score(Y_test[:,1], Y_predict[:,1]) #Y_test = true labels, Y_predict = net determined positive rate
      roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1]) #[fpr,tpr]
      make_single_roc(r,'tpr',sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1]), roc_auc_score(Y_test[:,1], Y_predict[:,1]),sigmas[r],"plots/"+saveTag+"_sigma"+str(sigmas[r]),sizeeach,len(X_sig_sr[0]))

      rocs.append(roc_curve)
      aucs.append(auc)

  print(aucs)
  if '350' in signal: finalSaveTag = 'Signal (m$_X$ = 350 GeV) vs. background, \n'+get_sqrts_type(savename)
  else: finalSaveTag = 'Signal (m$_X$ = 700 GeV) vs. background, \n'+get_sqrts_type(savename)
  make_roc_plots(anomalyRatios,'TPR',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
  make_roc_plots(anomalyRatios,'TPR/$\sqrt{(FPR)}$',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
  #make_roc_plots(anomalyRatios,'tpr',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
  #make_roc_plots(anomalyRatios,'tpr/sqrt(fpr)',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
   
  print('runtime: ',datetime.now() - startTime)
