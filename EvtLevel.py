#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import glob
import energyflow as ef
from energyflow.archs import DNN
from energyflow.utils import data_split, remap_pids, to_categorical
from keras.models import Sequential
from keras.layers import Dense 
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from eventHelper import *
from datetime import datetime
from ROOT import *
from prep_shufflesplit import *


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
  print('getting files of form ', "lumifix_data/"+savee+"*X*"+typee+"*.npy")
  X_arr = []
  y_arr=[]
  for s in glob.glob("lumifix_data/"+savee+"*X*"+typee+"*.npy"):
    X_arr.append(np.load(s))
  for s in glob.glob("lumifix_data/"+savee+"*y*"+typee+"*.npy"):
    y_arr.append(np.load(s))
  return X_arr, y_arr

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




#-------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--savename", default = '', type=str, 
                     help="savename")
  parser.add_argument("-s", "--sizeeach", default = 25000, type=int, 
                     help="sizeeach")
  parser.add_argument("-te", "--testset", default = '', type=str,
                     help="testset")
  parser.add_argument("-tr", "--trainset", default = '', type=str,
                     help="trainset")
  parser.add_argument("-sig", "--signal", default = '350', type=str,
                     help="type of signal run")
  args = parser.parse_args()
  sizeeach = args.sizeeach
  savename = args.savename
  testset = args.testset
  trainset = args.trainset
  signal = args.signal
  saveTag = savename+"_"+testset+"_"+trainset

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))
  print('arguments: signal: ', signal, ', sizeeach: ', sizeeach, ', saveTag: ', saveTag, ', testSet: ', testset, ", training: ", trainset)


  # -- Get input files
  X_bg_arr, y_bg_arr = load_arrs("run_lhe",savename.split("_")[0])
  if '350' in signal: X_sig_arr, y_sig_arr = load_arrs("signal_fixed",savename.split("_")[0])
  elif '700' in signal: X_sig_arr, y_sig_arr = load_arrs("signal_700_fixed",savename.split("_")[0])

  X_bg = np.vstack(X_bg_arr)
  X_sig = np.vstack(X_sig_arr)
  y_bg = np.concatenate(y_bg_arr)
  y_sig = np.concatenate(y_sig_arr)
  print(np.shape(X_bg))
  print(np.shape(X_sig))
  print('Running over '+str(len(X_bg))+' background events and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')

  # --  Identify signal and side band 
  sb_left, sb_right, sr_left, sr_right = get_region_defs(signal,savename.split("_")[0],dowide)

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
  print('Bkg in SB: ', len(X_sideband))
  print('Bkg in SR: ', len(X_selected))
  print('Sig in SB: ', len(X_sig_sb))
  print('Sig in SR: ', len(X_sig_sr))
  print('total sig :', len(X_sig))


  # ---------------------------- Building the model 

  # network architecture parameters
  dense_sizes = (100, 100)
  # network training parameters
  num_epoch = 30
  batch_size = 100
 
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
      X_train,X_train_b,X_train_s,X_val,X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(savename.split("_")[0],signal, X_selected, X_sideband, X_sig_sr,X_sig_sb,X_sig, anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.7, val = 0.2, test=0.1,doRandom=random)
      #X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.5, val = 0.5, test_size_each = int(np.divide(sizeeach,2)))
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
 
      plot_loss(h,sigmas[r],"plots/"+saveTag) 
       
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
   
  print('runtime: ',datetime.now() - startTime)
