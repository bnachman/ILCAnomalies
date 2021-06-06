#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IO: do this:
#source activate fullenv
#python -m ipykernel install --user --name fullenv --display-name "fullenv"
# also see this https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
import energyflow as ef
from energyflow.archs import DNN, PFN
#from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical
from keras.models import Sequential
from keras.layers import Dense 
from tensorflow.keras import optimizers
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from sklearn.preprocessing import quantile_transform
from eventHelper import *
from datetime import datetime
#from ROOT import *
import math
#from prep_shufflesplit_jerry import *
from prep_shufflesplit import *
from stacking import *

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
  print('getting files of form ', "training_pfn_data/"+savee+"*X*"+typee+"*.npy")
  X_arr = []
  y_arr=[]
  for s in glob.glob("training_pfn_data/"+savee+"*X*"+typee+"*.npy"):
    X_arr.append(np.load(s))
  for s in glob.glob("training_pfn_data/"+savee+"*y*"+typee+"*.npy"):
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
def fit_model(X_train, Y_train, X_val, Y_val,num_epoch,batch_size,saveTag=''):
      #model = DNN(input_dim=15, dropouts=0.2, dense_sizes=dense_sizes, summary=True)
      #model = DNN(input_dim=int(len(X_sig[0])), dropouts=0.2, dense_sizes=dense_sizes, summary=True)
      opt = optimizers.Adam(learning_rate=0.0001)
      model = PFN(input_dim=X_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes,optimizer=opt)
      h = model.fit(X_train, Y_train,
      epochs=num_epoch,
      batch_size=batch_size,
      validation_data=(X_val, Y_val),
      verbose=0)
      # save model
      filename = 'models/'+saveTag+'.h5'
      model.save(filename)
      print('>Saved %s' % filename)
      return model, h


#---  make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    #Y_predicts = [model.predict(testX) for model in members]
    Y_predicts_scaled = [quantile_transform(model.predict(testX)) for model in members]
    Y_predicts_scaled = np.array(Y_predicts_scaled)
    result = np.average(Y_predicts_scaled,axis=0)
    # sum across ensemble members
    #summed = np.sum(Y_predicts, axis=0)
    # argmax across classes
    #result = np.argmax(summed, axis=1)
    #print('Avg Y_predict!!!!', np.shape(result))
    return result





#-------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--savename", default = '', type=str, nargs='+',
                     help="savename")
  parser.add_argument("-s", "--sizeeach", default = 75000, type=int, nargs='+',
                     help="sizeeach")
  parser.add_argument("-te", "--testset", default = '', type=str, nargs='+',
                     help="testset")
  parser.add_argument("-tr", "--trainset", default = '', type=str, nargs='+',
                     help="trainset")
  parser.add_argument("-e", "--doEnsemble", default = 1, type=int,
                     help="do ensembling")
  parser.add_argument("-r", "--doRandom", default = 0, type=int,
                     help="do random signal init")
  parser.add_argument("-sig", "--signal", default = '350', type=str,
                     help="type of signal run")
  args = parser.parse_args()
  sizeeach = int(args.sizeeach[0])
  savename = args.savename[0]
  testset = args.testset[0]
  trainset = args.trainset[0]
  doEnsemb = args.doEnsemble
  random = args.doRandom
  signal = args.signal
  saveTag = savename+"_"+testset+"_"+trainset

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))
  print('arguments: signal = ', signal, ', sizeeach: ', sizeeach, ', saveTag: ', saveTag, ', testSet: ', testset, ", training: ", trainset, ", doing ensembling ?", doEnsemb)


  # -- Get input files 
  X_bg_arr, y_bg_arr = load_arrs("background",savename.split("_")[0])
  if '350' in signal: X_sig_arr, y_sig_arr = load_arrs("sig",savename.split("_")[0])
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

  #make_var_plots(X_sig,X_bg,saveTag+"_npy")

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
  Phi_sizes, F_sizes = (20, 20, 20), (20,20,20)
  # network training parameters
  num_epoch = 30
  batch_size = 100
  if doEnsemb: n_models=50
  else: n_models=1
  saveTag += 'ep'+str(num_epoch)+"bt"+str(batch_size)+"nm"+str(n_models)
 
  aucs = []
  rocs = []
  sigs=[]
  #anomalyRatios = [0.0, 0.004, 0.008, 0.016, 0.04, 0.12, 1.0]
  sigmas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
  anomalyRatios = get_ars(sigmas,sizeeach)
  sigmas.append('inf')
 
  for r in range(len(anomalyRatios)):
      anom_size = int(round(anomalyRatios[r]* sizeeach)) #amount of sig contamination
      bgsig_size = int(sizeeach - anom_size) #remaining background to get to 100%
      sigs.append(np.round(anom_size/np.sqrt(bgsig_size),3))
      print('S labelled 1s:', anom_size, ", B labelled 1s: ", bgsig_size, ", sig: ", anom_size/np.sqrt(bgsig_size))
      print('-------------- Anomaly Ratio = '+str(anomalyRatios[r]))
      
      # ---- ensembling ! 
      if doEnsemb: 
        print('~~~~~~~~~~~~~~~~~~~~~~ ENSEMBLING ~~~~~~~~~~~~~~~~~~~~~~~~~')
        if random: print('***** WITH RANDOMIZING *******')
        ensembModels = []
        thisAucs = []
        thisRocs = []
        for i in range(n_models):
          perSaveTag = saveTag+str(i)+"_sigma"+str(sigmas[r])
          print('~~~~~~~~~~ MODEL '+str(i)+', perSaveTag='+str(perSaveTag))
          X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(X_selected, X_sideband, X_sig_sr, anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.7, val = 0.2, test=0.1,doRandom=random) 
          model, h = fit_model(X_train, Y_train, X_val, Y_val,num_epoch,batch_size,perSaveTag)
          ensembModels.append(model)
          # do some plotting
          draw_hist(model,X_train,Y_train,X_test,Y_test,"plots/"+perSaveTag)
          plot_loss(h,sigmas[r],"plots/"+perSaveTag) 
          thisYPredict = model.predict(X_test)
          print(' & & & & range of this model prediction on full bkg signal test set!' , np.amax(thisYPredict[:,1])- np.amin(thisYPredict[:,1]))
          if (np.amax(thisYPredict[:,1]) - np.amin(thisYPredict[:,1])) <= 0.04: #2 bins wide at 0.02 bins width
            print('&&&&&&&&&&&&&&&&&&&&&&&&&& A BAD MODEL!')
            plt.hist(thisYPredict[:,1])
            plt.savefig("plots/BROKEN_"+perSaveTag+"_hist.pdf")
            plt.clf()
            i = i-1
            continue
          thisAucs.append(roc_auc_score(Y_test[:,1], thisYPredict[:,1]))
          thisRocs.append(sklearn.metrics.roc_curve(Y_test[:,1], thisYPredict[:,1]))
          make_single_roc(r,'tpr',sklearn.metrics.roc_curve(Y_test[:,1], thisYPredict[:,1]), roc_auc_score(Y_test[:,1], thisYPredict[:,1]),sigmas[r],"plots/"+saveTag+str(i)+"_sigma"+str(sigmas[r]),sizeeach,len(X_sig_sr[0]))

        print('~~~~~~~~~~ AUCs ', thisAucs)
        print('~~~~~~~~~~ mean & std: ', np.mean(thisAucs), np.std(thisAucs))
        plt.hist(thisAucs, np.linspace(0,1.0,50), label=saveTag+': AUCs s='+str(sigmas[r]))
        plt.xlabel("AUCs")
        plt.legend()
        plt.savefig("plots/"+saveTag+"_"+str(sigmas[r])+"_histAUCs.pdf")
        plt.clf()

      else: 
          X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(X_selected, X_sideband, X_sig_sr, anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.7, val = 0.2, test=0.1,doRandom=random) 
          model, h = fit_model(X_train, Y_train, X_val, Y_val,num_epoch,batch_size) 
          draw_hist(model,X_train,Y_train,X_test,Y_test,saveTag+"_sigma"+str(sigmas[r]))
          plot_loss(h,sigmas[r],saveTag) 

       
      # ROCs 
      if not doEnsemb: 
        Y_predict = model.predict(X_test)
      else: Y_predict = ensemble_predictions(ensembModels, X_test)
      auc = roc_auc_score(Y_test[:,1], Y_predict[:,1]) #Y_test = true labels, Y_predict = net determined positive rate
      roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1]) #[fpr,tpr]
      rocs.append(roc_curve)
      aucs.append(auc)


  print('FINAL AUCs: ', aucs)
  if '350' in signal: finalSaveTag = 'Signal (m$_X$ = 350 GeV) vs. background, \n'+get_sqrts_type(savename)
  else: finalSaveTag = 'Signal (m$_X$ = 700 GeV) vs. background, \n'+get_sqrts_type(savename)
  make_roc_plots(anomalyRatios,'TPR',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
  make_roc_plots(anomalyRatios,'TPR/$\sqrt{(FPR)}$',rocs,aucs,sigs,"plots/"+saveTag,finalSaveTag)
  #make_roc_plots(anomalyRatios,'fpr',rocs,aucs,sigs,"plots/"+saveTag)
  #make_roc_plots(anomalyRatios,'tpr/sqrt(fpr)',rocs,aucs,sigs,"plots/"+saveTag)
   
  print('runtime: ',datetime.now() - startTime)
