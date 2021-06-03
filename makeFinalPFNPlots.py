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
from eventHelper import *
from datetime import datetime
#from ROOT import *
import math
from prep_shufflesplit import *
from PFNLevel import *
from keras import models

#-----------------------------------------------------------------------------------
def binary_side_band(y_thing):
      if y_thing >= sr_left and y_thing < sr_right:
          return 1
      elif y_thing >= sb_left and y_thing < sb_right:
          return 0
      else:
          return -1


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
  saveTag = savename.split("_")[2]

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




  aucs = []
  rocs = []
  sigs=[]
  sigmas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
  sigmas=[5.0]
  anomalyRatios = get_ars(sigmas,sizeeach)
  #sigmas.append('inf')
 
  for r in range(len(sigmas)):
    anom_size = int(round(anomalyRatios[r]* sizeeach)) #amount of sig contamination
    bgsig_size = int(sizeeach - anom_size) #remaining background to get to 100%
    sigs.append(np.round(anom_size/np.sqrt(bgsig_size),3))
    print('S labelled 1s:', anom_size, ", B labelled 1s: ", bgsig_size, ", sig: ", anom_size/np.sqrt(bgsig_size))
    print('-------------- Anomaly Ratio = '+str(anomalyRatios[r]))
    #--------------------------------------------
    # -- Get pre-saved models 
    #--------------------------------------------
    if '350' in signal: 
      modelList = glob.glob('models/*'+savename.split("_")[1]+'_SvsB*sigma'+str(sigmas[r])+"*")
      n_models = len(modelList)
      print('Making plot with ', n_models, ', models: ','models/*'+savename.split("_")[1]+'_SvsB*sigma'+str(sigmas[r])+"*",': ', modelList)
    else: 
      modelList = glob.glob('models/*'+savename.split("_")[1]+'*s700*sigma'+str(sigmas[r])+"*")
      n_models = len(modelList)
      print('Making plot with ', n_models, ', models: ','models/*'+savename.split("_")[1]+'*s700*sigma'+str(sigmas[r])+"*",': ', modelList)
    n_models = 2

    thisAucs = []
    thisRocs = []
    ensembModels=[]
    for i in range(n_models):
      print('~~~~~~~~~~ MODEL '+str(i))
      X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(X_selected, X_sideband, X_sig_sr, anomaly_ratio=anomalyRatios[r], train_set=trainset, test_set=testset, size_each=sizeeach, shuffle_seed = 69,train = 0.7, val = 0.2, test=0.1,doRandom=random) 

      #model, h = fit_model(X_train, Y_train, X_val, Y_val,num_epoch,batch_size,saveTag,i)
      model = models.load_model(modelList[i])
      ensembModels.append(model)

      # do some plotting
      draw_hist(model,X_train,Y_train,X_test,Y_test,saveTag+str(i)+"_sigma"+str(sigmas[r]))
      thisYPredict = model.predict(X_test)
      #thisAucs.append(roc_auc_score(Y_test[:,1], thisYPredict[:,1]))
      #thisRocs.append(sklearn.metrics.roc_curve(Y_test[:,1], thisYPredict[:,1]))
      thisAucs.append(roc_auc_score(Y_test[:,0], thisYPredict[:,0]))
      thisRocs.append(sklearn.metrics.roc_curve(Y_test[:,0], thisYPredict[:,0]))
      make_single_roc(r,'tpr',sklearn.metrics.roc_curve(Y_test[:,1], thisYPredict[:,1]), roc_auc_score(Y_test[:,1], thisYPredict[:,1]),sigmas[r],saveTag+str(i)+"_sigma"+str(sigmas[r]),sizeeach,len(X_sig_sr[0]))

    print('~~~~~~~~~~ AUCs ', thisAucs)
    print('~~~~~~~~~~ mean & std: ', np.mean(thisAucs), np.std(thisAucs))
    plt.hist(thisAucs, np.linspace(0,1.0,50), label=saveTag+': AUCs s='+str(sigmas[r]))
    plt.xlabel("AUCs")
    plt.legend()
    plt.savefig("plots/"+saveTag+"_"+str(sigmas[r])+"_histAUCs.pdf")
    plt.clf()

       
    # ROCs 
    Y_predict = ensemble_predictions(ensembModels, X_test)
    #auc = roc_auc_score(Y_test[:,1], Y_predict[:,1]) #Y_test = true labels, Y_predict = net determined positive rate
    #roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1]) #[fpr,tpr]
    auc = roc_auc_score(Y_test[:,0], Y_predict[:,0]) #Y_test = true labels, Y_predict = net determined positive rate
    roc_curve = sklearn.metrics.roc_curve(Y_test[:,0], Y_predict[:,0]) #[fpr,tpr]
    rocs.append(roc_curve)
    aucs.append(auc)


  print('FINAL AUCs: ', aucs)
  make_roc_plots(anomalyRatios,'fpr',rocs,aucs,sigs,saveTag,sizeeach,len(X_sig_sr[0]))
  make_roc_plots(anomalyRatios,'tpr/sqrt(fpr)',rocs,aucs,sigs,saveTag,sizeeach,len(X_sig_sr[0]))
   
  print('runtime: ',datetime.now() - startTime)
