#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IO: do this:
#source activate fullenv
#python -m ipykernel install --user --name fullenv --display-name "fullenv"
# also see this https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
import numpy as np
import glob
from eventHelper import *
from datetime import datetime

#-----------------------------------------------------------------------------------
def load_arrs(typee,savee):
  print(typee,savee)
  X_arr = []
  y_arr=[]
  bg_globs = ["bigger1_noZ_0to50000.npy","bigger1_noZ_50000to100000.npy","bigger1_noZ_100000to150000.npy","bigger1_noZ_150000to200000.npy","bigger1_noZ_200000to250000.npy","bigger1_noZ_250000to300000.npy","bigger1_noZ_300000to350000.npy","bigger1_noZ_350000to400000.npy","bigger1_noZ_400000to450000.npy","bigger1_noZ_450000to500000.npy","bigger1_noZ_500000to550000.npy","bigger1_noZ_550000to600000.npy","bigger1_noZ_600000to650000.npy","bigger1_noZ_650000to700000.npy","bigger1_noZ_700000to750000.npy","bigger1_noZ_750000to800000.npy","bigger1_noZ_800000to850000.npy","bigger1_noZ_850000to900000.npy","bigger1_noZ_900000to950000.npy","bigger1_noZ_950000to1000000.npy","bigger1_noZ_1000000to1050000.npy","bigger1_noZ_1050000to1100000.npy","bigger1_noZ_1100000to1150000.npy","bigger1_noZ_1150000to1200000.npy","bigger1_noZ_1200000to1250000.npy","bigger1_noZ_1250000to1300000.npy","bigger1_noZ_1300000to1350000.npy","bigger1_noZ_1350000to1400000.npy","bigger1_noZ_1400000to1450000.npy","bigger1_noZ_1450000to1500000.npy"]  
  for g in bg_globs:
    for s in glob.glob("training_pfn_data/"+savee+"*X*"+typee+"*"+g):
      print(s)
      X_arr.append(np.load(s))
    for s in glob.glob("training_pfn_data/"+savee+"*y*"+typee+"*"+g):
      print(s)
      y_arr.append(np.load(s))
  return X_arr, y_arr

#-------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tag", default = '', type=str, nargs='+',
                     help="file name")
  args = parser.parse_args()
  mergeTag = args.tag[0]

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))

  X_bg_arr, y_bg_arr = load_arrs("background",mergeTag)
  #X_sig_arr, y_sig_arr = load_arrs("signal",mergeTag)

  X_bg = np.vstack(X_bg_arr)
  #X_sig = np.vstack(X_sig_arr)
  y_bg = np.concatenate(y_bg_arr)
  #y_sig = np.concatenate(y_sig_arr)
  print('Running over '+str(len(X_bg))+' background events ')#and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events ')#and '+str(len(y_sig))+' signal events....')

  np.save("training_pfn_data/"+mergeTag+"_X_lumifix_bg.npy", X_bg)
  #np.save("training_pfn_data/"+mergeTag+"_all_X_sig.npy", X_sig)
  np.save("training_pfn_data/"+mergeTag+"_y_lumifix_bg.npy", y_bg)
  #np.save("training_pfn_data/"+mergeTag+"_all_y_sig.npy", y_sig)
   
  print('runtime: ',datetime.now() - startTime)
