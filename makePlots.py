#!/usr/bin/env python
# coding: utf-8

import numpy as np
import glob
from eventHelper import *
from datetime import datetime
from ROOT import *

#-----------------------------------------------------------------------------------
def load_arrs(typee,savee,sample):
  X_arr = []
  y_arr=[]
  dirname = 'FINAL_data'
  print('Getting arrays of type: ', dirname+"/"+savee+"*X*"+typee+"*.npy")
  for s in glob.glob(dirname+"/"+savee+"*X*"+typee+"*.npy"):
    X_arr.append(np.load(s))
  for s in glob.glob(dirname+"/"+savee+"*y*"+typee+"*.npy"):
    y_arr.append(np.load(s))
  return X_arr, y_arr

#-------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tag", default = '', type=str, 
                     help="file name")
  parser.add_argument("-s", "--sample", default = '', type=str,
                     help="pfn or evt")
  args = parser.parse_args()
  saveTag = args.tag
  sample = args.sample

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))

  X_bg_arr, y_bg_arr = load_arrs("bg",saveTag,sample)
  X_sig_arr, y_sig_arr = load_arrs("sig350",saveTag,sample)
  X_sig_arr700, y_sig_arr700 = load_arrs("sig700",saveTag,sample)

  X_bg = np.vstack(X_bg_arr)
  X_sig = np.vstack(X_sig_arr)
  X_sig700 = np.vstack(X_sig_arr700)
  y_bg = np.concatenate(y_bg_arr)
  y_sig = np.concatenate(y_sig_arr)
  y_sig700 = np.concatenate(y_sig_arr700)
  print('Running over '+str(len(X_bg))+' background events and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')


  if 'evt' in sample: make_var_plots(X_sig,X_sig700,X_bg,saveTag+"npy")
  if 'pfn' in sample: make_pfn_plots(X_sig,X_sig700,X_bg,saveTag+"npy")
  make_sqrts_plot(y_sig,y_bg,y_sig700,saveTag+"npy")
   
  print('runtime: ',datetime.now() - startTime)
