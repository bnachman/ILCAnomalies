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
from ROOT import *

#-----------------------------------------------------------------------------------
def load_arrs(typee,savee):
  X_arr = []
  y_arr=[]
  for s in glob.glob("training_data/"+savee+"*X*"+typee+"*.npy"):
    X_arr.append(np.load(s))
  for s in glob.glob("training_data/"+savee+"*y*"+typee+"*.npy"):
    y_arr.append(np.load(s))
  return X_arr, y_arr

#-------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tag", default = '', type=str, nargs='+',
                     help="file name")
  args = parser.parse_args()
  saveTag = args.tag[0]

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))

  X_bg_arr, y_bg_arr = load_arrs("bg",saveTag)
  X_sig_arr, y_sig_arr = load_arrs("sig",saveTag)

  X_bg = np.vstack(X_bg_arr)
  X_sig = np.vstack(X_sig_arr)
  y_bg = np.concatenate(y_bg_arr)
  y_sig = np.concatenate(y_sig_arr)
  print('Running over '+str(len(X_bg))+' background events and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')


  #make_npy_plots(X_sig,X_bg,'total_jet_mass',np.linspace(0,2.0,100),saveTag)
  make_var_plots(X_sig,X_bg,saveTag+"npy")
  make_sqrts_plot(y_sig,y_bg,saveTag+"npy")
   
  print('runtime: ',datetime.now() - startTime)
