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
  for s in glob.glob("condor/"+savee+"*X*"+typee+"*.npy"):
    print(s)
    X_arr.append(np.load(s))
  for s in glob.glob("condor/"+savee+"*y*"+typee+"*.npy"):
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
  X_sig_arr, y_sig_arr = load_arrs("signal",mergeTag)

  X_bg = np.vstack(X_bg_arr)
  X_sig = np.vstack(X_sig_arr)
  y_bg = np.concatenate(y_bg_arr)
  y_sig = np.concatenate(y_sig_arr)
  print('Running over '+str(len(X_bg))+' background events and '+str(len(X_sig))+' signal events....')
  print('Running over '+str(len(y_bg))+' background events and '+str(len(y_sig))+' signal events....')

  np.save("training_data/"+mergeTag+"_all_X_bg.npy", X_bg)
  np.save("training_data/"+mergeTag+"_all_X_sig.npy", X_sig)
  np.save("training_data/"+mergeTag+"_all_y_bg.npy", y_bg)
  np.save("training_data/"+mergeTag+"_all_y_sig.npy", y_sig)
   
  print('runtime: ',datetime.now() - startTime)
