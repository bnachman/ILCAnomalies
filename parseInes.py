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

#--------------------------- Parse text files
def parse_file(file_object):
    all_records = []
    mymeasuredenergy = []

    count = 0
    for line in file_object:

        metadata = line.split("J")[0]
        eventinfo = line.split("J")[1]
        jets = eventinfo.split("P")[0]
        particles = eventinfo.split("P")[1]

        this_record = {}
        this_record['label'] = count
        count += 1
        eventweight = float(metadata.split()[0])
        this_record['eventweight'] = eventweight #this is the event "weight".  Let's ignoreit for now (we will need it later).
        njets = int(len(jets.split())/11) #number of "jets"

        nparticles  = int(len(particles.split())/5) #number of particles

        #True collision quantities
        this_record['truthcenterofmassenergy'] = float(metadata.split()[1]) #true total energy - should be delta function at 1000 GeV
        this_record['truthsqrtshat'] = float(metadata.split()[2]) #energy available for making new particles (electron energy - photon)
        this_record['truthphotonpT'] = float(metadata.split()[3]) #photon momentum |p| in units of GeV
        this_record['truthphotoneta'] = float(metadata.split()[4]) #photon pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        this_record['truthphotonphi'] = float(metadata.split()[5]) #photon azimuthal angle

        #Measured collision quantities
        measuredcenterofmassenergy  = float(metadata.split()[6]) #true measured energy - should be noisy version of truthcenterofmassenergy
        this_record['measuredcenterofmassenergy'] = measuredcenterofmassenergy
        this_record['measuredsqrtshat'] = float(metadata.split()[7]) #energy available for making new particles (electron energy - photon)
        this_record['measuredphotonpT'] = float(metadata.split()[8]) #photon momentum |p| in units of GeV
        this_record['measuredphotoneta'] = float(metadata.split()[9]) #photon pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        this_record['measuredphotonphi'] = float(metadata.split()[10]) #photon azimuthal angle
        this_record['metadata'] = metadata.split()

        mymeasuredenergy+=[measuredcenterofmassenergy]

        this_record['njets'] = njets
        jets = jets.split()
        jets_vec = []
        for i in range(njets):
            jet = np.zeros(11)
            #order:
            # - index
            # - magnitude of momentum |p| (units of GeV)
            # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
            # - azimuthal angle
            # - mass (units of GeV/c^2)
            # - bit encoding of the jet "flavor" (not totally sure what the bit means, but will look it up)
            # - 0th angular moment of jet radiation
            # - 1th angular moment of jet radiation
            # - 2th angular moment of jet radiation
            # - 3th angular moment of jet radiation
            # - 4th angular moment of jet radiation
            jet = jets[i*11:i*11+11]
            jets_vec+=[jet]

        #this_record['jets']=jets_vec

        this_record['lny23'] = lny23(jets_vec)
        this_record['total_jet_mass'] = total_jet_mass(jets_vec)
        thrust_maj, thrust_min = thrust(jets_vec)
        this_record['thrust_major'] = thrust_maj
        this_record['thrust_minor'] = thrust_min
        #print("thrust major: ", thrust_maj, ", minor: ", thrust_min)

        w,v = momentum_tensor(jets_vec,2)
        this_record['sphericity'] = sphericity(w,v)
        this_record['transverse_sphericity'] = transverse_sphericity(w,v)
        this_record['aplanarity'] = aplanarity(w,v)

        this_record['nparticles'] = nparticles

        #particles = particles.split()
        #particles_vec = []
        #for i in range(nparticles):
        #    particle = np.zeros(5)
        #    #order:
        #    # - index
        #    # - magnitude of momentum |p| (units of GeV)
        #    # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        #    # - azimuthal angle
        #    # - particle identifier (https://pdg.lbl.gov/2006/reviews/pdf-files/montecarlo-web.pdf)
        #    particle = particles[i*5:i*5+5]
        #    particles_vec+=[particle]
        #    #print(particles[i*5],particles[i*5+1],particles[i*5+2],particles[i*5+3],particles[i*5+4])
        #this_record['particles'] = particles_vec
        
        #w,v = momentum_tensor(particles_vec,3)
        #this_record['sphericity'] = sphericity(w,v)
        #this_record['transverse_sphericity'] = transverse_sphericity(w,v)
        #his_record['aplanarity'] = aplanarity(w,v)
        
        all_records.append(this_record)
        #if(len(all_records)) > 5000000: break
    return all_records


#-------------------------------------------------------------------------
if __name__ == "__main__":

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))


  #dataDir = '/data/users/jgonski/Snowmass/training_npy/'
  #sig_records = np.ndarray.tolist(np.load(dataDir+"1202_sig_records.npy",allow_pickle=True))
  #bg_records = np.ndarray.tolist(np.load(dataDir+"1202_bg_records_smaller.npy",allow_pickle=True))
  #bg_records = np.ndarray.tolist(np.load(dataDir+"1202_bg_records_bigger3.npy",allow_pickle=True))
  bg_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger9.txt")
  bg_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe001_background.txt")
  sig_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe001_signal.txt")


  for filename in bg_file_list:
      file = open(filename)
      bg_records += parse_file(file)
  for filename in sig_file_list:
      file = open(filename)
      sig_records += parse_file(file)

  print('Running over '+str(len(bg_records))+' background events and '+str(len(sig_records))+' signal events....')

  #for i in sig_records:
  #    i['from_anomaly_data'] = True
  #for i in bg_records:
  #    i['from_anomaly_data'] = False

  #all_records = sig_records[:79999] + bg_records
  #all_records = sig_records + bg_records

  # Make some plots 
  #make_var_plots(sig_records,bg_records)

  


  #----------------- ----------
  # # NN training
  #----------------- ----------
  #X = make_evt_arrays(all_records)
  X_bg = make_evt_arrays(bg_records)
  X_sig = make_evt_arrays(sig_records)
  y_bg = np.array([i['truthsqrtshat'] for i in bg_records])
  y_sig = np.array([i['truthsqrtshat'] for i in sig_records])

  np.save("training_data/X_bg", X_bg)
  np.save("training_data/y_bg", y_bg)
  np.save("training_data/X_sig", X_sig)
  np.save("training_data/y_sig", y_sig)

  # Identify signal and side band 
  # 0126 harmonized Ines
  #sb_left = 275
  #sb_right = 425
  #sr_left = 325
  #sr_right = 375

  #y_bg_binary = np.vectorize(binary_side_band)(y_bg)
  #np.unique(y_bg_binary,return_counts = True)

  #side_band_indicator = (y_bg_binary == 0)
  #within_bounds_indicator = (y_bg_binary == 1)
  #s_side_band_indicator = (y_bg_binary == 0)
  #s_within_bounds_indicator = (y_bg_binary == 1)
  ## This is the background data in the SB
  #X_sideband = X_bg[side_band_indicator]
  #y_sideband = y_bg_binary[side_band_indicator]
  ## This is the background data in the SR
  #X_selected = X_bg[within_bounds_indicator]
  #y_selected = y_bg_binary[within_bounds_indicator]
  ## This is the signal yield in the SR
  ##X_allsignal = X_sig[s_within_bounds_indicator]


  ##print('Yields!') 
  ##print('Bkg in SR: ', len(X_selected))
  ##print('Sig in SR: ', len(X_allsignal))

   
  print('runtime: ',datetime.now() - startTime)
