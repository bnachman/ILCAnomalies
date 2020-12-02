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
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from eventHelper import *


#--------------------------- Helper methods
def add_to_padded_part(records):
  array = []
  for record in records:
      # convert to np array
      these_particles = np.array(record['particles']).astype('float')
      # omit index 0
      these_particles = these_particles[:,1:]
      # determine how many zero values to pad
      pad_length = max_nparticles - these_particles.shape[0]
      #pad
      padded_particles = np.pad(these_particles, ((0,pad_length),(0,0)),'constant')
      # check padding
      assert padded_particles.shape == (max_nparticles, 4)
      # add to list
      array.append(padded_particles)

  return array

def add_to_padded_jet(records):
  array = []
  for record in records:
    # convert to np array
    these_jets = np.array(record['jets']).astype('float')
    # omit index 0
    if len(these_jets) == 0:
        these_jets = np.zeros(11).reshape([1,11])
    these_jets = these_jets[:,1:]
        
    # determine how many zero values to pad
    pad_length = max_njets - these_jets.shape[0]
    #pad
    padded_jets = np.pad(these_jets, ((0,pad_length),(0,0)),'constant')
    # check padding
    assert padded_jets.shape == (max_njets, 10)
    # add to list
    array.append(padded_jets)

  return array


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

        this_record['jets']=jets_vec

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

        particles = particles.split()
        particles_vec = []
        for i in range(nparticles):
            particle = np.zeros(5)
            #order:
            # - index
            # - magnitude of momentum |p| (units of GeV)
            # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
            # - azimuthal angle
            # - particle identifier (https://pdg.lbl.gov/2006/reviews/pdf-files/montecarlo-web.pdf)
            particle = particles[i*5:i*5+5]
            particles_vec+=[particle]
            #print(particles[i*5],particles[i*5+1],particles[i*5+2],particles[i*5+3],particles[i*5+4])
        this_record['particles'] = particles_vec
        
        #w,v = momentum_tensor(particles_vec,3)
        #this_record['sphericity'] = sphericity(w,v)
        #this_record['transverse_sphericity'] = transverse_sphericity(w,v)
        #his_record['aplanarity'] = aplanarity(w,v)
        
        all_records.append(this_record)
        #if(len(all_records)) > 5000000: break
    return all_records


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
        padded_evt_arrays.append(np.array(evt_vars).real)
    return np.array(padded_evt_arrays)


#-------------------------------------------------------------------------
if __name__ == "__main__":

  bg_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe*_background.txt")
  #bg_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger*.txt")
  #bg_file_list = ["/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger1.txt","/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger2.txt","/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger3.txt","/data/users/jgonski/Snowmass/LHE_txt_fils/processed_background_randomseeds_bigger4.txt"]
  signal_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe*signal.txt")

  bg_records = []
  for filename in bg_file_list:
      file = open(filename)
      bg_records += parse_file(file)
      #if len(bg_records) > 100: break
  sig_records = []
  for filename in signal_file_list:
      file = open(filename)
      sig_records += parse_file(file)
      #if len(sig_records) > 100: break

  print('Parsed '+str(len(bg_records))+' background events and '+str(len(sig_records))+' signal events....')
  
  print("======================= regular: ", bg_records)
  print("======================= as array: " , np.asarray(bg_records))
  np.save("1202_bg_records_smaller.npy",np.asarray(bg_records))
  np.save("1202_sig_records.npy",np.asarray(sig_records))

  


