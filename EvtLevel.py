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
#import energyflow as ef
#from energyflow.archs import PFN
#from energyflow.datasets import qg_jets
#from energyflow.utils import data_split, remap_pids, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle
from training import *



# # Defining event level variables
# arXiv:1206.2135.pdf

#--------------------------- Variable defs
# In[3]:
def lny23(jets):
    if len(jets) > 2:
        jet1_pt = float(jets[0][1])/np.cosh(float(jets[0][2]))
        jet2_pt = float(jets[1][1])/np.cosh(float(jets[1][2]))
        jet3_pt = float(jets[2][1])/np.cosh(float(jets[2][2]))
        return np.log((jet3_pt*jet3_pt)/((jet1_pt+jet2_pt)*(jet1_pt+jet2_pt)))
    return 0


# In[4]:
def momentum_tensor(jets,r):
    m = np.zeros((3,3))
    totalPSq = 1e-10
    for jet in jets:
    #[index, p [GeV], eta, phi, m]
        pt = float(jet[1]) / np.cosh(float(jet[2]))
        px = pt*np.cos(float(jet[3]))
        py = pt*np.sin(float(jet[3]))
        pz = pt*np.sinh(float(jet[2]))
        #px = float(jet[1])*np.cos(float(jet[3]))/np.cosh(float(jet[2]))
        #py = float(jet[1])*np.sin(float(jet[3]))/np.cosh(float(jet[2]))
        #pz = float(jet[1])*np.sinh(float(jet[2]))/np.cosh(float(jet[2]))
        pr = np.power(float(jet[1]),r-2)
        m += [[px*px*pr, px*py*pr, px*pz*pr], [py*px*pr, py*py*pr, py*pz*pr], [pz*px*pr, pz*py*pr, pz*pz*pr]]
        totalPSq += np.power(float(jet[1]),r)
    #print(totalPSq)
    m = m/totalPSq
    from numpy import linalg as LA
    w, v = LA.eig(m)
    #print("eigenvalues: ", w)
    #print("eigenvectors: ",v)
    return w, v
    #return m  #From this, the sphericity, aplanarity and planarity can be calculated by combinations of eigenvalues.

def sphericity(w,v):
    return (3/2) * (sorted(w)[1]+sorted(w)[2])
def aplanarity(w,v):
    return (3/2) * sorted(w)[2]
def transverse_sphericity(w,v): 
    return (2*sorted(w)[1])/(sorted(w)[0]+sorted(w)[1])


#thrust
def thrust():


#def planarity(w,v):
#    return



#--------------------------- 
def plot_something(var,R,doLog):
    #plt.figure(figsize=(20,5))
    sig_arr = np.array([i[var] for i in sig_records[:79999]])
    bkg_arr = np.array([i[var] for i in bg_records])    
    plt.hist(bkg_arr, R, color="steelblue", histtype='step', linewidth=2,label='Background')
    plt.hist(sig_arr, R, color="tomato", histtype='step', linewidth=2,label='Signal')
    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
    #plt.xscale('log')
    #plt.xticks(R)
    plt.xlabel(var)
    if doLog == True: plt.yscale('log')
    plt.ylabel("Number of Events / bin")
    plt.legend()
    plt.savefig("plt_"+var+".pdf")
    plt.clf()

def plot_jets(index,R,doLog):
    sig_arr = np.array([i['jets'][0][index] for i in sig_records[:79999]])
    #bkg_arr = np.array([i['jets'][0][index] for i in bg_records[:]])    
    plt.hist(bkg_arr, R, color="steelblue", histtype='step', linewidth=2)
    plt.hist(sig_arr, R, color="tomato", histtype='step', linewidth=2)
    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
    #plt.xscale('log')
    #plt.xticks(R)
    plt.xlabel(var)
    if doLog == True: plt.yscale('log')
    plt.ylabel("Number of Events / bin")
    plt.savefig("plt_"+index+".pdf")
    plt.clf()

#--------------------------- Parse text files
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

#From Ben, function to parse files:
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
    return all_records

#-----------------------------------------------------------------------------------
def prep_and_shufflesplit_data(anomaly_ratio, size_each = 5000, shuffle_seed = 69, train = 0.7, val = 0.2, test = 0.1):
  
    print('Starting prep and shuffle split....') 
    print(X_sideband.shape[0], X_sig.shape[0]) 
    assert (size_each <= min(X_sideband.shape[0], X_sig.shape[0]))
    
    #how much bg and signal data to take?
    anom_size = round(anomaly_ratio * size_each)
    bg_sig_size = size_each - anom_size
    
    # select sideband datapoints
    this_X_sideband = X_sideband[:size_each]
    this_y_sideband = y_sideband[:size_each]
    
    # duplicate bgsignal datapoints
    this_X_bgsignal = np.copy(X_bgsignal)
    this_y_bgsignal = np.copy(y_bgsignal)
        
    (this_X_bgsignal, this_X_bgsignal_v, this_X_bgsignal_t,
     this_y_bgsignal, this_y_bgsignal_v, this_y_bgsignal_t) = data_split(this_X_bgsignal, this_y_bgsignal, val=val, test=test)
    
    bg_sig_size_tr = round(bg_sig_size * train)
    
    if this_X_bgsignal.shape[0] < bg_sig_size_tr:
        
        multiplier = math.ceil(bg_sig_size_tr/this_X_bgsignal.shape[0])
        
        this_X_bgsignal = np.concatenate([this_X_bgsignal] * multiplier)
        this_y_bgsignal = np.concatenate([this_y_bgsignal] * multiplier)
        
        this_X_bgsignal_v = np.concatenate([this_X_bgsignal_v] * multiplier)
        this_y_bgsignal_v = np.concatenate([this_y_bgsignal_v] * multiplier)
        
        this_X_bgsignal_t = np.concatenate([this_X_bgsignal_t] * multiplier)
        this_y_bgsignal_t = np.concatenate([this_y_bgsignal_t] * multiplier)
        
        
        
    assert this_X_bgsignal.shape[0] == this_y_bgsignal.shape[0]
    
    #select bgsignal datapoints
    this_X_bgsignal = this_X_bgsignal[:bg_sig_size_tr]
    this_y_bgsignal = this_y_bgsignal[:bg_sig_size_tr]
    
    this_X_bgsignal_v = this_X_bgsignal_v[:round(bg_sig_size * val)]
    this_y_bgsignal_v = this_y_bgsignal_v[:round(bg_sig_size * val)]
    
    this_X_bgsignal_t = this_X_bgsignal_t[:round(bg_sig_size * test)]
    this_y_bgsignal_t = this_y_bgsignal_t[:round(bg_sig_size * test)]
    
    #select anomaly datapoints
    this_X_anom = X_sig[:anom_size]
    this_y_anom = np.ones(anom_size)
    
    
    
    # only bg_sig has been split. Now, we have to shuffle then split the others.
    this_X = np.concatenate([this_X_sideband, this_X_anom])
    this_y = np.concatenate([this_y_sideband, this_y_anom])
    
    assert this_X.shape[0] == this_y.shape[0]
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    (this_X_train, this_X_val, this_X_test,
     this_y_train, this_y_val, this_y_test) = data_split(this_X, this_y, val=val, test=test)
    
    # now, we can add the bg_sig to the rest of the data and shuffle again
    X_train, y_train = shuffle(np.concatenate([this_X_train, this_X_bgsignal]),
                               np.concatenate([this_y_train, this_y_bgsignal]),
                              random_state = shuffle_seed)
    X_val, y_val = shuffle(np.concatenate([this_X_val, this_X_bgsignal_v]),
                               np.concatenate([this_y_val, this_y_bgsignal_v]),
                              random_state = shuffle_seed)
    X_test, y_test = shuffle(np.concatenate([this_X_test, this_X_bgsignal_t]),
                               np.concatenate([this_y_test, this_y_bgsignal_t]),
                              random_state = shuffle_seed)
    
    
    # Centre and normalize all the Xs
    for x in X_train:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    for x in X_val:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    for x in X_test:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
    
    # remap PIDs for all the Xs
    remap_pids(X_train, pid_i=3)
    remap_pids(X_val, pid_i=3)
    remap_pids(X_test, pid_i=3)
    
    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)

    
    return X_train, X_val, X_test, Y_train,Y_val,Y_test

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

  bg_file_list = glob.glob("/data/users/jgonski/snowmass/LHE_txt_fils/processed_lhe*_background.txt")
  signal_file_list = glob.glob("/data/users/jgonski/snowmass/LHE_txt_fils/processed_lhe*signal.txt")

  bg_records = []
  for filename in bg_file_list:
      file = open(filename)
      bg_records += parse_file(file)
      #if len(bg_records) > 300: break
  sig_records = []
  for filename in signal_file_list:
      file = open(filename)
      sig_records += parse_file(file)
      #if len(sig_records) > 300: break

  print('Running over '+str(len(bg_records))+' background files and '+str(len(sig_records))+' signal files....')

  #for i in sig_records:
  #    i['from_anomaly_data'] = True
  #for i in bg_records:
  #    i['from_anomaly_data'] = False

  all_records = sig_records[:79999] + bg_records

  # Make some plots 
  plot_something('truthsqrtshat',range(0,1000,20),1)
  plot_something('lny23',np.linspace(-10,-0.00001,10),1)
  plot_something('transverse_sphericity',np.linspace(0,5,50),1)
  plot_something('sphericity',np.linspace(0,1,50),1)
  plot_something('aplanarity',np.linspace(0,1,50),1)
  
  # didn't validate this plotting function works
  # plot_jets?
  
  #----------------- ----------
  # # NN training
  #----------------- ----------

  #plt.hist([i['nparticles'] for i in sig_records],label='signal')
  #plt.hist([i['nparticles'] for i in bg_records if (i['truthsqrtshat'] > 140 and i['truthsqrtshat'] < 560)],label='backgound')
  #plt.legend()
  #plt.savefig('nparticles.pdf')
  #plt.clf()

  #max_bg = max([i['nparticles'] for i in bg_records])
  #max_sig = max([i['nparticles'] for i in sig_records])
  #max_nparticles = max((max_bg, max_sig))

  ## Save particles only as X
  #X_bg = np.array(add_to_padded_part(bg_records))
  #X_sig = np.array(add_to_padded_part(sig_records))
  #y_bg = np.array([i['truthsqrtshat'] for i in bg_records])
  #y_sig = np.array([i['truthsqrtshat'] for i in sig_records])
  #print(X_bg.shape)
  #print(X_sig.shape)

  ## Save jets as Z
  #print(all_records[0]['jets'])
  #max_njets = max([i['njets'] for i in all_records])
  #Z= np.array(add_to_padded_jet(all_records))
  #y = np.array([i['truthsqrtshat'] for i in all_records])
  #print(Z.shape)
  #print(y.shape)

  ## Identify signal and side band 
  #side_band_left = 160
  #side_band_right = 540
  #signal_left = 300
  #signal_right = 400
  #def binary_side_band(y_thing):
  #    if y_thing >= signal_left and y_thing < signal_right:
  #        return 1
  #    elif y_thing >= side_band_left and y_thing < side_band_right:
  #        return 0
  #    else:
  #        return -1
  #y_binary = np.vectorize(binary_side_band)(y_bg)
  #side_band_indicator = (y_binary == 0) 
  #bg_signal_indicator = (y_binary == 1)
  #X_sideband = X_bg[side_band_indicator]
  #y_sideband = y_binary[side_band_indicator]
  #print(np.unique(y_sideband,return_counts = True))
  #X_bgsignal = X_bg[bg_signal_indicator]
  #y_bgsignal = y_binary[bg_signal_indicator]
  ##X_selected = X_sig[within_bounds_indicator]
  ##y_selected = y_binary[within_bounds_indicator]


  ## Pre processing   
  #print(X_bgsignal[:50000].shape)
  #print(X_sideband.shape)
  #print(np.concatenate([X_bgsignal]*3).shape)
  #print(np.concatenate([y_sideband, y_bgsignal]).shape)

  #X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(0.1)

  #Phi_sizes, F_sizes = (10, 10, 16), (40, 20)
  #num_epoch = 5
  #batch_size = 10


