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
#from training import *



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
#def thrust():


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
def prep_and_shufflesplit_data(anomaly_ratio, size_each = 76000, shuffle_seed = 69,
                               train = 0.8, val = 0.2, test_size_each = 5000):
    
    """
    Pre-Data Selection
    """
        
    #how much bg and signal data to take?
    
    anom_size = int(round(anomaly_ratio * size_each))
    bgsig_size = int(size_each - anom_size)

    
    # make sure we have enough data.
    print('Anom size: ', anom_size, ', bgsig size: ', bgsig_size,', size each: ',size_each,', test size each: ', test_size_each) 
    print('X sideband: ', X_sideband.shape)
    print('X selected: ',X_selected.shape)
    print('X sig: ',X_sig.shape)
    assert (size_each <= X_sideband.shape[0])
    assert (anom_size + test_size_each <= X_sig.shape[0])
    assert (bgsig_size + test_size_each <= X_selected.shape[0])
    
    """
    Data Selection
    """
    
    # select sideband datapoints
    this_X_sb = X_sideband[:size_each]
    this_y_sb = np.zeros(size_each) # 0 for bg in SB
    
    # select bgsig datapoints
    this_X_bgsig = X_selected[:bgsig_size]
    this_y_bgsig = np.ones(bgsig_size) # 1 for bg in SR
    
    # select anomaly datapoints
    this_X_sig = X_sig[:anom_size]
    this_y_sig = np.ones(anom_size) # 1 for signal in SR
    
    """
    Shuffle + Train-Val-Test Split (not test set)
    """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_sb, this_X_bgsig, this_X_sig])
    this_y = np.concatenate([this_y_sb, this_y_bgsig, this_y_sig])
    
    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    
    (this_X_tr, this_X_v, _,
     this_y_tr, this_y_v, _) = data_split(this_X, this_y, val=val, test=0)
        
    
    print('Size of sb:')
    print(this_X_sb.shape)
    print('Size of bgsig:')
    print(this_X_bgsig.shape)
    print('Size of sig:')
    print(this_X_sig.shape)
        
    
    """
    Get the test set
    """
    
    # select the data
    this_X_test_P = X_sig[anom_size:anom_size+test_size_each]
    this_X_test_N = X_selected[bgsig_size:bgsig_size+test_size_each]
    
    this_y_test_P = np.ones(test_size_each)
    this_y_test_N = np.zeros(test_size_each)
        
    # Shuffle the combination    
    this_X_te = np.concatenate([this_X_test_P, this_X_test_N])
    this_y_te = np.concatenate([this_y_test_P, this_y_test_N])
    
    this_X_te, this_y_te = shuffle(this_X_te, this_y_te, random_state = shuffle_seed)
    print('Size of test set:')
    print(this_X_te.shape)
    print('Test set distribution:')
    print(np.unique(this_y_te,return_counts = True))
    
    
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
    
    """
    Data processing
    """
    from sklearn import preprocessing
    X_train = preprocessing.scale(X_train)
    X_val = preprocessing.scale(X_val)
    X_test = preprocessing.scale(X_test)
    # Centre and normalize all the Xs
    '''
    for x in X_train:
        #print(x)
        #mask = x[:,0] > 0
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    for x in X_val:
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    for x in X_test:
        yphi_avg = np.average(x, axis=0)
        x -= yphi_avg
        x /= x.sum()
    '''
    # remap PIDs for all the Xs
    #remap_pids(X_train, pid_i=3)
    #remap_pids(X_val, pid_i=3)
    #remap_pids(X_test, pid_i=3)
    
    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('Training set size, distribution:')
    print(X_train.shape)
    print(np.unique(y_train,return_counts = True))
    print('Validations set size, distribution:')
    print(X_val.shape)
    print(np.unique(y_val,return_counts = True))
    print('Test set size, distribution:')
    print(X_test.shape)
    print(np.unique(y_test,return_counts = True))
    
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
        evt_vars = [record['lny23'],record['aplanarity'],record['sphericity']]
        padded_evt_arrays.append(np.array(evt_vars).real)
    return np.array(padded_evt_arrays)


#-------------------------------------------------------------------------
if __name__ == "__main__":

  bg_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe*_background.txt")
  signal_file_list = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/processed_lhe*signal.txt")

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

  X = make_evt_arrays(all_records)
  X_bg = make_evt_arrays(bg_records)
  X_sig = make_evt_arrays(sig_records)
  print(X_sig.shape)
  y_bg = np.array([i['truthsqrtshat'] for i in bg_records])
  y_sig = np.array([i['truthsqrtshat'] for i in sig_records])

  plt.hist(y_sig,label='signal')
  plt.hist(y_bg,label='backgound')
  plt.legend()
  plt.savefig('truthsqrtshat.pdf')
  plt.clf()

  # Identify signal and side band 
  sb_left = 160
  sb_right = 540
  sr_left = 300
  sr_right = 400

  #-----------------------------------------------------------------------------------
  def binary_side_band(y_thing):
      if y_thing >= sr_left and y_thing < sr_right:
          return 1
      elif y_thing >= sb_left and y_thing < sb_right:
          return 0
      else:
          return -1

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


  # ---------------------------- Building the model 

  # network architecture parameters
  dense_sizes = (100, 100)
  # network training parameters
  num_epoch = 100
  batch_size = 100
  
  dnn = DNN(input_dim=3, dense_sizes=dense_sizes, summary=(i==0))

  aucs = []
  rocs = []
  anomalyRatios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.4]
  for r in anomalyRatios:
      X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=r, size_each = 2000, shuffle_seed = 69,train = 0.8, val = 0.2, test_size_each = 100)
      #X_train, X_val, X_test, Y_train,Y_val,Y_test = prep_and_shufflesplit_data(anomaly_ratio=r, size_each = 100, shuffle_seed = 69,train = 0.8, val = 0.2, test_size_each = 50)
      
      dnn.fit(X_train, Y_train,
      epochs=num_epoch,
      batch_size=batch_size,
      validation_data=(X_val, Y_val),
      verbose=0)
      
      
      Y_predict = dnn.predict(X_test)#,batch_size=1000)
      auc = roc_auc_score(Y_test[:,1], Y_predict[:,1])
      #roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1])
      roc_curve = sklearn.metrics.roc_curve(Y_test[:,1], Y_predict[:,1])
      rocs.append(roc_curve)
      aucs.append(auc)

  print(aucs)
  for i,r in enumerate(anomalyRatios):
      plt.plot(rocs[i][0],rocs[i][1],label=r)
  plt.xlabel('fpr')
  plt.ylabel('tpr')
  plt.title('ROC curve')
  plt.legend()
  plt.show()


