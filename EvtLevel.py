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



# # Defining event level variables
# arXiv:1206.2135.pdf

#--------------------------- Variable defs
# In[3]:
def y23(jets):
    if len(jets) > 2:
        jet1_pt = float(jets[0][1])/np.cosh(float(jets[0][2]))
        jet2_pt = float(jets[1][1])/np.cosh(float(jets[1][2]))
        jet3_pt = float(jets[2][1])/np.cosh(float(jets[2][2]))
        return (jet3_pt*jet3_pt)/((jet1_pt+jet2_pt)*(jet1_pt+jet2_pt))
    return 0


# In[4]:
def momentum_tensor(jets,r):
    m = np.zeros((3,3))
    totalPSq = 1e-10
    for jet in jets:
        px = float(jet[1])*np.cos(float(jet[3]))/np.cosh(float(jet[2]))
        py = float(jet[1])*np.sin(float(jet[3]))/np.cosh(float(jet[2]))
        pz = float(jet[1])*np.sinh(float(jet[2]))/np.cosh(float(jet[2]))
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


# In[5]:
def sphericity(w,v):
    return (3/2) * (sorted(w)[1]+sorted(w)[2])


# In[6]:
def aplanarity(w,v):
    return (3/2) * sorted(w)[2]


# In[7]:
def transverse_sphericity(w,v): #
    #print(sorted(w)[0],sorted(w)[1])
    return (2*sorted(w)[1])/(sorted(w)[0]+sorted(w)[1])


# In[8]:
#def planarity(w,v):
#    return


# In[9]:
#thrust


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

#--------------------------- Parse text file
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

        this_record['y23'] = y23(jets_vec)

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

#-------------------------------------------------------------------------
if __name__ == "__main__":

  bg_file_list = glob.glob("/data/users/jgonski/snowmass/LHE_txt_fils/processed_lhe*_background.txt")
  signal_file_list = glob.glob("/data/users/jgonski/snowmass/LHE_txt_fils/processed_lhe*signal.txt")

  bg_records = []
  for filename in bg_file_list:
      file = open(filename)
      bg_records += parse_file(file)
      if len(bg_records) > 100: break
  sig_records = []
  for filename in signal_file_list:
      file = open(filename)
      sig_records += parse_file(file)
      if len(sig_records) > 100: break

  print('Running over '+str(len(bg_records))+' background files and '+str(len(sig_records))+' signal files....')

  for i in sig_records:
      i['from_anomaly_data'] = True
  for i in bg_records:
      i['from_anomaly_data'] = False

  all_records = sig_records[:100] + bg_records[:100]
  #:79999

  plot_something('truthsqrtshat',range(0,1000,20),1)
  plot_something('y23',np.linspace(0,0.24,50),1)
  plot_something('transverse_sphericity',np.linspace(0,10,100),1)
  plot_something('sphericity',np.linspace(1.0,2,50),1)
  plot_something('aplanarity',np.linspace(0,1,100),1)
  
  # didn't validate this plotting function works
  # plot_jets?
  
  
  # # NN training
  # TODO 


