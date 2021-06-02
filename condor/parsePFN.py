#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IO: do this:
#source activate fullenv
#python -m ipykernel install --user --name fullenv --display-name "fullenv"
# also see this https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import logging
from eventHelper import *
from datetime import datetime
from ROOT import *
import math


#--------------------------- Parse text files
def parse_file(file_object,startNum,endNum,filename,tag):
    all_records = []
    mymeasuredenergy = []
    max_n_jets = 15

    count = 0
    pdgIds = []
    chargedPts=  []
    neutronPts = []
    for line in file_object:
        if count < int(startNum): 
          count += 1
          continue 
        if count > int(endNum): break
        if count%1000 == 0: print('Line '+str(count))

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
        this_record['truthphotonpT'] = float(metadata.split()[3]) #photon momentum pT in units of GeV
        this_record['truthphotoneta'] = float(metadata.split()[4]) #photon pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        this_record['truthphotonphi'] = float(metadata.split()[5]) #photon azimuthal angle

        #Measured collision quantities
        measuredcenterofmassenergy  = float(metadata.split()[6]) #true measured energy - should be noisy version of truthcenterofmassenergy
        this_record['measuredcenterofmassenergy'] = measuredcenterofmassenergy
        this_record['measuredsqrtshat'] = float(metadata.split()[7]) #energy available for making new particles (electron energy - photon)
        this_record['measuredphotonpT'] = float(metadata.split()[8]) #photon momentum pT in units of GeV
        this_record['measuredphotoneta'] = float(metadata.split()[9]) #photon pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        this_record['measuredphotonphi'] = float(metadata.split()[10]) #photon azimuthal angle
        this_record['metadata'] = metadata.split()

        outgoingPhoton = TLorentzVector(0.0,0.0,0.0,0.0)
        outgoingPhoton.SetPtEtaPhiM(float(metadata.split()[8]), float(metadata.split()[9]), float(metadata.split()[10]), 0.0)

        mymeasuredenergy+=[measuredcenterofmassenergy]


        this_record['njets'] = njets
        jets_list = jets.split()  
        jets_vec = np.array(jets_list).astype('float')
        jets_vec = np.reshape(jets_vec, (-1, 11))
        jets_vec = jets_vec[:,1:] #omit index 0 
        #for i in range(njets):
        #    jet = np.zeros(11)
        #    #order:
        #    # - index
        #    # - magnitude of momentum pT (units of GeV)
        #    # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
        #    # - azimuthal angle
        #    # - mass (units of GeV/c^2)
        #    # - bit encoding of the jet "flavor" (not totally sure what the bit means, but will look it up)
        #    # - 0th angular moment of jet radiation
        #    # - 1th angular moment of jet radiation
        #    # - 2th angular moment of jet radiation
        #    # - 3th angular moment of jet radiation
        #    # - 4th angular moment of jet radiation
        #    jet = jets[i*11:i*11+11]
        #    jets_vec+=[jet]
        j_pad_length = max_n_jets - jets_vec.shape[0]
        padded_jets = np.pad(jets_vec, ((0,j_pad_length),(0,0)))
        assert padded_jets.shape == (max_n_jets, 10)
        this_record['jets']=padded_jets


        this_record['nparticles'] = nparticles
        particles_vec = []
        particles = particles.split()
        hadronSqrtSHat = TLorentzVector(0.0,0.0,0.0,0.0)
        for i in range(nparticles):
            particle = np.zeros(5)
            pVec = TLorentzVector(0.0,0.0,0.0,0.0)
            pdgId = particles[i*5+4]
            pdgIds.append(pdgId)
            if '2112' in pdgId: neutronPts.append(float(particles[i*5+1]))
            elif '211' in pdgId: chargedPts.append(float(particles[i*5+1]))
            #order:
            # - index
            # - magnitude of momentum pT (units of GeV)
            # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
            # - azimuthal angle
            # - particle identifier (https://pdg.lbl.gov/2006/reviews/pdf-files/montecarlo-web.pdf)
            particle = particles[i*5:i*5+5]
            pVec.SetPtEtaPhiM(float(particles[i*5+1]), float(particles[i*5+2]), float(particles[i*5+3]), 0.0)
            hadronSqrtSHat = hadronSqrtSHat + pVec
            particles_vec+=[particle]
        #this_record['particles'] = particles_vec
        this_record['hadronsqrtshat'] = (hadronSqrtSHat-outgoingPhoton).M() #energy available for making new particles (sum of final state hadrons, i.e. lost photon)

        #---------- debug hadron calc
        #print('truth: ', this_record['truthsqrtshat'], ', measured with photon: ', this_record['measuredsqrtshat'] ,', hadronsqrtshat: ', hadronSqrtSHat.M(), ', corrected minus photon: ', (hadronSqrtSHat-outgoingPhoton).M())
        sqrtSDiff = this_record['measuredsqrtshat'] - this_record['hadronsqrtshat']
        #print('diff: ', sqrtSDiff)
        # visibleenergy = hadronSqrtSHat 
        #print("BEN DEBUG: ", visibleenergy.M(),  " ", (visibleenergy-highestpTphoton).M(), " ", (incomingelectron+incomingpositron-highestpTphoton).M())
        #print("BEN DEBUG: visible energy:", hadronSqrtSHat.M(),  ", visible energy-photon: ", (hadronSqrtSHat-outgoingPhoton).M(), ", measured CoM (e+e- - ph):", float(metadata.split()[7]))
        newMeasured = float(metadata.split()[1]) - float(metadata.split()[8]) # truthCoM - reco photon
        print(hadronSqrtSHat.M(),  " ", (hadronSqrtSHat-outgoingPhoton).M(), " ", newMeasured)

        all_records.append(this_record)

    print('unique pdgids: ', np.unique(np.array(pdgIds)))
    bins=np.linspace(0.0,2.0,400) 
    plt.hist(chargedPts,bins=bins,label='charged pions',color='green')
    plt.hist(neutronPts,bins=bins,label='neutrons',color='purple')
    ticks=np.linspace(0.0,2.0,20)
    plt.xticks(ticks,rotation ='vertical',fontsize=5)
    plt.legend()
    if 'sig' in filename: plt.title('Signal 350 GeV')
    else: plt.title('Background')
    plt.xlabel('pT [GeV?]')
    plt.savefig(tag+"_particlePts_"+filename.split('/')[-1].split('.')[0]+'.pdf')
    plt.clf()

    bins2=np.linspace(-1.0,1.0,500) 
    plt.hist(sqrtSDiff,bins=bins2)
    plt.xlabel('Measured - hadron sqrtshat [GeV]')
    if 'sig' in filename: plt.title('Signal 350 GeV')
    else: plt.title('Background')
    plt.savefig(tag+"_sqrtSDiff_"+filename.split('/')[-1].split('.')[0]+'.pdf')
    plt.clf()
    

    return all_records

#-----------------------------------------------------------------------------------
def make_pfn_arrays(these_records):
    padded_jet_arrays = []
    for i,record in enumerate(these_records):
        padded_jet_arrays.append(record['jets'])
    #print('jets shape:', np.shape(padded_jet_arrays))
    return np.array(padded_jet_arrays)

#-------------------------------------------------------------------------
if __name__ == "__main__":

  startTime = datetime.now()
  print('hello! start time = ', str(startTime))
  tag=sys.argv[1]
  filename=sys.argv[2]
  startNum=sys.argv[3]
  endNum=sys.argv[4]
  logging.basicConfig(level=logging.ERROR)

  records = []
  print('Running filename ', filename, ' from line ', startNum, ' to ', endNum)
  #file = open('../'+str(filename))
  file = open(str(filename))
  records += parse_file(file,startNum,endNum,filename,tag)
  X = make_pfn_arrays(records)
  #y = np.array([i['truthsqrtshat'] for i in records])
  #y = np.array([i['measuredsqrtshat'] for i in records])
  y = np.array([i['hadronsqrtshat'] for i in records])
  np.save(tag+"_X_"+filename.split('/')[-1].split('.')[0]+"_"+str(startNum)+"to"+str(endNum), X)
  np.save(tag+"_y_"+filename.split('/')[-1].split('.')[0]+"_"+str(startNum)+"to"+str(endNum), y)

  
  print('runtime: ',datetime.now() - startTime)

