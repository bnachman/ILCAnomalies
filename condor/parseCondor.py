#!/usr/bin/env python
# coding: utf-8

# In[1]:
# IO: do this:
#source activate fullenv
#python -m ipykernel install --user --name fullenv --display-name "fullenv"
# also see this https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
import sys
import numpy as np
#import matplotlib.pyplot as plt
import glob
#import energyflow as ef
#from energyflow.archs import DNN
#from energyflow.datasets import qg_jets
#from energyflow.utils import data_split, remap_pids, to_categorical
#from keras.models import Sequential
#from keras.layers import Dense 
import logging
from eventHelper import *
from datetime import datetime
from ROOT import *
import math
#from pylorentz import Momentum4

#--------------------------- Parse text files
def get_x_vec(jet1,jet2):
        jet14Vec = TLorentzVector(0.0,0.0,0.0,0.0)
        jet14Vec.SetPtEtaPhiM(float(jet1[1]),float(jet1[2]),float(jet1[3]),float(jet1[4]))
        jet24Vec = TLorentzVector(0.0,0.0,0.0,0.0)
        jet24Vec.SetPtEtaPhiM(float(jet2[1]),float(jet2[2]),float(jet2[3]),float(jet2[4]))
        xVec = jet14Vec + jet24Vec
        return xVec

#--------------------------- Parse text files
def parse_file(file_object,startNum,endNum,filename):
    all_records = []
    mymeasuredenergy = []
    ## Generate spherical sample
    #sphereSample = np.array([sphericalGen(i) for i in range(5)])
    #sphereEng = np.array([engFromVec(sphereSample[j]) for j in range(5)])
    ## Choose sphere n points
    #sphInd = 2
    #spherePoints1 = sphereSample[sphInd]
    #sphereEng1 = sphereEng[sphInd]
    photonPs= []
    jetPs = []

    count = 0
    for line in file_object:
        if count < int(startNum): 
          count += 1
          continue 
        if count > int(endNum): break
        if count%100 == 0: print('Line '+str(count))

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

        #--- boosting 
        #photon4Vec = TLorentzVector(0.0,0.0,0.0,0.0)
        #photon4Vec.SetPtEtaPhiM(float(metadata.split()[8]), float(metadata.split()[9]),float(metadata.split()[10]), 0.0)
        #logging.info('photon boost: ' +str(photon4Vec.Px())+ " "+ str(photon4Vec.Py())+" "+ str(photon4Vec.Pz()))
        #logging.info('boost vecotr mag: '+str(photon4Vec.BoostVector().Mag()))
        #if photon4Vec.BoostVector().Mag() >= 1.0: 
        #  #print('greater than 1.')
        #  continue
        #photon4Vec.Boost(-photon4Vec.BoostVector())
        #logging.info('after boosting ' +str(photon4Vec.Px())+ " "+ str(photon4Vec.Py())+" "+ str(photon4Vec.Pz()))
        #logging.info('size of photon boost: '+str(np.sqrt(photon4Vec.Px()**2 + photon4Vec.Py()**2 + photon4Vec.Pz()**2)))
        #photonPs.append(np.sqrt(photon4Vec.Px()**2 + photon4Vec.Py()**2 + photon4Vec.Pz()**2))

        #pylorentz
        #photonVec = Momentum4.m_eta_phi_pt(0.0, float(metadata.split()[9]), float(metadata.split()[10]), float(metadata.split()[8]))
        #logging.info('before boost: '+str(photonVec.p_t) +" " + str(photonVec.eta)+" " + str(photonVec.phi)+ " "+ str(photonVec.m))
        #photon4Vec = photonVec.boost_particle(-photonVec)
        #logging.info('after boost: '+str(photon4Vec.p_t) +" " + str(photon4Vec.eta)+" " + str(photon4Vec.phi)+ " "+ str(photon4Vec.m))



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


        mymeasuredenergy+=[measuredcenterofmassenergy]


        this_record['njets'] = njets
        if njets < 2: continue
        jets = jets.split()
        jets_vec = []
        jet1 = jets[0*11:0*11+11]
        jet2 = jets[1*11:1*11+11]
        #--- boosting 
        boostVec = get_x_vec(jet1,jet2)
        #logging.info('leading jet pair boost: '+ str(boostVec.Px()) +" " +str(boostVec.Py())+" " +str(boostVec.Pz()))
        #logging.info('unboosted X pt: '+ str(boostVec.Pt()))
        jetPs.append(np.sqrt(boostVec.Px()**2 + boostVec.Py()**2 +boostVec.Pz()**2))
        
  
        for i in range(njets):
            jet = np.zeros(11)
            #order:
            # - index
            # - magnitude of momentum pT (units of GeV)
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
            # replace some comps with boost 
            thisJet4Vec = TLorentzVector(0.0,0.0,0.0,0.0)
            thisJet4Vec.SetPtEtaPhiM(float(jet[1]),float(jet[2]),float(jet[3]),float(jet[4]))
            thisJet4Vec.Boost(-boostVec.BoostVector())
            #thisJet4Vec.Boost(-photon4Vec.BoostVector())
            #thisJet4Vec = Momentum4.m_eta_phi_pt(float(jet[4]), float(jet[2]), float(jet[3]), float(jet[1]))
            #thisJet4Vec.boost_particle(photon4Vec)
            logging.info('before jet: ' +str(float(jet[1])) +" " +str(float(jet[2])) + " " +str(float(jet[3])) + " " +str(float(jet[4])))
            jet[1] = thisJet4Vec.Pt()
            jet[2] = thisJet4Vec.Eta()
            jet[3] = thisJet4Vec.Phi()
            jet[4] = thisJet4Vec.M() 
            logging.info('after jet: ' +str(float(jet[1])) +" " +str(float(jet[2])) + " " +str(float(jet[3])) + " " +str(float(jet[4])))
            jets_vec+=[jet]

        logging.warning('measured X pt: '+ str(get_x_vec(jets_vec[0], jets_vec[1]).Pt()))
        this_record['jets']=jets_vec
        this_record['leadingjetpT']= float(jets_vec[0][1]) if len(jets_vec)>0 else -1
        this_record['subleadingjetpT']= float(jets_vec[1][1]) if len(jets_vec)>1 else -1
        this_record['measuredXpT']= get_x_vec(jets_vec[0], jets_vec[1]).Pt() if len(jets_vec)>1 else -1

        this_record['lny23'] = lny23(jets_vec)
        this_record['total_jet_mass'] = total_jet_mass(jets_vec)

        #thrust_maj, thrust_min = thrust(jets_vec)
        #this_record['thrust_major'] = thrust_maj
        #this_record['thrust_minor'] = thrust_min
        #print("thrust major: ", thrust_maj, ", minor: ", thrust_min)
        #w,v = momentum_tensor(jets_vec,2)
        #this_record['sphericity'] = sphericity(w,v)
        #this_record['transverse_sphericity'] = transverse_sphericity(w,v)
        #this_record['aplanarity'] = aplanarity(w,v)

        this_record['nparticles'] = nparticles

        particles = particles.split()
        particles_vec = []
        for i in range(nparticles):
            particle = np.zeros(5)
            #order:
            # - index
            # - magnitude of momentum pT (units of GeV)
            # - pseudorapidity (~polar angle - see e.g. https://en.wikipedia.org/wiki/Pseudorapidity)
            # - azimuthal angle
            # - particle identifier (https://pdg.lbl.gov/2006/reviews/pdf-files/montecarlo-web.pdf)
            particle = particles[i*5:i*5+5]
            # replace some comps with boost 
            particle4Mom = TLorentzVector(0.0,0.0,0.0,0.0)
            particle4Mom.SetPtEtaPhiM(float(particle[1]), float(particle[2]),float(particle[3]),0.0)
            particle4Mom.Boost(-boostVec.BoostVector())
            #particle4Mom.Boost(-photon4Vec.BoostVector())
            logging.info('before particle: ' + particle[1] +" " + particle[2] + " " + particle[3])
            particle[1] = particle4Mom.Pt()
            particle[2] = particle4Mom.Eta()
            particle[3] = particle4Mom.Phi()
            particles_vec+=[particle]
            logging.info('afterr particle: ' + str(particle[1]) +" " + str(particle[2]) + " " + str(particle[3]))
            #print(particles[i*5],particles[i*5+1],particles[i*5+2],particles[i*5+3],particles[i*5+4])
        this_record['particles'] = particles_vec
        
        #this_record['lny23'] = lny23(particles_vec)
        isNan = False
        for ele in particles_vec:
          for j in ele: 
            if j!=j: 
              isNan = True
        if isNan: continue
        w,v = momentum_tensor(particles_vec,2)
        this_record['sphericity'] = sphericity(w,v)
        this_record['transverse_sphericity'] = transverse_sphericity(w,v)
        this_record['aplanarity'] = aplanarity(w,v)

        # Event isotropy 
        #this_record['evIsoSphere'] = evIsoSphere(particles_vec,spherePoints1,sphereEng1)
         
        
        all_records.append(this_record)

    #plt.hist(photonPs,label='Photon boost momentum')
    #plt.hist(jetPs,label='2 leading jet boost momentum')
    #plt.xlabel('|p| [GeV]')
    #plt.legend()
    #if 'sig' in filename: plt.savefig('0323_boostComp_sig.pdf')
    #else: plt.savefig('0323_boostComp_bkg.pdf')
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
        #evt_vars = [record['leadingjetpT'], record['subleadingjetpT'],record['measuredXpT'],record['measuredphotonpT'],record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['total_jet_mass'],record['evIsoSphere']]
        evt_vars = [record['leadingjetpT'], record['subleadingjetpT'],record['measuredXpT'],record['measuredphotonpT'],record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['total_jet_mass']]
        padded_evt_arrays.append(np.array(evt_vars).real)
    return np.array(padded_evt_arrays)

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
  records += parse_file(file,startNum,endNum,filename)
  X = make_evt_arrays(records)
  y = np.array([i['truthsqrtshat'] for i in records])
  np.save(tag+"_X_"+filename.split('/')[-1].split('.')[0]+"_"+str(startNum)+"to"+str(endNum), X)
  np.save(tag+"_y_"+filename.split('/')[-1].split('.')[0]+"_"+str(startNum)+"to"+str(endNum), y)


  
  print('runtime: ',datetime.now() - startTime)
