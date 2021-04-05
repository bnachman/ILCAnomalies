#!/usr/bin/env python
# coding: utf-8
import numpy as np
import argparse
#import matplotlib.pyplot as plt
import glob
from ROOT import *
#from eventIsotropy.spherGen import sphericalGen, engFromVec
#from eventIsotropy.emdVar import _cdist_cos, emd_Calc


# # Defining event level variables
# IO: arXiv:1206.2135.pdf
# JG: https://arxiv.org/pdf/1811.00588.pdf (total jet mass)

d_npy={
'leadingjetpT':0,
'subleadingjetpT':1,
'measuredXpT':2,
'measuredphotonpT':3,
'njets':4,
'nparticles':5,
'lny23':6,
'aplanarity':7,
'transverse_sphericity':8,
'sphericity':9,
'total_jet_mass':10,
'evIsoSphere':11
}

#--------------------------- Variable defs
def get_three_vec(jet):
      #BUG pt = float(jet[1])/np.cosh(float(jet[2]))
      pt = float(jet[1])
      px = pt*np.cos(float(jet[3]))
      py = pt*np.sin(float(jet[3]))
      pz = pt*np.sinh(float(jet[2]))
 
      return [px,py,pz]

def evIsoSphere(particles_vec,spherePoints1,sphereEng1):
  momenta=[]
  engL=[]
  
  for p in particles_vec:
    v = TLorentzVector(0,0,0,0)
    v.SetPtEtaPhiM(float(p[1]),float(p[2]),float(p[3]),0.0)
    eng, px, py, pz = v.E(), v.Px(), v.Py(), v.Pz()
    if eng > 1e-05:
        momenta.append(np.array([px, py, pz]))
        engL.append(eng)
  
  ## Calculate the \semd values
  M = _cdist_cos(spherePoints1,np.array(momenta)) # Calculates distance with 1 - \cos metric
  emdval = emd_Calc(sphereEng1, np.array(engL), M) # Computes EMD
  print(emdval)
  return emdval

def total_jet_mass(jets):
    sumVec = TLorentzVector(0.,0.,0.,0.)
    sumP = 0.0
    for jet in jets:
      vec = TLorentzVector(0.,0.,0.,0.)
      sumP += float(jet[1])
      pt = float(jet[1])
      vec.SetPtEtaPhiM(float(pt),float(jet[2]),float(jet[3]),float(jet[4]))
      sumVec = sumVec+vec
    tjm = np.divide(np.power(sumVec.M(),2),np.power(sumP,2))
    return tjm

def lny23(jets):
    if len(jets) > 2:
        #jet1_pt = float(jets[0][1])/np.cosh(float(jets[0][2]))
        #jet2_pt = float(jets[1][1])/np.cosh(float(jets[1][2]))
        #jet3_pt = float(jets[2][1])/np.cosh(float(jets[2][2]))
        jet1_pt = float(jets[0][1])
        jet2_pt = float(jets[1][1])
        jet3_pt = float(jets[2][1])
        return np.log((jet3_pt*jet3_pt)/((jet1_pt+jet2_pt)*(jet1_pt+jet2_pt)))
    return 0


def momentum_tensor(jets,r):
    m = np.zeros((3,3))
    totalPSq = 1e-10
    for jet in jets:
        #print(jet)
        #[index, p [GeV], eta, phi, m]
        px,py,pz = get_three_vec(jet)
        #print('three vec: ', px, py, pz)
        pr = np.power(float(jet[1]),r-2)
        m += [[px*px*pr, px*py*pr, px*pz*pr], [py*px*pr, py*py*pr, py*pz*pr], [pz*px*pr, pz*py*pr, pz*pz*pr]]
        totalPSq += np.power(float(jet[1]),r)
    #print(totalPSq)
    m = m/totalPSq
    from numpy import linalg as LA
    w, v = LA.eig(m)
    #print("eigenvalues (sum should be normalized to 1): ", sorted(w))
    #print("eigenvectors: ",v)
    return w, v
    #return m  #From this, the sphericity, aplanarity and planarity can be calculated by combinations of eigenvalues.


def aplanarity(w,v):
    return (3/2) * sorted(w)[0] #lambda3
def sphericity(w,v): 
    return (3/2) * (sorted(w)[1]+sorted(w)[0]) #lambda2 + lamdba3
def transverse_sphericity(w,v): 
    return (2*sorted(w)[1])/(sorted(w)[2]+sorted(w)[1]) #2*lambda2 / (lam1+lam2)




#thrust
# adapted from the jet-level thrust axis calculation, which uses constituents. See eg. https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Reconstruction/Jet/JetSubStructureUtils/Root/Thrust.cxx 
def thrust(jets):

  thrust_major = -999
  thrust_minor = -999
  useThreeD = True

  if len(jets) < 2: return [thrust_major,thrust_minor]

  agree = 0
  disagree = 0
  max_tests = 2 #TODO
  n_tests = 0
  #n_0 = [TVector3(0.,0.,0.),TVector3(0.,0.,0.),TVector3(0.,0.,0.),TVector3(0.,0.,0.)] 
  n_0 = []
  for n in range(max_tests):
    n_0.append([0.,0.,0])
  #while (disagree>0 or agree<2 ) and n_tests < max_tests:
  add0= [ 1, 0, 1, 1,-1,-1 ]
  add1= [ 0, 1, 1,-1, 1,-1 ]
  #add0= [ 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1 ]
  #add1= [ 0, 1, 0, 0, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1 ]
  #add2= [ 0, 0, 1, 0, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1 ]
  #add3= [ 0, 0, 0, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1 ]


  # ------- Determine n_0 = first guess
  #jets are already pt sorted; take hardest 2 to find thrust
  # assign direction of 2 most energetic particles  
  px0,py0,pz0 = get_three_vec(jets[0])
  px1,py1,pz1 = get_three_vec(jets[1])
  j_0 = [float(px0),float(py0),float(pz0)]
  j_1 = [float(px1),float(py1),float(pz1)]
  #print('Jet 3 vec j_0: ', j_0[0], j_0[1], j_0[2])
  #print('Jet 3 vec j_1: ', j_1[0], j_1[1], j_1[2])

  while (disagree>0 or agree <2) and n_tests < max_tests:
    n_0[n_tests] =  (add0[n_tests] * [px0,py0,pz0] + add1[n_tests]*[px1,py1,pz1])
    #print('Thrust axis n_0: ', n_0[0], n_0[1], n_0[2])
 
    #if useThreeD==False: n_0.SetZ(0.0)

    #protect against small number of input particles (smaller than 4!)
    #if (n_0[n_tests].Mag() > 0)
    #  n_0[n_tests] *= 1/n_0[n_tests].Mag();
    


    #--------- SKIP FOR NOW: take only two hardest jets
    # ------- Determine n_1 = include all particles
    run = False
    loop = 0 
    while run: 
      n_1 = [0.,0.,0]
      #loop over all jets this time: 
      for j in range(len(jets)):
        px0,py0,pz0 = get_three_vec(jets[j])
        if (float(px)* n_0[n_tests][0] + float(py)* n_0[n_tests][1] + float(pz)* n_0[n_tests][2] ) > 0: 
          n_1[0] += px
          n_1[1] += py
          n_1[2] += pz
        else:
          n_1[0] -= px
          n_1[1] -= py
          n_1[2] -= pz
    
      #if useThreeD==False: n_1[n_tests].SetZ(0.0)
      #protect against small number of input particles (smaller than 4!)
      #if (n_1[n_tests].Mag() > 0)
      #  n_1[n_tests] *= 1/n_1[n_tests].Mag();

      # has axis changed ? if so, try at most ten times (thrust axis has two fold ambiguity)
      run = (n_0[n_tests] != n_1) and (-n_0[n_tests] != n_1) and loop < 10
      n_0[n_tests] = n_1
      loop += 1

    # agrees or disagrees with first result ?
    #  thrust has a sign ambiguity
    if n_tests > 0:
      if n_0[0][0] == np.abs(n_0[n_tests][0]) and n_0[0][1] == np.abs(n_0[n_tests][1]) and n_0[0][2] == np.abs(n_0[n_tests][2]): 
        agree+=1
      else: disagree += 1
    n_tests += 1
         


  # now that we have the thrust axis, we determine the thrust value
  #  if the various calculations of the thrust axes disagree, try all
  #  and take the maximum, calculate minor and mayor axis
  n_tests=0
  while n_tests < max_tests:
    denominator = 0.0
    numerator_t = 0.0
    numerator_m = 0.0
    for h in range(2): #just products of 2 leading jets 
        pt = float(jets[h][1])/np.cosh(float(jets[h][2]))
        px = pt*np.cos(float(jets[h][3]))
        py = pt*np.sin(float(jets[h][3]))
        pz = pt*np.sinh(float(jets[h][2]))
        c = [float(px),float(py),float(pz)]
        #why ? c.setZ(0)
        numerator_t += abs(np.dot(c,n_0[n_tests]))
        numerator_m += np.linalg.norm(np.cross(c,n_0[n_tests]))
        denominator += np.linalg.norm(c)
    inv_denominator = 1. / denominator
    if numerator_t * inv_denominator > thrust_major: 
        thrust_major = numerator_t * inv_denominator
        thrust_minor = numerator_m * inv_denominator
    n_tests += 1

  return [thrust_major,thrust_minor]

#def planarity(w,v):
#    return



#---------------------------  Plotting help
#def make_roc_plots(anomalyRatios,saveTag,Ylabel,rocs,aucs,sigs):
#  for i,r in enumerate(anomalyRatios):
#      #Ines plt.plot(rocs[i][1],rocs[i][1]/np.sqrt(rocs[i][0]),label=r'AnomRatio=%0.3f, $\sigma$ = %0.1f, AUC %0.2f'%(anomaly_ratios[i],significances[i],aucs[i]))
#      if 'sqrt' in Ylabel: plt.plot(rocs[i][1],rocs[i][1]/np.sqrt(rocs[i][0]),label=str(r)+", $\sigma$="+str(sigs[i])+": AUC="+str(np.round(aucs[i],2)))
#      else: plt.plot(rocs[i][0],rocs[i][1],label=str(r)+", $\sigma$="+str(sigs[i])+": AUC="+str(np.round(aucs[i],2)))
#  if 'sqrt' in Ylabel: 
#    plt.xlabel('tpr')
#    plt.ylim(0,6.0)
#  else: plt.xlabel('fpr')
#  plt.ylabel(Ylabel)
#  plt.title('ROC curve: S vs. B in SR')
#  plt.legend()
#  plt.savefig('plots/'+saveTag+'_roc_aucs_benchmark_'+Ylabel.replace("/","")+'.pdf')
#  plt.clf()
#  #plt.show()
#
#def plot_loss(h,r,save):
#      plt.plot(h.history['loss'])
#      plt.plot(h.history['val_loss'])
#      plt.title('model loss, ar='+str(r))
#      plt.ylabel('loss')
#      plt.xlabel('epoch')
#      plt.legend(['train', 'val'], loc='upper left')
#      plt.savefig('plots/'+save+'_lossVsEpoch_anomalyRatio'+str(r)+'.pdf')
#      plt.clf()

#def make_npy_plots(X_sig,X_bg,var,R,save,doLog=True):
#  sig_arr = np.array([float(i[d_npy[var]]) for i in X_sig])
#  bkg_arr = np.array([float(i[d_npy[var]]) for i in X_bg])    
#  print('bkg arr: ', bkg_arr)
#  plt.hist(bkg_arr, np.linspace(0,2.0,100), Color="steelblue", histtype='step', linewidth=2,label='Background')
#  #plt.hist(sig_arr, Color="tomato", histtype='step', linewidth=2,label='Signal')
#  plt.xlabel(var)
#  if doLog == True: plt.yscale('log')
#  plt.ylabel("Number of Events / bin")
#  plt.legend()
#  plt.savefig("plots/"+save+"_plt_"+var+".pdf")
#  plt.clf()
#

#def make_var_plots(sig_records,bg_records,save):
#  #plot_something(save,sig_records,bg_records,'truthsqrtshat',range(0,1000,20),1)
#  #plot_something(save,sig_records,bg_records,'lny23',np.linspace(-10,-0.00001,10),1)
#  #plot_something(save,sig_records,bg_records,'transverse_sphericity',np.linspace(0,1,50),1)
#  #plot_something(save,sig_records,bg_records,'sphericity',np.linspace(0,1,50),1)
#  #plot_something(save,sig_records,bg_records,'aplanarity',np.linspace(0,0.3,15),1)
#  #plot_something(save,sig_records,bg_records,'total_jet_mass',np.linspace(0,2.0,100),1)
#  plot_something(save,sig_records,bg_records,'leadingjetpT',np.linspace(0,300,100),1)
#  plot_something(save,sig_records,bg_records,'measuredphotonpT',np.linspace(0,300,100),1)
#  plot_something(save,sig_records,bg_records,'measuredXpT',np.linspace(0,300,100),1)
#  #plot_something(save,sig_records,bg_records,'thrust_major',np.linspace(0,500,50),1)
#  #plot_something(save,sig_records,bg_records,'thrust_minor',np.linspace(0,500,50),1)
#
#def plot_something(save,sig_records,bg_records,var,R,doLog):
#    #plt.figure(figsize=(20,5))
#    if 'npy' in save: 
#      sig_arr = np.array([float(i[d_npy[var]]) for i in sig_records])
#      bkg_arr = np.array([float(i[d_npy[var]]) for i in bg_records])    
#    else:
#      sig_arr = np.array([i[var] for i in sig_records[:79999]])
#      bkg_arr = np.array([i[var] for i in bg_records])    
#    plt.hist(bkg_arr, R, color="steelblue", histtype='step', linewidth=2,label='Background')
#    plt.hist(sig_arr, R, color="tomato", histtype='step', linewidth=2,label='Signal')
#    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
#    #plt.xscale('log')
#    #plt.xticks(R)
#    plt.xlabel(var)
#    if doLog == True: plt.yscale('log')
#    plt.ylabel("Number of Events / bin")
#    plt.legend()
#    plt.savefig("plots/"+save+"_plt_"+var+".pdf")
#    plt.clf()
#
#def plot_jets(index,R,doLog):
#    sig_arr = np.array([i['jets'][0][index] for i in sig_records[:79999]])
#    #bkg_arr = np.array([i['jets'][0][index] for i in bg_records[:]])    
#    plt.hist(bkg_arr, R, color="steelblue", histtype='step', linewidth=2)
#    plt.hist(sig_arr, R, color="tomato", histtype='step', linewidth=2)
#    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
#    #plt.xscale('log')
#    #plt.xticks(R)
#    plt.xlabel(var)
#    if doLog == True: plt.yscale('log')
#    plt.ylabel("Number of Events / bin")
#    plt.savefig("plots/plt_"+index+".pdf")
#    plt.clf()
