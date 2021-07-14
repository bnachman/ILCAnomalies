#!/usr/bin/env python
# coding: utf-8
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
from sklearn.preprocessing import quantile_transform
#from ROOT import *
#from eventIsotropy.spherGen import sphericalGen, engFromVec
#from eventIsotropy.emdVar import _cdist_cos, emd_Calc


# # Defining event level variables
# IO: arXiv:1206.2135.pdf
# JG: https://arxiv.org/pdf/1811.00588.pdf (total jet mass)

nice_colors_rtob_vib = ['#EE6677','#CCBB44','#228833','#66CCEE','#4477AA','#AA3377']
nice_colors_rtob_mut = ['#882255','#CC6677','#999933','#117733','#88CCEE','#332288','#AA4499']
prettySigmas = ['0.0', '0.5', '1.0', '2.0', '3.0', '5.0','$\infty$'] 

#--------------------------- Variable defs
#pretty labels
jet_dict=[
    ['p$_T$ [GeV]',np.linspace(0,550,220),    0.000001,10.0,0.0,3.0],
    ['$\eta$',np.linspace(-3.0,3.0,30),       0.0001,10.0,-3.0,3.0],      
    ['$\phi$',np.linspace(-3.0,3.0,30),       0.0001,10.0,-3.0,3.0],
    ['mass [GeV]',np.linspace(0,300,150),     0.000001,10.0,0.0,3.0],
    ['flavor',np.linspace(0,8,8),           0.0001,10.0,0.0,3.0],
    ['angular radiation moment 1',np.linspace(0,1.0,50),0.00005,100.0,0.0,40.0],
    ['angular radiation moment 2',np.linspace(0,1.0,50),0.00005,100.0,0.0,40.0],
    ['angular radiation moment 3',np.linspace(0,1.0,50),0.00005,100.0,0.0,40.0],
    ['angular radiation moment 4',np.linspace(0,1.0,50),0.00005,100.0,0.0,40.0],
    ['angular radiation moment 5',np.linspace(0,1.0,50),0.00005,100.0,0.0,40.0],
]
get_pretty={
    'measuredXpT':['p$_T$(X)',0.000001,10,0.0,1.0],
    'xpT_Over_PhpT':['p$_T$(X) / p$_T$($\gamma$)',0.00001,10,0.0,5.0],
    'ljpT_Over_PhpT':['p$_T$($j_1$) / p$_T$($\gamma$)',0.0001,100,0.0,20.0],
    'measuredphotonpT':['Outgoing photon p$_T$ [GeV]',0.000001,10,0.0,1.0],
    'njets':['Number of jets',0.000001,100,0.0,5.0],
    'nparticles':['Number of particles',0.000001,10,0.0, 1.0],
    'lny23':['ln(y$_{23}$)',0.00001,50,-10.0, 8.0],
    'aplanarity':['Aplanarity',0.0001,1000,0.0, 100.0],
    'transverse_sphericity':['Transverse sphericity',0.0001,1000,0.0,300.0],
    'sphericity':['Sphericity',0.001, 1000,0.0,300.0],
    'total_jet_mass':['Total jet mass',0.0001,1000,0.0,200.0]
} 
    #'leadingjetpT':'Leading jet p$_T$ [GeV]',
    #'subleadingjetpT':'Subleading jet p$_T$ [GeV]',
    #'leadingjetmass':'Leading jet mass [GeV]',
    #'subleadingjetmass':'Subleading jet mass [GeV]',


#--------------------------- Variable defs
def get_npy_dict(save):
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
  }

  if '040' in save:
        #evt_vars = [record['xpT_Over_PhpT'], record['ljpT_Over_PhpT'],record['leadingjetpT'], record['subleadingjetpT'],record['measuredXpT'],record['measuredphotonpT'],record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['total_jet_mass'],record['splitting'],record['leadingjetmass'],record['subleadingjetmass']]
    d_npy={
    'xpT_Over_PhpT':0,
    'ljpT_Over_PhpT':1,
    'leadingjetpT':2,
    'subleadingjetpT':3,
    'measuredXpT':4,
    'measuredphotonpT':5,
    'njets':6,
    'nparticles':7,
    'lny23':8,
    'aplanarity':9,
    'transverse_sphericity':10,
    'sphericity':11,
    'total_jet_mass':12,
    'splitting':13,
    'leadingjetmass':14,
    'subleadingjetmass':15
    }
  elif '060' in save: 
  #evt_vars = [record['measuredXpT'],record['xpT_Over_PhpT'], record['ljpT_Over_PhpT'],record['leadingjetpT'], record['subleadingjetpT'],record['measuredphotonpT'],record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['leadingjetmass'],record['subleadingjetmass'],record['total_jet_mass']]
    d_npy={
    'measuredXpT':0,
    'xpT_Over_PhpT':1,
    'ljpT_Over_PhpT':2,
    'leadingjetpT':3,
    'subleadingjetpT':4,
    'measuredphotonpT':5,
    'njets':6,
    'nparticles':7,
    'lny23':8,
    'aplanarity':9,
    'transverse_sphericity':10,
    'sphericity':11,
    'leadingjetmass':12,
    'subleadingjetmass':13,
    'total_jet_mass':14
    }
  elif '0329' in save:
    d_npy['total_jet_mass'] =10
    d_npy['leadingjetmass'] =11
    d_npy['subleadingjetmass'] =12

  return d_npy

#feb files
#d_npy={
#'njets':0,
#'nparticles':1,
#'lny23':2,
#'aplanarity':3,
#'transverse_sphericity':4,
#'sphericity':5,
#'total_jet_mass':6,
#'evIsoSphere':7
#}

#--------------------------- Variable defs
def getpt(jet):
      pt = float(jet[1])/np.cosh(float(jet[2]))
      return pt

def get_three_vec(jet):
      pt = float(jet[1])/np.cosh(float(jet[2]))
      px = pt*np.cos(float(jet[3]))
      py = pt*np.sin(float(jet[3]))
      pz = pt*np.sinh(float(jet[2]))
 
      return [px,py,pz]

def evIsoSphere(particles_vec,spherePoints1,sphereEng1):
  momenta=[]
  engL=[]
  
  for p in particles_vec:
    v = TLorentzVector(0,0,0,0)
    v.SetPtEtaPhiM(getpt(p),float(p[2]),float(p[3]),0.0)
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
      pt = float(jet[1])/np.cosh(float(jet[2]))
      vec.SetPtEtaPhiM(float(pt),float(jet[2]),float(jet[3]),float(jet[4]))
      sumVec = sumVec+vec
    tjm = np.divide(np.power(sumVec.M(),2),np.power(sumP,2))
    return tjm

def lny23(jets):
    if len(jets) > 2:
        jet1_pt = float(jets[0][1])/np.cosh(float(jets[0][2]))
        jet2_pt = float(jets[1][1])/np.cosh(float(jets[1][2]))
        jet3_pt = float(jets[2][1])/np.cosh(float(jets[2][2]))
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

#-----------------------------------------------------------------------------------
def get_sqrts_type(saveTag):
  iden = saveTag.split("_")[0]
  print(iden)
  if '041' in iden or 'tru' in iden or '06' in iden: return 'truth $\sqrt{\hat{s}}$'
  if '513' in iden: return 'measured $\sqrt{\hat{s}}$ (all hadrons)'
  if '531' in iden: return 'measured $\sqrt{\hat{s}}$ (outgoing photon)'


#---------------------------  Plotting help
def draw_hist(model,X_train,X_train_b,X_train_s,Y_train,X_test,Y_test,saveTag):

          # draw this nets s vs. b hist 
          if len(X_train_s) > 0: train_truth_s = model.predict(X_train_s)[:,0]
          else: train_truth_s = []
          train_truth_b = model.predict(X_train_b)[:,0] 
          train_1 =  model.predict(X_train[Y_train[:,0] >0])[:,0] #select signal from train
          train_0 =  model.predict(X_train[Y_train[:,0] <1])[:,0] 
          test_s =  model.predict(X_test[Y_test[:,0] >0])[:,0] #select signal from test
          test_b =  model.predict(X_test[Y_test[:,0] <1])[:,0] 
          #BUGGY 
          #test_s_scaled =  quantile_transform(model.predict(X_test[Y_test[:,0] >0]))[:,0] #select signal from test
          #test_b_scaled =  quantile_transform(model.predict(X_test[Y_test[:,0] <1]))[:,0] 
          test_s_scaled =  quantile_transform(model.predict(X_test))[Y_test[:,0] >0][:,0] #select signal from test
          test_b_scaled =  quantile_transform(model.predict(X_test))[Y_test[:,0] <1][:,0] 

          bins = np.arange(-0.1,1.1,0.02)
          plt.hist(train_truth_b,bins,density=True,label="Train: bkg ("+str(len(train_truth_b))+")",alpha=0.6)
          plt.hist(train_truth_s,bins,density=True,label="Train: sig ("+str(len(train_truth_s))+")",alpha=0.6)
          plt.hist(train_0,bins,density=True,label="Train: 0s ("+str(len(train_0))+")",alpha=0.5,hatch='.',color='green')
          plt.hist(train_1,bins,density=True,label="Train: 1s ("+str(len(train_1))+")",alpha=0.5,hatch='.',color='darkviolet')
          plt.hist(test_b, bins,histtype='step',density=True,label="Test: bkg ("+str(len(test_b))+")",color="b")
          plt.hist(test_s, bins,histtype='step',density=True,label="Test: sig ("+str(len(test_s))+")",color="r")
          plt.hist(test_s_scaled, bins,histtype='step',density=True,label="Scaled test: sig ("+str(len(test_s_scaled))+")",color="r", linestyle="dashed")
          plt.hist(test_b_scaled, bins,histtype='step',density=True,label="Scaled test: bkg ("+str(len(test_b_scaled))+")",color="b", linestyle="dashed")
          np.save(saveTag+"_hist_train_0",train_0)
          np.save(saveTag+"_hist_train_1",train_1)
          np.save(saveTag+"_hist_train_truth_b",train_truth_b)
          np.save(saveTag+"_hist_train_truth_s",train_truth_s)
          np.save(saveTag+"_hist_test_b",test_b)
          np.save(saveTag+"_hist_test_s",test_s)
          np.save(saveTag+"_hist_test_b_scaled",test_b_scaled)
          np.save(saveTag+"_hist_test_s_scaled",test_s_scaled)
          plt.yscale('log')
          plt.legend()
          plt.title('Score Hist: '+saveTag.split("/")[-1])
          plt.xlabel('NN Score')  
          #plt.show()
          plt.savefig(saveTag+"_hist.pdf")
          plt.clf()

  
def make_single_roc(r,Ylabel,rocs,aucs,sigs,saveTag,sizeeach, nInputs):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  np.save(saveTag+"_singleRoc_roc0",rocs[0])
  np.save(saveTag+"_singleRoc_roc1",rocs[1])
  np.save(saveTag+"_singleRoc_aucs",aucs)
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag)
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+'.pdf')
  plt.clf()
  #plt.show()

def make_roc_plots(anomalyRatios,Ylabel,rocs,aucs,sigs,saveTag,finalSaveTag=''):
  if finalSaveTag== '': finalSaveTag = saveTag
  #print('the sigsI have: ', sigs)
  #print('the rocs I have: ', rocs)
  #print('the aucs I have: ', aucs)

  #print('the anomaly ratios I have:')
  for i,r in enumerate(anomalyRatios):
      #print('ar: ', r)
      #Ines plt.plot(rocs[i][1],rocs[i][1]/np.sqrt(rocs[i][0]),label=r'AnomRatio=%0.3f, $\sigma$ = %0.1f, AUC %0.2f'%(anomaly_ratios[i],significances[i],aucs[i])) 
      if 'sqrt' in Ylabel: plt.plot(rocs[i][1],rocs[i][1]/np.sqrt(rocs[i][0]),label=str(100*np.round(r,3))+"% ($\sigma$="+str(prettySigmas[i])+"): AUC="+str(np.round(aucs[i],2)),color=nice_colors_rtob_mut[i]) #tpr, tpr/sqrt(fpr)
      #else: plt.plot(rocs[i][0],rocs[i][1],label=str(100*np.round(r,3))+"% ($\sigma$="+str(prettySigmas[i])+"): AUC="+str(np.round(aucs[i],2))),color=nice_colors_rtob_mut[i]
      else: plt.plot(rocs[i][1],1-rocs[i][0],label=str(100*np.round(r,3))+"% ($\sigma$="+str(prettySigmas[i])+"): AUC="+str(np.round(aucs[i],2)),color=nice_colors_rtob_mut[i])
  plt.xlabel('Signal efficiency (TPR)')
  if 'sqrt' in Ylabel: 
    plt.title('SIC: '+finalSaveTag)
    plt.ylim(0.01,80.0)
    plt.yscale('log')
    plt.ylabel('Signal sensitivity ('+Ylabel+')')
    plt.plot([0,1],[1,1], '--',color='tab:gray')
  else:
    plt.title('ROC: '+finalSaveTag)
    plt.ylabel('Background rejection (1-FPR)')
    plt.plot([0,1],[1,0], '--',color='tab:gray')
  #plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+'.pdf')
  plt.clf()
  #plt.show()

def plot_loss(h,r,save):
      print(h.history)
      plt.plot(h.history['loss'])
      plt.plot(h.history['val_loss'])
      np.save(save+"_loss",h.history['loss'])
      np.save(save+"_valloss",h.history['val_loss'])
      plt.title('model loss, sigma='+str(r))
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(save+'_lossVsEpoch.pdf')
      plt.clf()

#

def make_pfn_plots(sig_records,s700,bg_records,save):
  print('Making plots with name ', save)
  for i in range(2): 
      for j in range(10): # vars per jet to plot
        plot_jetthing(save,sig_records,s700,bg_records,i,j)


def make_var_plots(sig_records,s700,bg_records,save):
  print('Making plots with name ', save)
  #02 files: record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['total_jet_mass'],record['evIsoSphere']
  #plot_something(save,sig_records,bg_records,'truthsqrtshat',range(0,1000,20),1)
  #evt_vars = [record['measuredXpT'],record['xpT_Over_PhpT'], record['ljpT_Over_PhpT'],record['leadingjetpT'], record['subleadingjetpT'],record['measuredphotonpT'],record['njets'],record['nparticles'],record['lny23'],record['aplanarity'],record['transverse_sphericity'],record['sphericity'],record['leadingjetmass'],record['subleadingjetmass'],record['total_jet_mass']]
  #plot_something(save,sig_records,s700,bg_records,'leadingjetmass',np.linspace(0,300,150),1)
  #plot_something(save,sig_records,s700,bg_records,'subleadingjetmass',np.linspace(0,300,150),1)
  plot_something(save,sig_records,s700,bg_records,'njets',np.linspace(0,12,12),1)
  plot_something(save,sig_records,s700,bg_records,'nparticles',np.linspace(0,200,200),1)
  plot_something(save,sig_records,s700,bg_records,'lny23',np.linspace(-10,-0.0001,20),1)
  plot_something(save,sig_records,s700,bg_records,'aplanarity',np.linspace(0,0.5,20),1)
  plot_something(save,sig_records,s700,bg_records,'transverse_sphericity',np.linspace(0,1,50),1)
  plot_something(save,sig_records,s700,bg_records,'sphericity',np.linspace(0,1,50),1)
  #plot_something(save,sig_records,s700,bg_records,'leadingjetpT',np.linspace(0,600,200),1)
  #plot_something(save,sig_records,s700,bg_records,'subleadingjetpT',np.linspace(0,600,200),1)
  plot_something(save,sig_records,s700,bg_records,'measuredphotonpT',np.linspace(0,600,200),1)
  plot_something(save,sig_records,s700,bg_records,'measuredXpT',np.linspace(0,600,200),1)
  plot_something(save,sig_records,s700,bg_records,'ljpT_Over_PhpT',np.linspace(0,40,100),1)
  plot_something(save,sig_records,s700,bg_records,'xpT_Over_PhpT',np.linspace(0,20,100),1)
  plot_something(save,sig_records,s700,bg_records,'total_jet_mass',np.linspace(0,2.0,100),1)
  #plot_something(save,sig_records,s700,bg_records,'splitting',np.linspace(0,0.5,50),1)
  ##plot_something(save,sig_records,bg_records,'evIsoSphere',np.linspace(0,2.0,100),1)
  #plot_something(save,sig_records,bg_records,'thrust_major',np.linspace(0,500,50),1)
  #plot_something(save,sig_records,bg_records,'thrust_minor',np.linspace(0,500,50),1)


def make_sqrts_plot(sig_arr,bkg_arr,sig_arr700,save):
    if '0406' in save or '0513' in save:  
      saveName = 'measuredsqrtshat'
      var = 'Measured $\sqrt{\^{s}}$ (all final state hadrons) [GeV]'
    #elif '0521' in save: var = 'hadronsqrtshat'
    elif '0531' in save: 
      saveName = 'measuredsqrtshatwphoton'
      var = 'Measured $\sqrt{\^{s}}$ (outgoing photon) [GeV]'
    else: 
      saveName = 'truthsqrtshat'
      var = 'Truth $\sqrt{\^{s}}$ [GeV]'
    plt.hist(bkg_arr, np.linspace(0,1000,250), color="steelblue",alpha=0.6,histtype='stepfilled', linewidth=2,label='Background',density=True)
    #plt.hist(sig_arr, np.linspace(0,1000,250), color="tomato", histtype='step',hatch='///', linewidth=2,label='Signal, m$_{X}$ = 350 GeV',density=True)
    plt.hist(sig_arr700, np.linspace(0,1000,250), color="g", histtype='step',hatch='.', linewidth=2,label='Signal, m$_{X}$ = 700 GeV',density=True)
    plt.xlabel(var)
    plt.text(0.1,5.0,'$\it{MadGraph5 + Pythia8 + Delphes3}$',weight='bold')
    #plt.xticks( np.arange(10) )
    plt.yscale('log')
    plt.ylim(0.000001,30.0)
    plt.ylabel("Number of events [A.U.]")
    plt.legend()
    plt.savefig("plots_FINAL/only700"+saveName+".pdf")
    plt.clf()
  


def plot_jetthing(save,sig_records,s700,bg_records,jet,var,doLog=True):

    #if jet == 1: #remove 0 padding 
    #  print('len: ', len(sig_records))
    #  for i in sig_records: 
    #    print(i)
    #    if i[1][0]== 0.0 and i[1][1]==0.0 and  i[1][2] ==0.0 and i[1][3] ==0.0: 
    #      print("A ZERO PAD! ", i[1][0],i[1][1],i[1][2],i[1][3])
    #      sig_records = np.delete(sig_records,i[1],0)
    #  print('len after removal: ', len(sig_records))
    #  for i in sig_records: 
    #    print(i)
    #    if i[1][0]== 0.0 and i[1][1]==0.0 and  i[1][2] ==0.0 and i[1][3] ==0.0: print('STILL A ZERO PAD')

    #print('ALL: len: ', len(sig_records), ' shape ', sig_records.shape)
    #print('sig_records[0] = event:" ', len(sig_records[0]), ', shape: ', sig_records[0].shape)
    #print('sig_records[0][0] = leading jet" ', len(sig_records[0][0]), ', shape: ', sig_records[0][0].shape)
    #sig_records_new = []
    #for event in range(len(sig_records)):
    #  sig_records_new.append([])
    #  print('number of starting jets in this event: ', len(sig_records[event])) 
    #  for j in range(len(sig_records[event])):
    #      jet  = sig_records[event][j]
    #      if jet[0] == 0.0 and jet[1] == 0.0 and jet[2] == 0.0 and jet[3] == 0.0: 
    #        #print('four vector zero padded! event: ', event, ", jet: ", j)  
    #        sig_records_new[event] = np.delete(sig_records[event], j,axis=1 )
    #  print('number of ending jets in this event: ', len(sig_records_new[event])) 
    #    

    #for event in range(len(sig_records_new)):
    #  for j in range(len(sig_records_new[event])):
    #      jet  = sig_record_new[event][j]
    #      if jet[0] == 0.0 and jet[1] == 0.0 and jet[2] == 0.0 and jet[3] == 0.0: print('STILLLLL four vector zero padded! event: ', event, ", jet: ", j)  
    #      #np.delete(sig_records[event][j])

   
    #print('np where: ', np.where(~sig_records.any(axis=1))[0])
    #print('len not all: ', len(sig_records[~np.all(sig_records[:,1] == 0.0)]))
    #sig_records = sig_records[~np.all(sig_records[:,:] == 0, axis=1)]
    #print('len AFTER: ', len(sig_records), ' shape ', sig_records.shape)

    #sig_arr = np.array([float(i[jet][var]) for i in sig_records if not np.allclose(i[:],0)])
    #sig700_arr = np.array([float(i[jet][var]) for i in s700 if not np.allclose(i[:],0)])
    #bkg_arr = np.array([float(i[jet][var]) for i in bg_records if not np.allclose(i[:],0)])
    sig_arr = np.array([float(i[jet][var]) for i in sig_records if not np.all((i[jet]==0))])
    sig700_arr = np.array([float(i[jet][var]) for i in s700 if not np.all((i[jet]==0))])
    bkg_arr = np.array([float(i[jet][var]) for i in bg_records if not np.all((i[jet]==0))]) 
   
    prettyLabel = jet_dict[var][0]
    R = jet_dict[var][1]
    
    plt.hist(bkg_arr, R, color="steelblue",alpha=0.6,histtype='stepfilled', linewidth=2,label='Background',density=True)
    plt.hist(sig_arr, R, color="tomato",hatch='///', histtype='step', linewidth=2,label='Signal, m$_{X}$ = 350 GeV',density=True)
    plt.hist(sig700_arr, R, color="g",hatch='.', histtype='step', linewidth=2,label='Signal, m$_{X}$ = 700 GeV',density=True)
    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
    #plt.xscale('log')
    #plt.xticks(R)

    if jet ==0: plt.xlabel('Leading jet '+prettyLabel)
    elif jet== 1: plt.xlabel('Subleading jet '+prettyLabel)
    else: plt.xlabel('Jet '+prettyLabel)
    if doLog == True: plt.yscale('log')
    plt.ylabel("Number of events [A.U.]")
    plt.ylim(jet_dict[var][2],jet_dict[var][3])
    plt.text(jet_dict[var][4],jet_dict[var][5],'$\it{MadGraph5 + Pythia8 + Delphes3}$',weight='bold')
    plt.legend()
    plt.savefig("plots_FINAL/"+str(save)+"_jet"+str(jet)+"_var"+str(var)+".pdf")
    plt.clf()

def plot_something(save,sig_records,s700,bg_records,var,R,doLog):
  
    d_npy = get_npy_dict(save) 
    #plt.figure(figsize=(20,5))
    if 'ljpT' in var:
      sig_arr = np.array([float(i[d_npy['leadingjetpT']]/i[d_npy['measuredphotonpT']]) for i in sig_records])
      sig700_arr = np.array([float(i[d_npy['leadingjetpT']]/i[d_npy['measuredphotonpT']]) for i in s700])
      bkg_arr = np.array([float(i[d_npy['leadingjetpT']]/i[d_npy['measuredphotonpT']]) for i in bg_records])    
    elif 'xpT' in var:
      sig_arr = np.array([float(i[d_npy['measuredXpT']]/i[d_npy['measuredphotonpT']]) for i in sig_records])
      sig700_arr = np.array([float(i[d_npy['measuredXpT']]/i[d_npy['measuredphotonpT']]) for i in s700])
      bkg_arr = np.array([float(i[d_npy['measuredXpT']]/i[d_npy['measuredphotonpT']]) for i in bg_records])    
    else:
      if 'npy' in save: 
        sig_arr = np.array([float(i[d_npy[var]]) for i in sig_records])
        sig700_arr = np.array([float(i[d_npy[var]]) for i in s700])
        bkg_arr = np.array([float(i[d_npy[var]]) for i in bg_records])    
      else:
        sig_arr = np.array([i[var] for i in sig_records])
        sig700_arr = np.array([i[var] for i in s700])
        bkg_arr = np.array([i[var] for i in bg_records])    

    plt.hist(bkg_arr, R, color="steelblue", alpha=0.6,histtype='stepfilled', linewidth=2,label='Background',density=True)
    plt.hist(sig_arr, R, color="tomato", hatch='///',histtype='step', linewidth=2,label='Signal, m$_{X}$ = 350 GeV',density=True)
    plt.hist(sig700_arr, R, color="g", hatch='.',histtype='step', linewidth=2,label='Signal, m$_{X}$ = 700 GeV',density=True)
    #plt.hist(this_arr, bins=np.logspace(1.5,3,30))
    #plt.xscale('log')
    #plt.xticks(R)

    plt.xlabel(get_pretty[var][0])
    if doLog == True: plt.yscale('log')
    plt.ylabel("Number of events [A.U.]")
    plt.ylim(get_pretty[var][1],get_pretty[var][2])
    #plt.ylabel("Number of Events / bin")
    plt.text(get_pretty[var][3],get_pretty[var][4],'$\it{MadGraph5 + Pythia8 + Delphes3}$',weight='bold')
    plt.legend()
    plt.savefig("plots_FINAL/"+save+"_"+var+".pdf")
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
    plt.savefig("plots/plt_"+index+".pdf")
    plt.clf()

