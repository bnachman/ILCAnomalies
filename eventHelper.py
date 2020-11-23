#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import glob
from ROOT import *


# # Defining event level variables
# IO: arXiv:1206.2135.pdf
# JG: https://arxiv.org/pdf/1811.00588.pdf (total jet mass)

#--------------------------- Variable defs
def get_three_vec(jet):
      pt = float(jet[1])/np.cosh(float(jet[2]))
      px = pt*np.cos(float(jet[3]))
      py = pt*np.sin(float(jet[3]))
      pz = pt*np.sinh(float(jet[2]))
 
      return [px,py,pz]

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
        #[index, p [GeV], eta, phi, m]
        px,py,pz = get_three_vec(jet)
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
# adapted from the jet-level thrust axis calculation, which uses constituents. See eg. https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Reconstruction/Jet/JetSubStructureUtils/Root/Thrust.cxx 
def thrust(jets):

  thrust_major = -999
  thrust_minor = -999
  useThreeD = True

  if len(jets) < 2: return [thrust_major,thrust_minor]

  #agree = 0
  #disagree = 0
  #max_tests = 2 #TODO
  n_tests = 0
  #n_0 = [TVector3(0.,0.,0.),TVector3(0.,0.,0.),TVector3(0.,0.,0.),TVector3(0.,0.,0.)] 
  n_0 = [0.,0.,0.]
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

  n_0 =  (add0[n_tests] * [px0,py0,pz0] + add1[n_tests]*[px1,py1,pz1])
  #print('Thrust axis n_0: ', n_0[0], n_0[1], n_0[2])
 
    #if useThreeD==False: n_0.SetZ(0.0)

    #protect against small number of input particles (smaller than 4!)
    #if (n_0[n_tests].Mag() > 0)
    #  n_0[n_tests] *= 1/n_0[n_tests].Mag();
  
    #--------- SKIP FOR NOW: take only two hardest jets
    # ------- Determine n_1 = include all particles
    #run = False
    #loop = 0 
    #while run: 
    #  n_1 = TVector3(0.,0.,0.)
    #  #loop over all jets this time: 
    #  for j in range(len(jets)):
    #    if (float(jets[j][1])*np.cos(float(jets[j][3]))/np.cosh(float(jets[j][2])) * n_0[n_tests].X() 
    #      +float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])) * n_0[n_tests].Y()
    #      +float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])) * n_0[n_tests].Z()) > 0: n_1 += TVector3(float(jets[j][1])*np.cos(float(jets[j][3]))/np.cosh(float(jets[j][2])), float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])), float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])))
    #    else: n_1 -= TVector3(float(jets[j][1])*np.cos(float(jets[j][3]))/np.cosh(float(jets[j][2])), float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])), float(jets[j][1])*np.sin(float(jets[j][3]))/np.cosh(float(jets[j][2])))
    #
    #  if useThreeD==False: n_1[n_tests].SetZ(0.0)
    #  #protect against small number of input particles (smaller than 4!)
    #  #if (n_1[n_tests].Mag() > 0)
    #  #  n_1[n_tests] *= 1/n_1[n_tests].Mag();
  

    #  # has axis changed ? if so, try at most ten times (thrust axis has two fold ambiguity)
    #  run = (n_0[n_tests] != n_1) and (-n_0[n_tests] != n_1) and loop < 10
    #  n_0[n_tests] = n_1
    #  while run: 
    #      # agrees or disagrees with first result ?
    #        #  thrust has a sign ambiguity
    #        if (n_tests > 0 and (n_0[0] == n_0[n_tests] or n_0[0] == -n_0[n_tests])) agree+=1
    #        if (n_tests > 0 and  n_0[0] != n_0[n_tests] and n_0[0] != -n_0[n_tests])  disagree+=1
         
  # now that we have the thrust axis, we determine the thrust value
  #  if the various calculations of the thrust axes disagree, try all
  #  and take the maximum, calculate minor and mayor axis
  n_tests=0
  #while n_tests < max_tests:
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
      numerator_t += abs(np.dot(c,n_0))
      numerator_m += np.linalg.norm(np.cross(c,n_0))
      denominator += np.linalg.norm(c)
  inv_denominator = 1. / denominator
  if numerator_t * inv_denominator > thrust_major: 
      thrust_major = numerator_t * inv_denominator
      thrust_minor = numerator_m * inv_denominator
  #n_tests += 1

  return [thrust_major,thrust_minor]

#def planarity(w,v):
#    return



#---------------------------  Plotting help

def plot_something(sig_records,bg_records,var,R,doLog):
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

