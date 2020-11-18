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
def total_jet_mass(jets):
    sumVec = TLorentzVector(0.,0.,0.,0.)
    sumP = 0.0
    for jet in jets:
      vec = TLorentzVector(0.,0.,0.,0.)
      sumP += float(jet[1])
      pt = float(jet[1])/np.cosh(float(jet[2]))
      vec.SetPtEtaPhiM(float(pt),float(jet[2]),float(jet[3]),float(jet[4]))
      sumVec = sumVec+vec
    return np.divide(np.power(sumVec.M(),2),np.power(sumP,2))

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

