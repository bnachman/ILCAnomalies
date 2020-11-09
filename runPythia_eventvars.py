#!/usr/bin/python
# main01.py is a part of the PYTHIA event generator.
# Copyright (C) 2016 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
#
# This is a simple test program. It fits on one slide in a talk.  It
# studies the charged multiplicity distribution at the LHC. To set the
# path to the Pythia 8 Python interface do either (in a shell prompt):
#      export PYTHONPATH=$(PREFIX_LIB):$PYTHONPATH
# or the following which sets the path from within Python.
import sys
cfg = open("Makefile.inc")
lib = "../lib"
for line in cfg:
    if line.startswith("PREFIX_LIB="): lib = line[11:-1]; break
sys.path.insert(0, lib)
#import pythia
import pythia8
sys.path.insert(0,"python/")
#from Variables import *
              
from array import array

# Import the Pythia module.                                                                                             
import pythia8
import math
from math import sqrt,log
from ROOT import (TH1F,TCanvas,TLegend)
import ROOT
import numpy as np
from scipy import linalg as la
from array import array


pythia = pythia8.Pythia()

sph = pythia8.Sphericity()
thr = pythia8.Thrust()
lund = pythia8.ClusterJet()
jade = pythia8.ClusterJet()
durham = pythia8.ClusterJet()

thrust = array('f',[0])
sphericity = array('f',[0])

physics = 'bkgnd'
histograms = []
output = ROOT.TFile(physics+".root","RECREATE")
tree = ROOT.TTree('T','T')
tree.Branch('thrust',thrust,'thrust/F')
tree.Branch('sphericity',sphericity,'sphericity/F')
#tree.Branch('aplanarity',&aplanarity)

# Histograms.
hist_thrust   = TH1F('thrust','thrust', 100, 0.5, 1.)
hist_sphericity   = TH1F('sphericity','sphericity', 100, 0.0, 1.)
#hist_aplanarity   = TH1F('aplanarity','aplanarity', 100, 0.5, 1.)

# Input pythia cmnd 
pythia.readString("Main:numberOfEvents = 10000         ! number of events to generate")
pythia.readString("Main:timesAllowErrors = 300          ! how many aborts before run stops")
pythia.readString("Init:showChangedSettings = on      ! list changed settings")
pythia.readString("Init:showChangedParticleData = off ! list changed particle data")
pythia.readString("Next:numberCount = 100             ! print message every n events")
pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = /data/users/jgonski/snowmass/MG5_aMC_v2_7_3/0911_eeajj/Events/run_01/unweighted_events.lhe")
pythia.readString("PartonLevel:Remnants = off")
pythia.readString("Check:epTolErr = 1e-2")

pythia.init()


# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0,1000):
    if iEvent % 100 == 0:print '%s' %iEvent 
    
    if not pythia.next(): continue
    
    ## calculate event level quantities
    sph.analyze( pythia.event )
    thr.analyze( pythia.event )

    hist_thrust.Fill(thr.thrust())
    hist_sphericity.Fill(sph.sphericity())
    thrust[0] = thr.thrust()
    sphericity[0] = sph.sphericity()
    tree.Fill()   

### after event loop
output.Write()
output.Close()
                    

