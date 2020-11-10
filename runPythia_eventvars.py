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
thrust = array('f',[0])
sphericity = array('f',[0])
aplanarity = array('f',[0])

physics = 'signal'
histograms = []
output = ROOT.TFile(physics+".root","RECREATE")
tree = ROOT.TTree('T','T')
tree.Branch('thrust',thrust,'thrust/F')
tree.Branch('sphericity',sphericity,'sphericity/F')
tree.Branch('aplanarity',aplanarity,'aplanarity/F')

# Input pythia cmnd 
#pythia.readFile("config_bkgnd.cmnd")
pythia.readFile("config_sig.cmnd")

pythia.init()

# Begin event loop. Generate event. Skip if error. List first one.
for iEvent in range(0,1000):
    if iEvent % 100 == 0:print '%s' %iEvent 
    
    if not pythia.next(): continue
    
    ## calculate event level quantities
    sph.analyze( pythia.event )
    thr.analyze( pythia.event )

    thrust[0] = thr.thrust()
    sphericity[0] = sph.sphericity()
    aplanarity[0] = sph.aplanarity()
    tree.Fill()   

### after event loop
output.Write()
output.Close()
                    

