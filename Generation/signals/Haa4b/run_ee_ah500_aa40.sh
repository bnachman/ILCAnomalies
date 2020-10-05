import sm-full
generate e+ e- > a h
output ee_ah500_aa40

launch

shower=OFF
madspin=OFF
analysis=OFF

set ebeam1 500.0
set ebeam2 500.0

set pta 10.0
set etaa 5.0

set ptj 10.0
set etaj 5.0

set param_card mass 25 5.00000e+02
set param_card decay 25 0.00407

#for pythia8 card
#partonlevel:mpi = off
#
#Higgs:useBSM = on
#35:m0 = 500.0
#35:mWidth = 0.00407
#35:doForceWidth = on
#35:onMode = off
#35:onIfMatch = 36 36  
#
#36:onMode = off
#36:onIfAny = 5
#36:m0 = 40
#36:tau0 = 0