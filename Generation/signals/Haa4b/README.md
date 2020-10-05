# README

## Run ME generation
./bin/mg5_aMC run_ee_${dir}.sh 

## Prepare for showering
gzip -d ee_${dir}/Events/run_01/unweighted_events.lhe.gz
sed -i 's+25  1    1    2+35  1    1    2+' ee_${dir}/Events/run_01/unweighted_events.lhe
gzip ee_${dir}/Events/run_01/unweighted_events.lhe

## Shower e.g.
##cd ee_${dir}/Events/run_01
##DYLD_LIBRARY_PATH=/Users/inesochoa/PhysicsWorkdir/MG5_aMC_v2_7_3/HEPTools/lib:$DYLD_LIBRARY_PATH /Users/inesochoa/PhysicsWorkdir/MG5_aMC_v2_7_3/HEPTools/MG5aMC_PY8_interface/MG5aMC_PY8_interface /Users/inesochoa/PhysicsWorkdir/MG5_aMC_v2_7_3/my_pythia8_cards/ee_${dir}.cmnd >& pythia8.log
