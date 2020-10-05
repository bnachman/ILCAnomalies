# README

## Run ME generation
- e.g. dir=ah350_aa40
- ./bin/mg5_aMC run_ee_${dir}.sh 

## Prepare for showering
- gzip -d ee_${dir}/Events/run_01/unweighted_events.lhe.gz
- sed -i 's+25  1    1    2+35  1    1    2+' ee_${dir}/Events/run_01/unweighted_events.lhe
- gzip ee_${dir}/Events/run_01/unweighted_events.lhe

## Shower
- use ee_${dir}.cmnd
