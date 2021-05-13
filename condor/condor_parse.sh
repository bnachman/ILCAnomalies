#!/bin/bash
# Set up the Nevis environment, including our custom environment modules.
export PATH=/sbin:/usr/sbin:/bin:/usr/bin
source /usr/nevis/adm/nevis-init.sh
module load root

echo $PATH
echo $LD_LIBRARY_PATH
printenv | grep xenia
#mkdir event_isotropy
#tar -xzf event_isotropy.tar.gz
#cd event_isotropy
#mv ../eventiso_condor.tar.gz .
#mkdir -p eventiso_condor
#tar -xzf eventiso_condor.tar.gz -C eventiso_condor
#mv ../processed*txt .
#eventiso_condor/bin/python parseCondor.py $1 $2 $3 $4
#mv *npy ../
#rm -rf event_isotropy
#python parseCondor.py $1 $2 $3 $4
python parsePFN.py $1 $2 $3 $4

