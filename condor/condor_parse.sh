#!/bin/bash
# Set up the Nevis environment, including our custom environment modules.
export PATH=/sbin:/usr/sbin:/bin:/usr/bin
source /usr/nevis/adm/nevis-init.sh
module load root

python parseCondor.py $1 $2 $3 $4 $5

