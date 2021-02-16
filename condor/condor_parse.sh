#!/bin/bash
mkdir event_isotropy
tar -xzf event_isotropy.tar.gz
cd event_isotropy
mv ../*txt .
python parseCondor.py $1 $2 
mv *npy ../
rm -rf event_isotropy

