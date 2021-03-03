#!/bin/bash
mkdir event_isotropy
tar -xzf event_isotropy.tar.gz
cd event_isotropy
mv ../*bigger*txt .
python parseCondor.py $1 $2 $3 $4
mv *npy ../
rm -rf event_isotropy

