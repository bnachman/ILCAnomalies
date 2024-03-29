# ILCAnomalies

## Data generation 
Turns Delphes output into text file (`Generation/myprocess.C`)

## Data processing 
Turns text files into numpy with corrrect format for training.
Pre-made numpy files can be found on Zenodo (TODO: link).

`python condor/parseCondor.py '0810_test' LHE_txt_fils/processed_lhe_signal_700_fixed.txt 0 1000 evt`
* 1 = tag for output files 
* 2 = input text file 
* 3 = start event 
* 4 = end event 
* 5 = type of processing (evt or pfn)

## Plots 
Makes plots of collison center-of-mass energy and input variables used in training.

`python makePlots.py -t yourDataTag -s pfn`
* s = pfn or evt 
* t = prefix of npy files (optional, default = '')


## Training 
Does training and makes output ROC/SIC curves.

`python PFNLevel.py -n 'yourDataTag' -sig 350 -tr CWoLa -te SvsB -s 25000 -r 1 -e 1 -DEBUG 0`

`python EvtLevel.py -n 'yourDataTag' -sig 350 -tr CWoLa -te SvsB -s 25000`

* n = data tag, stylized as `inputFilePrefix_saveName`
* sig = signal to train with (350 or 700)
* tr = training type (CWoLa or benchmark)
* te = test set (SvsB or BvsB)
* s = number of background events to be used in the sideband training region 
* r = do random signal injections\*
* e = do ensembling\*
* DEBUG = add various helpful print statements\*

\* PFN only option


