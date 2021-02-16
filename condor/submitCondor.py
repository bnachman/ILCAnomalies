import numpy as np 
import os
import glob
import argparse

#-------------------------------------------------------------------------
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-f", "--fileName", default = [], type=str, nargs='+',
                      help="file name")
   args = parser.parse_args()
   myFile = args.fileName[0]


   # SUBMIT HERE
   print("SUBMITTING: "+myFile)
   args = open("args.txt","w")
   os.system("echo 0216_condor_test "+myFile+" >>  args.txt")
   args.close()
   open("submit.sub","w")

   os.system("echo '#!/bin/bash' >> submit.sub")
   os.system("echo 'executable            = condor_parse.sh' >> submit.sub") ##expand here to PC 
   os.system("echo 'output                = logs.$(ClusterId).$(ProcId).out' >> submit.sub")
   os.system("echo 'error                 = logs.$(ClusterId).$(ProcId).err' >> submit.sub")
   os.system("echo 'log                   = logs.$(ClusterId).log' >> submit.sub")
   os.system("echo 'universe         = vanilla' >> submit.sub")
   os.system("echo 'getenv           = True' >> submit.sub")
   os.system("echo 'Rank            = Mips' >> submit.sub")
   os.system("echo '' >> submit.sub")
   #os.system("echo '#some other stuff from Bill's twiki' >> submit.sub")
   #os.system("echo 'Requirements = (machine != \"xenia00.nevis.columbia.edu\")' >> submit.sub")
   os.system("echo 'should_transfer_files = YES' >> submit.sub")
   os.system("echo 'when_to_transfer_output = ON_EXIT' >> submit.sub")
   os.system("echo 'initialdir = /data/users/jgonski/Snowmass/ILCAnomalies_fork/condor' >> submit.sub")
   #os.system("echo 'sampledir = /nevis/xenia/data/users/jgonski/xbb/Xbb_merged_samples/0121_PCJKDL1r' >> submit.sub")
   os.system("echo 'workdir = /data/users/jgonski/Snowmass/LHE_txt_fils/' >> submit.sub")
   os.system("echo 'transfer_input_files = $(initialdir)/condor_parse.sh, $(initialdir)/event_isotropy.tar.gz, $(workdir)/"+myFile+" ' >> submit.sub")
   os.system("echo 'queue arguments from args.txt' >> submit.sub")

   os.system("condor_submit submit.sub")
   #time.sleep(.2)
   
   #open('submit.sub', 'w').close()

   print("DONE SUBMITTING... ")
