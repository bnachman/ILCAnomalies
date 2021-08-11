import numpy as np 
import os
import glob
import argparse
import time

d_maxLines={
"processed_background_randomseeds_bigger1_noZ.txt":[ "6231328", "xenia06.nevis.columbia.edu"  ,     "/data0/atlas/dq2/rucio/user/jgonski/f0/ca/processed_background_randomseeds_bigger1_noZ.txt"]  
,"processed_background_randomseeds_bigger1.txt":[    "2776081", "xenia06.nevis.columbia.edu"     ,  "/data0/atlas/dq2/rucio/user/jgonski/ff/17/processed_background_randomseeds_bigger1.txt"]
,"processed_background_randomseeds_bigger2_noZ.txt":["1766308", "xenia06.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/6c/90/processed_background_randomseeds_bigger2_noZ.txt"]
,"processed_background_randomseeds_bigger2.txt":[    "818319", "xenia06.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/2d/df/processed_background_randomseeds_bigger2.txt"]
,"processed_background_randomseeds_bigger3_noZ.txt":["1701664", "xenia06.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/02/7c/processed_background_randomseeds_bigger3_noZ.txt"]
,"processed_background_randomseeds_bigger3.txt":[    "600203", "xenia06.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/e9/7e/processed_background_randomseeds_bigger3.txt"]
,"processed_background_randomseeds_bigger4_noZ.txt":["1866266", "xenia06.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/49/dd/processed_background_randomseeds_bigger4_noZ.txt"]
,"processed_background_randomseeds_bigger4.txt":[    "739050", "xenia06.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/58/54/processed_background_randomseeds_bigger4.txt"]
,"processed_background_randomseeds_bigger5_noZ.txt":["1803221", "xenia19.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/ba/bd/processed_background_randomseeds_bigger5_noZ.txt"]
,"processed_background_randomseeds_bigger5.txt":[    "960553", "xenia06.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/e7/00/processed_background_randomseeds_bigger5.txt"]
,"processed_background_randomseeds_bigger6_noZ.txt":["1617234", "xenia20.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/b1/07/processed_background_randomseeds_bigger6_noZ.txt"]
,"processed_background_randomseeds_bigger6.txt":[    "792983", "xenia20.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/49/62/processed_background_randomseeds_bigger6.txt"]
,"processed_background_randomseeds_bigger7_noZ.txt":["1820783", "xenia20.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/78/67/processed_background_randomseeds_bigger7_noZ.txt"]
,"processed_background_randomseeds_bigger7.txt":[    "929055", "xenia20.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/38/84/processed_background_randomseeds_bigger7.txt"]
,"processed_background_randomseeds_bigger8_noZ.txt":["1776227", "xenia20.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/1c/02/processed_background_randomseeds_bigger8_noZ.txt"]
,"processed_background_randomseeds_bigger8.txt":[    "774294", "xenia20.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/87/88/processed_background_randomseeds_bigger8.txt"]
,"processed_background_randomseeds_bigger9_noZ.txt":["1713670", "xenia20.nevis.columbia.edu" ,      "/data0/atlas/dq2/rucio/user/jgonski/f8/cb/processed_background_randomseeds_bigger9_noZ.txt"]
,"processed_background_randomseeds_bigger9.txt":[    "610636", "xenia20.nevis.columbia.edu"     ,   "/data0/atlas/dq2/rucio/user/jgonski/ee/ec/processed_background_randomseeds_bigger9.txt"]
}
#-------------------------------------------------------------------------
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("-f", "--fileName", default = '', type=str,
                      help="file name")
   parser.add_argument("-r", "--runType", default = '', type=str, 
                      help="pfn or evt")
   args = parser.parse_args()
   #myFile = args.fileName[0]
   runType = args.runType


   files = glob.glob("/data/users/jgonski/Snowmass/LHE_txt_fils/*bigger*.txt")

   increment = 50000

   for myFile in files:
     # SUBMIT HERE

     shortName = myFile.split('/')[-1]
     if 'lhe' in shortName: maxLines = 10000
     else: maxLines = int(d_maxLines[shortName][0])
     
     for i in range(0,maxLines,increment): # number of events in largest file
       print("SUBMITTING: "+shortName+" from  "+str(i)+" to "+str(i+increment))
       args = open("args.txt","w")
       if 'lhe' not in myFile: os.system("echo '0606_final "+d_maxLines[shortName][2]+" " + str(i) + " " +str(i+increment)+" "+runType+"'>>  args.txt")
       else: os.system("echo '0606_final "+shortName+" " + str(i) + " " +str(i+increment)+" "+runType+"'>>  args.txt")
       args.close()
       open("submit.sub","w")

       os.system("echo '#!/bin/bash' >> submit.sub")
       os.system("echo 'executable            = condor_parse.sh' >> submit.sub") ##expand here to PC 
       os.system("echo 'output                = logs.$(ClusterId).$(ProcId).out' >> submit.sub")
       os.system("echo 'error                 = logs.$(ClusterId).$(ProcId).err' >> submit.sub")
       os.system("echo 'log                   = logs.$(ClusterId).log' >> submit.sub")
       os.system("echo 'universe         = vanilla' >> submit.sub")
       #os.system("echo 'getenv           = True' >> submit.sub")
       #os.system("echo 'Rank            = Mips' >> submit.sub")
       os.system("echo '' >> submit.sub")
       if not 'lhe' in myFile: os.system("echo 'Requirements = (machine == \""+d_maxLines[shortName][1]+ "\")' >> submit.sub")
       os.system("echo 'should_transfer_files = YES' >> submit.sub")
       os.system("echo 'when_to_transfer_output = ON_EXIT' >> submit.sub")
       os.system("echo 'initialdir = /data/users/jgonski/Snowmass/ILCAnomalies_fork/condor' >> submit.sub")
       if 'lhe' in myFile: os.system("echo 'transfer_input_files = $(initialdir)/condor_parse.sh, $(initialdir)/parseCondor.py, $(initialdir)/../eventHelper.py, "+myFile+"' >> submit.sub")  
       else: os.system("echo 'transfer_input_files = $(initialdir)/condor_parse.sh, $(initialdir)/parseCondor.py, $(initialdir)/../eventHelper.py' >> submit.sub")  
       os.system("echo 'queue arguments from args.txt' >> submit.sub")

       os.system("condor_submit submit.sub")
       time.sleep(0.2)
       

       print("DONE SUBMITTING... ")
