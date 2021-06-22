#python makePlots.py -t 0416_allPFN
#python makePlots.py -t 0513_measuredsqrtshat
#python makePlots.py -t 0531_measuredsqrtshatwphoton
#python makeFinalPFNPlots.py -n '0416_0605FINAL_0621' -tr CWoLa -te SvsB -s 25000 -r 1 
###python makeFinalPFNPlots.py -n '0416_0621new_0621' -tr CWoLa -te SvsB -s 25000 -r 1 
#python makeFinalPFNPlots.py -n '0416_0608FINAL_0621s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'
#truth SBs
#python makeFinalPFNPlots.py -n '0513_0605FINAL_0622' -tr CWoLa -te SvsB -s 25000 -r 1 > 0622_0513_s350.log
#python makeFinalPFNPlots.py -n '0513_0605FINAL_0622_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'  > 0622_0513_s700.log
#python makeFinalPFNPlots.py -n '0531_0605FINAL_0622' -tr CWoLa -te SvsB -s 25000 -r 1 > 0622_0531_s350.log
#python makeFinalPFNPlots.py -n '0531_0605FINAL_0622_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0622_0531_s700.log 
# wide SBs
python makeFinalPFNPlots.py -n '0513_0619wideSB_0622wideEval' -tr CWoLa -te SvsB -s 25000 -r 1 > 0622_0513_s350.log
python makeFinalPFNPlots.py -n '0513_0619wideSB_0622wideEval_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'  > 0622_0513_s700.log
python makeFinalPFNPlots.py -n '0531_0619wideSB_0622wideEval' -tr CWoLa -te SvsB -s 25000 -r 1 > 0622_0531_s350.log
python makeFinalPFNPlots.py -n '0531_0619wideSB_0622wideEval_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0622_0531_s700.log 

