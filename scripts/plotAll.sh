#python makePlots.py -t 0416_allPFN
#python makePlots.py -t 0513_measuredsqrtshat
#python makePlots.py -t 0531_measuredsqrtshatwphoton
python makeFinalPFNPlots.py -n '0416_0605FINAL_0615' -tr CWoLa -te SvsB -s 25000 -r 1 
python makeFinalPFNPlots.py -n '0416_0608FINAL_0615s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'
python makeFinalPFNPlots.py -n '0513_0605FINAL_0615' -tr CWoLa -te SvsB -s 25000 -r 1 
python makeFinalPFNPlots.py -n '0513_0605FINAL_0615s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'
python makeFinalPFNPlots.py -n '0531_0605FINAL_0615' -tr CWoLa -te SvsB -s 25000 -r 1 
python makeFinalPFNPlots.py -n '0531_0605FINAL_0615s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700'
