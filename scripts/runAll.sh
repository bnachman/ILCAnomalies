#-- June 18: wider sidebands
python PFNLevel.py -n '0513_0619wideSB' -tr CWoLa -te SvsB -s 25000 -w 1 -r 1  > 0513_0619wideSB_CWoLa.log
python PFNLevel.py -n '0513_0619wideSB_s700' -tr CWoLa -te SvsB -s 25000 -w 1 -r 1 -sig '700' > 0513_0619wideSB_CWoLa_s700.log
python PFNLevel.py -n '0531_0619wideSB' -tr CWoLa -te SvsB -s 25000 -w 1 -r 1  > 0531_0619wideSB_CWoLa.log
python PFNLevel.py -n '0531_0619wideSB_s700' -tr CWoLa -te SvsB -s 25000 -w 1 -r 1 -sig '700' > 0531_0619wideSB_CWoLa_s700.log

#---- FINAL 
#python PFNLevel.py -n '0416_0618FINAL' -tr CWoLa -te SvsB -s 25000 -r 1 > 0416_0618FINAL_CWoLa.log
#python PFNLevel.py -n '0416_0618FINAL_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0416_0618FINAL_s700_CWoLa.log
#python PFNLevel.py -n '0513_0618FINAL' -tr CWoLa -te SvsB -s 25000 -r 1 > 0513_0618FINAL_CWoLa.log
#python PFNLevel.py -n '0513_0618FINAL_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0513_0618FINAL_s700_CWoLa.log
#python PFNLevel.py -n '0531_0618FINAL' -tr CWoLa -te SvsB -s 25000 -r 1 > 0531_0618FINAL_CWoLa.log
#python PFNLevel.py -n '0531_0618FINAL_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0531_0618FINAL_s700_CWoLa.log


##- truthsqrtshat
#python PFNLevel.py -n '0416_0526adam10-5' -tr CWoLa -te SvsB -s 25000 -r 1 > 0526adam10-5_random_25k_CWoLa.log
#python PFNLevel.py -n '0416_0524removedBadNNs_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0524removedBadNNs_random_25k_CWoLa_s700.log
##- measuredsqrtshat
#python PFNLevel.py -n '0513_0529epochs100_' -tr CWoLa -te SvsB -s 25000 -r 1 > 0513_0529epochs100_random_25k_CWoLa.log
#python PFNLevel.py -n '0513_0529epochs100_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0513_0529epochs100_random_25k_CWoLa_s700.log
##- hadronsqrtshat
#python PFNLevel.py -n '0521_0524removedBadNNs_' -tr CWoLa -te SvsB -s 25000 -r 1 > 0521_0524removedBadNNs_random_25k_CWoLa.log
#python PFNLevel.py -n '0521_0524removedBadNNs_s700' -tr CWoLa -te SvsB -s 25000 -r 1 -sig '700' > 0521_0524removedBadNNs_random_25k_CWoLa_s700.log

#python PFNLevel.py -n '0416_0511adam50mods_random_50k' -tr CWoLa -te SvsB -s 25000 -r 1 > 0511adam50mods_random_50k_CWoLa.log
#python PFNLevel.py -n '0416_0511adam50mods_random_50k' -tr benchmark -te SvsB -s 25000 -r 1 > 0511adam50mods_random_50k_benchmark.log
#python PFNLevel.py -n '0416_0511adam50mods_random_50k' -tr CWoLa -te BvsB -s 25000 -r 1 > 0511adam50mods_random_50k_CWoLa_BvsB.log

#python PFNLevel.py -n '0416_0511RMSprop_random_1k' -tr CWoLa -te SvsB -s 1000 -r 1 > 0511RMSprop_random_1k_CWoLa.log
#python PFNLevel.py -n '0416_0511RMSprop_random_5k' -tr CWoLa -te SvsB -s 5000 -r 1 > 0511RMSprop_random_5k_CWoLa.log
#python PFNLevel.py -n '0416_0511RMSprop_random_10k' -tr CWoLa -te SvsB -s 10000 -r 1 > 0511RMSprop_random_10k_CWoLa.log
#python PFNLevel.py -n '0416_0511RMSprop_random_25k' -tr CWoLa -te SvsB -s 25000 -r 1 > 0511RMSprop_random_25k_CWoLa.log
#python PFNLevel.py -n '0416_0511RMSprop_random_50k' -tr CWoLa -te SvsB -s 25000 -r 1 > 0511RMSprop_random_50k_CWoLa.log

#python PFNLevel.py -n '0416_0507debug_ensemb' -tr CWoLa -te SvsB -s 75000 > 0507_debug_ensemb_CWoLa.log
#python PFNLevel.py -n '0416_0506myenv_ensemb' -tr benchmark -te SvsB -s 75000 > 0506_myenv_ensemb_benchmark.log
#python PFNLevel.py -n '0416_0506myenv_random' -tr CWoLa -te SvsB -s 75000 -r 1 > 0506_myenv_random_CWoLa.log
#python PFNLevel.py -n '0416_0506myenv_random' -tr benchmark -te SvsB -s 75000 -r 1 > 0506_myenv_random_benchmark.log

#python PFNLevel.py -n '0416_0504ensemb' -tr CWoLa -te SvsB -s 75000
#python PFNLevel.py -n '0416_0504ensemb' -tr benchmark -te SvsB -s 75000
#python PFNLevel.py -n '0416_0504ensemb' -tr CWoLa -te BvsB -s 75000
#python PFNLevel.py -n '0416_0504ensemb' -tr benchmark -te BvsB -s 75000
#python PFNLevel.py -n '0416_0504random' -tr CWoLa -te SvsB -s 75000 -e 0 -r 1
#python PFNLevel.py -n '0416_0504random' -tr benchmark -te SvsB -s 75000 -e 0 -r 1
#python PFNLevel.py -n '0416_0504random' -tr CWoLa -te BvsB -s 75000 -e 0 -r 1
#python PFNLevel.py -n '0416_0504random' -tr benchmark -te BvsB -s 75000 -e 0 -r 1
