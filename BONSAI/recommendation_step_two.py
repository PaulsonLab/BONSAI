#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:15:59 2024

Once the search process is completed, we turn to the second step, which is recommendation 
of the designs observed during the search process. There are 4 recommenders in this script:
    
    1) ARBO_recommender : This is called GP (Beta = 0) and GP-Quantile(beta = 2) in the paper
    2) BOFN_recommender: This is the nominal recommendation that ignores the uncertain variable
    3) Mean_recommender: This is called GPFN in the paper
    4) Quantile_Recommender: This is called GPFN-Quantile in the paper
    
    Typically, the recommender is used considering all the points, we are doing it every 5 
    iterations for analysis. The recommendation procedure is a computationally expensive process,
    so...  
    
    
    Select your recommender wisely! 

@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt
from robust_algorithms import ARBO_Recommendor_final, BOFN_Recommendor_final, Mean_Recommendor_final, Quantile_Recommendor_final
from ObjectiveFN import function_network_examples


case = 'robot'
algo_name = 'BONSAI_'


function_network, g, nominal_w = function_network_examples(case, algorithm_name= 'Recommender') 


recommendor = 'by_iteration_not'

# Recommendation is done every 5 iterations
T_val = [5*(i) for i in range(21)]

# This is just a naming convention that was followed while saving pickle file after the search process
with open(algo_name+case+'.pickle', 'rb') as handle:
    data = pickle.load(handle)

BONSAI_recommender_data = {}    


for i in data:
    print('##############################################')
    print('Run No', i)
    for T in T_val:
        print('T val', T)
        BONSAI_recommender_data[i,T] = Mean_Recommendor_final(data = data[i], g = g, T = T ) 
    
# with open(algo_name+recommender+case+ '_recommended.pickle', 'wb') as handle:
#     pickle.dump(BONSAI_recommender_data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
