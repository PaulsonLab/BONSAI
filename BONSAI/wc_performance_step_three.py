#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:15:12 2024
Plot worst case performance. Usually, the final recommended robust solution is usually
deployed in the real world. However, for the sake of analysis, we check how the recommended point every 
five iterations performs in terms of the worst case value.

@author: kudva.7
"""

import pickle 
import torch
import matplotlib.pyplot as plt
from ObjectiveFN import function_network_examples
import matplotlib.pyplot as plt
import time


case_study = ['robot']
case = case_study[0]

# To check the recommended value's performance
T_val = [5*(i) for i in range(21)]


data = {}

function_network, g, nominal_w = function_network_examples(case, algorithm_name= 'Recommender')

################# Main figures

with open('ARBO__GP_'+case+'_recommended.pickle', 'rb') as handle:
    data1 = pickle.load(handle)

with open('ARBO__UCB_'+case+'_recommended.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
    
with open('BOFN_'+case+'_recommended.pickle', 'rb') as handle:
    data3 = pickle.load(handle)
    
with open('BONSAI_'+case+'_recommended.pickle', 'rb') as handle:
    data4 = pickle.load(handle)
    
with open('Random_'+case+'_recommended.pickle', 'rb') as handle:
    data5 = pickle.load(handle)
    
with open('VBO_'+case+'_recommended.pickle', 'rb') as handle:
    data6 = pickle.load(handle)

data['ARBO_GP'] = data1
data['ARBO_UCB'] = data2
data['BOFN'] = data3
data['BONSAI'] = data4
data['Random'] = data5
data['VBO'] = data6


###############################################################################
########### Ablation studies


# with open('ARBO__GP_'+case+'_recommended.pickle', 'rb') as handle:
#     data1 = pickle.load(handle)
    
# with open('ARBO__BONSAI_'+case+'_recommended.pickle', 'rb') as handle:
#     data2 = pickle.load(handle)
    
# with open('ARBO__UCB_'+case+'_recommended.pickle', 'rb') as handle:
#     data3 = pickle.load(handle)

# with open('BONSAI_'+case+'_recommended.pickle', 'rb') as handle:
#     data4 = pickle.load(handle)
    
# with open('BONSAI__GP_'+case+'_recommended.pickle', 'rb') as handle:
#     data5 = pickle.load(handle)  
    
# with open('BONSAI__UCB_'+case+'_recommended.pickle', 'rb') as handle:
#     data6 = pickle.load(handle) 
    
# with open('Random_'+case+'_recommended.pickle', 'rb') as handle:
#     data7 = pickle.load(handle)
    
# with open('Random__GP_'+case+'_recommended.pickle', 'rb') as handle:
#     data8 = pickle.load(handle)
    
# with open('Random__UCB_'+case+'_recommended.pickle', 'rb') as handle:
#     data9 = pickle.load(handle)



# data['ARBO_GP'] = data1
# data['ARBO_BONSAI'] = data2
# data['ARBO_UCB'] = data3


# data['BONSAI'] = data4
# data['BONSAI_GP'] = data5
# data['BONSAI_UCB'] = data6

# data['Random'] = data7
# data['Random_GP'] = data8
# data['Random_UCB'] = data9


#####################################

# T_val = [25*(i) for i in range(5)]

# with open('BONSAI__Quantile_'+case+'_recommended.pickle', 'rb') as handle:
#     data1 = pickle.load(handle)
    
# with open('BONSAI__Mean_'+case+'_recommended.pickle', 'rb') as handle:
#     data2 = pickle.load(handle)



# data['BONSAI-Quantile'] = data1
# data['BONSAI-Mean'] = data2

####################################

val = {}


for algo in data:
    val[algo] = torch.zeros(30,len(T_val)) # Here 20 represents the steps for iteration 5, 10,... 100
    print(algo)   

    if g.nw != 0:   
        maxmin_val = torch.empty(g.w_combinations.size()[0],1)
        all_vals = 0 
        for i in range(30):
            print('Repeat Number', i)
            j = 0
            for t in T_val:
                t1 = time.time()
                print('Iteration value', t)
                X_empty = torch.empty(g.w_combinations.size()[0], g.nx)
                Z_new = data[algo][i,t]['Z'].repeat(g.w_combinations.size()[0],1)
                
                X_empty[..., g.design_input_indices] = Z_new
                X_empty[..., g.uncertain_input_indices] = g.w_combinations
                
                if case == 'classifier':
                    all_vals = g.objective_function(function_network(X_empty, test_mode = True))                    
                else:                    
                    all_vals = g.objective_function(function_network(X_empty))
                
                worst_case = all_vals.min()
                val[algo][i,j] = worst_case
                j += 1
                t2 = time.time()
                
                print('worst-case value =', worst_case)
                print('time for this evaluation',t2 - t1)
    
    else:
       for i in range(30):
           j = 0
           for t in T_val:           
               val[algo][i,j] = data[algo][i,t]['Y']
               j += 1   
        
    
    
    
    

# Get the maximum + minimum vals
max_val = []
min_val = []
for algo in data:
    max_val.append(val[algo].max())
    min_val.append(val[algo].min())

max_val = max(max_val)
min_val = min(min_val)

# Plot the robust regret:

color = ['blue', 'green','red','black', 'orange','purple','magenta','yellow', 'cyan']
j = 0

for algo in data:    

    #val1 = (val[algo] - min_val)/(max_val - min_val)
    val1 = val[algo]
    means = [torch.mean(element, dim = 0) for element in val1.T]
    std = [torch.std(element, dim = 0) for element in val1.T]
    
    plt.plot(T_val,means, label = algo, color = color[j])
    plt.fill_between(T_val,torch.tensor(means) - 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), torch.tensor(means) + 1.96*torch.tensor(std)/torch.sqrt(torch.tensor(30)), alpha=0.2, color= color[j])

    j += 1    
        
plt.xlabel('Iteration,t')
plt.xticks(T_val)
plt.ylabel('Worst case values')
plt.title('Performance for ' + case + ' case study')
plt.xlim([0,100])
plt.legend(labelspacing = 0.1, fontsize = 10)
plt.grid()
plt.show()    


with open(case+'_plot_final.pickle', 'wb') as handle:
    pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)