#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:45:30 2024

@author: kudva.7
"""
import torch

class Rosenbrock:
    def __init__(self):
        self.n_nodes = 3
        self.input_dim = 3

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        # 
        Z = X[...,0] + X[...,2]
        W = X[...,1]  
        
        # Network structure
        output[..., 0] = Z**2
        output[..., 1] = (Z - 1)**2
        output[..., 2] = (W - output[..., 0])**2
        
        return output

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    
    rosenbrock = Rosenbrock()
    input_dim = rosenbrock.input_dim
    fun  = lambda X: rosenbrock.evaluate(X)
    
    LB = torch.tensor([-1, 0, -0.1])
    UB = torch.tensor([2, 2, 0.1])
    
    Nz = 1000

    
    
    
    # For plot and robust solution
    
    x_lb = LB[...,0]
    x_ub = UB[...,0]
    
    w_lb = LB[...,[1,2]]
    w_ub = UB[...,[1,2]]
    
    
    
    rand_Z = torch.arange(x_lb, x_ub, (UB[0]-LB[0])/Nz)
    rand_Z = rand_Z.unsqueeze(1)
    
    #rand_W = torch.rand(Nw,2)*(w_lb - w_ub) + w_lb
    
    
    w_set = [list(torch.linspace(0,1,20).detach().numpy()), list(torch.linspace(0,1,3).detach().numpy()) ]
    w_set_1 = [float(i*(w_ub[0] - w_lb[0]) + w_lb[0])  for i in w_set[0]]
    w_set_2 = [float(i*(w_ub[1] - w_lb[1]) + w_lb[1]) for i in w_set[1]]
    
    w_set = [w_set_1, w_set_2]
    
    all_combinations = itertools.product(*w_set)
    rand_W = [torch.tensor(combination) for combination in all_combinations]
    rand_W = torch.stack(rand_W)
    Nw = rand_W.size(0)
    
    
    rand_Z_big = rand_Z.repeat_interleave(Nw,0)
    rand_W_big = rand_W.repeat(Nz,1)
    
    rand_X = torch.hstack([rand_Z_big, rand_W_big])
    
    fun_val = torch.empty(Nz*Nw,1)
    
    min_fun_val = torch.empty(Nz,1)
    design_vals = torch.empty(Nz,3)
    
    fun_val = fun(rand_X)
    fun_val = (-100*fun_val[...,2] - 1*fun_val[...,1]).unsqueeze(1)
    
    
    for i in range(Nz):
        
        min_fun_val[i] = fun_val[i*Nw:(i+1)*Nw].min()
        design_vals[i] = rand_X[i*Nw:(i+1)*Nw][fun_val[i*Nw:(i+1)*Nw].argmin()]
            
    
    best_value = min_fun_val.max()
    max_min_soln = design_vals[min_fun_val.argmax()]
    
    
    
    plt.plot(rand_Z.squeeze(1).detach().numpy(), min_fun_val.squeeze(1).detach().numpy(), c = 'k', label = 'G')
    plt.grid()
    
    plt.ylim([-2000,-140])
    plt.vlines(max_min_soln[0], -2000,-140, colors = 'red', linestyles = 'dashdot', label = '$x^{\star}_{\mathcal{W}}$' )
    
    ##############################
    
    N = rand_Z.size(0)
    w_nom = torch.tensor([[1.4,0]])
    w_nom = w_nom.repeat(N,1)
    
    new_Z = torch.hstack((rand_Z,w_nom))
    
    fun_val2 = fun(new_Z)
    fun_val2 = (-100*fun_val2[...,2] - 1*fun_val2[...,1]).unsqueeze(1)
    
    nom_index = fun_val2.argmax()
    best_val = rand_Z[nom_index]
    
    plt.vlines(best_val, -2000,-140, colors = 'magenta', linestyles = 'solid', label = 'Nominal solution' )
    
    
    
    #plt.ylim([-1750.,100.])
    plt.xlim([-1,2])
    plt.ylabel('G(x)', fontsize = 30)
    plt.xlabel('x', fontsize = 30)
    plt.legend(fontsize = 15, loc = 'lower left')
    
    #plt.savefig('rosenbrock_projection.pdf', format='pdf',bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    