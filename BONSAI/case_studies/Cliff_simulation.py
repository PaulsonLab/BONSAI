#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:01:00 2024

@author: kudva.7
"""

import torch
from torch import Tensor
#import matplotlib.pyplot as plt

class Cliff:
    def __init__(self):
        self.n_nodes = 5
        self.input_dim = 10

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        X1 = X[...,0] + 0.5*torch.sin(X[...,5])
        X2 = X[...,1] + 0.5*torch.sin(X[...,6])
        X3 = X[...,2] + 0.5*torch.sin(X[...,7])
        X4 = X[...,3] + 0.5*torch.sin(X[...,8])
        X5 = X[...,4] + 0.5*torch.sin(X[...,9])
        
        
        
        output[..., 0] = -10/(1 + 0.3*torch.exp(6*X1)) - 0.2*X1**2
        output[..., 1] = -10/(1 + 0.3*torch.exp(6*X2)) - 0.2*X2**2
        output[..., 2] = -10/(1 + 0.3*torch.exp(6*X3)) - 0.2*X3**2
        output[..., 3] = -10/(1 + 0.3*torch.exp(6*X4)) - 0.2*X4**2
        output[..., 4] = -10/(1 + 0.3*torch.exp(6*X5)) - 0.2*X5**2
        
        return output
    
    
if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    # a = Cliff()
    # LB = torch.tensor([0.,0.,0.,0.,0.,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2,-torch.pi/2])
    # UB = torch.tensor([5.,5.,5.,5.,5.,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2,torch.pi/2])
    
    # rand_init = LB + (UB - LB)*torch.rand(20,10)
    
    # print(a.evaluate(rand_init))
    robust_soln = True
    visualize_robust = False
    
    torch.set_default_dtype(torch.float64)
    dropwave = Cliff()
    input_dim = dropwave.input_dim
    
    
        
       
    if robust_soln:
        
        def fun2(X):
            X1 = X
            return -10/(1 + 0.3*torch.exp(6*X1)) - 0.2*X1**2
        
        xaxis2 = torch.arange(0, 5., (5-0)/1000)
        
        val2 = fun2(xaxis2)
        
        val_best2 = val2.max()
        val_index2= val2.argmax()
        nom_best = xaxis2[val_index2]
        
        
        def fun(X):
            X1 = X[...,0] + 0.5*torch.sin(X[...,1])
            return -10/(1 + 0.3*torch.exp(6*X1)) - 0.2*X1**2
        
        LB = torch.tensor([0.0,-torch.pi/2])
        UB = torch.tensor([5.,torch.pi/2])
            
        N = 1000   
            
        xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/N)
        yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/N)
        
        # create a mesh from the axis
        x2, y2 = torch.meshgrid(xaxis, yaxis)
        
        # reshape x and y to match the input shape of fun
        xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
        
        fig, ax = plt.subplots(1, 1)
        results = fun(xy)
        
        
        
        
        results2 = results.reshape(x2.size())
        
        inner_min = results2.min(dim = 1)
        
        min_indices = inner_min.indices
        
        robust_index = torch.argmax(inner_min.values)
        
        robust_find = torch.max(inner_min.values)
        robust_pt = x2[robust_index][0]
        worst_case = y2[:,min_indices[robust_index]][0]
        
        ######## Get the best value ###########
        xy_best = xy[results.argmax()]
        
        
        
        levels = np.linspace(results.min(), results.max(), 100)  
        contour_plot = ax.contourf(x2,y2,results2, levels = levels, cmap = 'jet')
   
    
        plt.vlines(robust_pt, -1.57,1.57, colors = 'white', linestyles = 'dashdot')
        plt.vlines(nom_best, -1.57,1.57, colors = 'magenta', linestyles = 'solid')
        
        plt.xlim(LB[0], UB[0])
        plt.ylim(LB[1], UB[1])
        
        fig.colorbar(contour_plot)
        
        
        plt.xlabel('$x_{i}$', fontsize = 20)
        plt.ylabel('$w_{i}$', fontsize = 20)
        
        
        #plt.savefig('cliff_robust.svg', format='svg', bbox_inches='tight')
        
        
        

    
     
   
    
    
    
    
    
    
        