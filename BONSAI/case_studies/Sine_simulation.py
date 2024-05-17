#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:22:25 2024

@author: kudva.7
"""


import torch

class Sine:
    def __init__(self):
        self.n_nodes = 6
        self.input_dim = 4

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))         
        
        output[..., 0] = X[...,0] + X[...,2]
        output[..., 1] = -1*torch.sin(2*torch.pi*output[...,0]**2)
        output[..., 2] = -1*(output[...,0]**2 + 0.2*output[...,0])
        output[..., 3] =  X[...,1] + X[...,3]
        output[..., 4] = -1*torch.sin(2*torch.pi*output[...,3]**2)
        output[..., 5] = -1*(output[...,3]**2 + 0.2*output[...,3])
        
        return output
    
    
if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    
    method1 = True
    
    rosenbrock = Sine()
    input_dim = rosenbrock.input_dim
    fun  = lambda X: rosenbrock.evaluate(X)
    
    LB = torch.tensor([-1.,-1.,-0.25,-0.25])
    UB = torch.tensor([1.,1.,0.25,0.25])
    
    if method1:
        visualize_robust = True
        
        torch.set_default_dtype(torch.float64)
        dropwave = Sine()
        input_dim = dropwave.input_dim

        fun = lambda z: -torch.sin(2*torch.pi*z[:,0]**2) - z[:,0]**2 - 0.2*z[:,0] -torch.sin(2*torch.pi*z[:,1]**2) - z[:,1]**2 - 0.2*z[:,1]
        
        LB = torch.tensor([-1.,-1.])
        UB = torch.tensor([1.,1.])
            
        N = 1000   
            
        xaxis = torch.arange(LB[0], UB[0], (UB[0]-LB[0])/N)
        yaxis = torch.arange(LB[1], UB[1], (UB[1]-LB[1])/N)
        
        # create a mesh from the axis
        x2, y2 = torch.meshgrid(xaxis, yaxis)
        
        # reshape x and y to match the input shape of fun
        xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
        
        
        results = fun(xy)
        
        results2 = results.reshape(x2.size())
        
        levels = np.linspace(results.min(), results.max(), 100)
        
        #w_set = [[0,0.2,0.4,0.6,1.],[0.0000, 0.100, 0.1250, 0.2000, 0.2500, 0.300, 0.3750, 0.4500, 0.5000, 0.5700, 0.6250, 0.700, 0.7500, 0.8750, 0.9500, 1.0000]]
        w_set = [[0.,0.33,0.5,0.66,1.], [0.,0.33,0.5,0.66,1.] ]
        w_set_1 = [i*0.5 - 0.25 for i in w_set[0]]
        w_set_2 = [i*0.5 - 0.25 for i in w_set[1]]
        
        w_set = [w_set_1, w_set_2]
        
        all_combinations = itertools.product(*w_set)
        tensors = [torch.tensor(combination) for combination in all_combinations]
        
        # Stack the tensors to create the final result
        w_combinations = torch.stack(tensors)
        J = w_combinations.shape[0]
        
        worst_case_vals = torch.empty(results.size())
        
        if visualize_robust:
            for i in range(N**2):
                x_tilde = xy[i].repeat(J,1)
                X_vals = torch.hstack((x_tilde, w_combinations))
                wc_vals = dropwave.evaluate(X_vals)
                yy = wc_vals[...,1] + wc_vals[...,2] + wc_vals[...,4] + wc_vals[...,5]
                worst_case_vals[i] = yy.min()       
                
            results3 = worst_case_vals.reshape(x2.size())     
            xy_robust = xy[worst_case_vals.argmax()]    
                
        
        ############## Generate 
        
        
        ######## Get the best value ###########
        xy_best = xy[results.argmax()]
        
        
        ####
        
        vmin = -1000
        vmax = 10
        
        
        
        fig, ax = plt.subplots(1, 1)
        #plt.set_cmap("jet")
        
        if visualize_robust:
            levels = np.linspace(worst_case_vals.min(), worst_case_vals.max(), 100)
            contour_plot = ax.contourf(x2,y2,results3, levels = levels, cmap = "jet")
            plt.scatter(xy_robust[0], xy_robust[1], marker = '*', color = 'white', s = 300, label = 'Robust Solution')
        else:
            contour_plot = ax.contourf(x2,y2,results2, levels = levels, cmap = 'jet')
            
        fig.colorbar(contour_plot)
        #plt.vlines(robust_pt, LB[1], UB[1], colors = 'black', linestyles = 'dashed')
        
        #plt.scatter(xy_best[0], xy_best[1], marker = '*', color = 'k', s = 300, label = 'Nominal Solution')
        
        
        plt.xlim(LB[0], UB[0])
        plt.ylim(LB[1], UB[1])
        
        
        plt.xlabel('$x_{1}$', fontsize = 20)
        plt.ylabel('$x_{2}$', fontsize = 20)

        #plt.savefig('sine_robust.svg', format='svg', bbox_inches='tight')
    
    
    
    
    else:
        Nz = 1000
        Nw = 1000
    
        
        
        
        # For plot and robust solution
        
        x_lb = LB[...,[0,1]]
        x_ub = UB[...,[0,1]]
        
        w_lb = LB[...,[2,3]]
        w_ub = UB[...,[2,3]]
        
        
        
        rand_Z = torch.rand(Nz,2)*(x_ub - x_lb) + x_lb
        ################################
        # w_set = [[0.,0.33,0.5,0.66,1.], [0.,0.33,0.5,0.66,1.] ]
        # w_set_1 = [float(i*(w_ub[0] - w_lb[0]) + w_lb[0])  for i in w_set[0]]
        # w_set_2 = [float(i*(w_ub[1] - w_lb[1]) + w_lb[1]) for i in w_set[1]]
        
        # w_set = [w_set_1, w_set_2]
        
        # all_combinations = itertools.product(*w_set)
        # rand_W = [torch.tensor(combination) for combination in all_combinations]
        # rand_W = torch.stack(rand_W)
        # Nw = rand_W.size(0)
        
        
        ################################
        
        rand_W = torch.rand(Nw,2)*(w_ub - w_lb) + w_lb
        rand_Z_big = rand_Z.repeat_interleave(Nw,0)
        rand_W_big = rand_W.repeat(Nz,1)
        
        rand_X = torch.hstack([rand_Z_big, rand_W_big])
        
        fun_val = torch.empty(Nz*Nw,1)
        
        min_fun_val = torch.empty(Nz,1)
        design_vals = torch.empty(Nz,4)
        
        fun_val = fun(rand_X)
        fun_val = (fun_val[...,1] + fun_val[...,2] + fun_val[...,4] + fun_val[...,5]).unsqueeze(1)
        
        
        for i in range(Nz):
            
            min_fun_val[i] = fun_val[i*Nw:(i+1)*Nw].min()
            design_vals[i] = rand_X[i*Nw:(i+1)*Nw][fun_val[i*Nw:(i+1)*Nw].argmin()]
                
        
        best_value = min_fun_val.max()
        max_min_soln = design_vals[min_fun_val.argmax()]
        
        
        
        