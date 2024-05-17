#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:18:52 2024

@author: kudva.7
"""

import torch
from torch import Tensor
import matplotlib.pyplot as plt

class Pharm:
    def __init__(self):
        self.n_nodes = 4
        self.input_dim = 6

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        x1 = X[...,0] 
        x2 = X[...,1] 
        x3 = X[...,2]
        x4 = X[...,3]
        x5 = X[...,4]
        x6 = X[...,5]
        
        # Add uncertainty
        x3 += x5
        x4 += x6
        
        output[...,0] = x3
        output[...,1] = x4
        
        # Calculating each component of the function 1 (f1)
        term1 = -3.95 + 9.20 * (1 + torch.exp(-(0.32 + 5.06*x1 - 4.07*x2 - 0.36*x3 - 0.34*x4)))**-1
        term2 = 9.88 * (1 + torch.exp(-(-4.83 + 7.43*x1 + 3.46*x2 + 9.19*x3 + 16.58*x4)))**-1
        term3 = 10.84 * (1 + torch.exp(-(-7.90 - 7.91*x1 - 4.48*x2 - 4.08*x3 - 8.28*x4)))**-1
        term4 = 15.18 * (1 + torch.exp(-(9.41 - 7.99*x1 + 0.65*x2 + 3.14*x3 + 0.31*x4)))**-1

        output[...,2] = term1 + term2 + term3 + term4
        
        # Calculating each component of the function
        terma = 1.07 + 0.62 * (1 + torch.exp(-(3.05 + 0.03*x1 + 0.16*x2 + 4.03*x3 - 0.54*x4)))**-1
        termb = 0.65 * (1 + torch.exp(-(1.78 + 0.60*x1 - 3.19*x2 + 0.10*x3 + 0.54*x4)))**-1
        termc = -0.72 * (1 + torch.exp(-(0.01 + 2.04*x1 - 3.73*x2 + 0.10*x3 - 1.05*x4)))**-1
        termd = -0.45 * (1 + torch.exp(-(1.82 + 4.78*x1 + 0.48*x2 - 4.68*x3 - 1.65*x4)))**-1
        terme = -0.32 * (1 + torch.exp(-(2.69 + 5.99*x1 + 3.87*x2 + 3.10*x3 - 2.17*x4)))**-1
               
        
        output[...,3] = terma + termb + termc+ termd + terme
        
        return output

if __name__ == '__main__':
    """        Hello world!
    """
    sim = Pharm()
    x_test = torch.tensor([-1.,0.,0.,0.,1.,1.])
    
    a = sim.evaluate(x_test)