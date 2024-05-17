#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:31:41 2024

@author: kudva.7
"""
import torch



class Spring:
    def __init__(self):
        self.n_nodes = 3
        self.input_dim = 3

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        # Pre-defined values
        mu = 0.1
        sigma1 = 0.1
        omega1 = 100
        
        sigma2 = X[...,0]
        T = X[...,1]
        beta =  X[...,2]

        
        output[..., 0] = torch.sqrt((1 - beta**2/T**2)**2 + 4*(sigma2*beta/T)**2)
        output[..., 1] = (beta**2 - 1)*(beta/T)**2 - (1 + mu)*beta**2 - 4*(sigma1*sigma2*beta*beta/T) + 1
        output[..., 2] = (sigma1*beta**3/T**2) + (sigma2*beta*beta*beta*(1 + mu) - sigma2*beta)/T - sigma1*beta
        
        return output