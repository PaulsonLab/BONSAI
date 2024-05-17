#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:59:03 2024

@author: kudva.7
"""

import torch


class Test_simulation:
    def __init__(self):
        self.n_nodes = 2
        self.input_dim = 2

    def evaluate(self, X):
        input_shape = X.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))  
        
        # Pre-defined values      
        
        x = X[...,0]
        w = X[...,1]
        
        output[..., 0] = x**2+w-5
        output[..., 1] = x+w**2-1
        
        return output