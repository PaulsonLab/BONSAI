#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:42:06 2024

@author: kudva.7
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:38:41 2024

@author: kudva.7
"""

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch.quasirandom import SobolEngine
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.normal import IIDNormalSampler
# import botorch.sampling.samplers import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from typing import Optional
import torch
import sys
import copy
from botorch.utils.safe_math import smooth_amax, smooth_amin

from gp_network_utils import MultivariateNormalNetwork

torch.set_default_dtype(torch.double)

############### Helper functions ##############################

def active_corners(theta_min,theta_max):
    """
    This code is mainly used to generate all corners of box constraints.
    Will be incorporated in the Object that will give us the bounds.

    inputs:
    theta_min -- N dimensional tensor
    theta_max -- N dimensional tensor

    output -- 2^(N) X N dimensional tensor
    """
    size_t1 = torch.Tensor.size(theta_min)
    size_t2 = torch.Tensor.size(theta_max)

    # Show error if dimensions dont match:
    if size_t1 != size_t2:
        sys.exit('The dimensions of bounds dont match: Please enter valid inputs')

    val = size_t1[0]
    size_out = 2**(val)
    output = torch.zeros(size_out,val)
    output_iter = torch.zeros(size_out)

    for i in range(val):
        div_size = int(size_out/(2**(i+1)))
        divs = int(size_out/div_size)
        div_count = 0
        for j in range(divs):
            if bool(j%2):
                output_iter[div_count:div_count+div_size] = theta_min[i]*torch.ones(div_size)
            else:
                output_iter[div_count:div_count+div_size] = theta_max[i]*torch.ones(div_size)
            div_count = div_count + div_size
        output[:,i] = output_iter
    return output

def get_nodes_coeffs(n_nodes = None): # 

    """
        Generates extreme eta values for  high dimensional cases
    """
    
    bound1 = torch.ones(n_nodes)
    bound2 = -1*torch.ones(n_nodes)
    
    return active_corners( bound1, bound2)


################################################################################
################ Adversarily robust bayesian optimization ######################
################################################################################
 
class ARBO_UCB(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        input_indices: list,
        maximize: bool = True,        
        fixed_variable = None,
        w_combinations = None
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.fixed_variable = fixed_variable
        self.design_input_indices = input_indices[0]
        self.uncertain_input_indices = input_indices[1]
        self.w_combinations = w_combinations

        # self.n = 1

    @t_batch_mode_transform(expected_q=1)
    def forward(self, Xi: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        # Remove unnecessary dimensions just like analytical acq function
        # self.n += 1
        # print(self.n)
        beta = self.beta.to(Xi)
        nz = len(self.design_input_indices)
        nw = len(self.uncertain_input_indices)
        
        if self.maximize:
            Nz = Xi.size()[0]
            #torch.manual_seed(10000)
            #nz
            
            if self.w_combinations is None:
                Nw = 100
            else:
                Nw = self.w_combinations.size()[0]
            X = torch.empty(Nz*Nw, nz + nw)
            X[..., self.design_input_indices] = Xi.squeeze(-2).repeat_interleave(Nw, dim=0)         
            
            
            if self.w_combinations is None:
                soboleng_w = SobolEngine(dimension= nw, seed = 10000)
                X[..., self.uncertain_input_indices] = soboleng_w.draw(Nw, dtype = torch.double).repeat(Nz,1)   
            else:
                X[..., self.uncertain_input_indices] = self.w_combinations.repeat(Nz,1)
            #X[..., self.uncertain_input_indices] = torch.rand(Nw, nw).repeat(Nz,1)           
            
            posterior = self.model.posterior(X)            
            mean = posterior.mean
            std = posterior.variance.sqrt()
            # Upper confidence bounds
            ucb = mean + beta*std
            
            ucb_mesh = torch.empty(Nz,Nw)
            
            for i in range(Nz):
                ucb_mesh[i] = ucb[i*Nw:(i+1)*Nw].reshape(Nw)
            
            objective = smooth_amin(ucb_mesh, dim = -1)
            
        else:
            if self.fixed_variable is None:
                print('Fixed variable needs to be specified if maximize == "False"')
                sys.exit()
            Nw = Xi.size()[0]
            X = torch.empty(Nw, nz + nw)
            X[..., self.design_input_indices] = self.fixed_variable.repeat_interleave(Nw, dim=0)
            X[..., self.uncertain_input_indices] = Xi.squeeze(-2)
            
            posterior = self.model.posterior(X)            
            mean = posterior.mean
            std = posterior.variance.sqrt()
            
            ucb =  -1*mean + beta*std          
            
            objective = ucb.squeeze(-2).squeeze(-1)          

        return objective
 