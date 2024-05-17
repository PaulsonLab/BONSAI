#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:58:50 2024
This program contains all the baselines structured in the classic bayesian optimization manner:
1) Build a poterior GP using available data
2) Use the acquisition function to search over the z scpace
3) Evaluate the fuction at a point found by step 2) 
We continue steps 1-3 until budget is exhausted. In the paper, we have used 100 evaluations 
as the budget. It can be changed in the file "search_process_step_one.py"

We write the above steps repeatedly for each baseline test. Yes, it could have been more elegant and all the 
functions could fit in one. This was a design choice by the author for better readability during writing of the scripts


Note: In the paper we use z:= [x,w] where x is design variable and w is uncertain variable
but in the py scrits we have used x:= [z,w], a nomenclature change we decided after the results   

Please feel free to reach out to the authors of the papers in case you have questions!

@author: kudva.7
"""
from graph_utils import Graph
from typing import Callable
import torch
from torch import Tensor
from gp_network_utils import GaussianProcessNetwork
import time
from botorch.optim import optimize_acqf
from ARBO_acquisition import ARBO_UCB
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import copy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model 
import matplotlib.pyplot as plt
from utils import round_to_nearest_set
from single_level_algorithms import BOFN, BayesOpt
import sys
from TSacquisition_functions import ThompsonSampleFunctionNetwork, GPNetworkThompsonSampler, maxmin_ThompsonSampleFunctionNetwork
from botorch.models.model import Model


def remove_numbers_from_list_of_lists(numbers_to_remove, list_of_lists):
    return [list(filter(lambda x: x not in numbers_to_remove, sublist)) for sublist in list_of_lists]
    
def min_W_TS(model: Model,
             z_star: Tensor):
    
    """
    This is used to find the worst case value corresponding to a design z_star
    """
    
    X_new = torch.empty(model.dag.w_num_combinations, model.dag.nx)
    
    X_new[...,model.dag.design_input_indices] = z_star.repeat(model.dag.w_num_combinations, 1)
    X_new[...,model.dag.uncertain_input_indices] = model.dag.w_combinations  
    
    
    ts_network = GPNetworkThompsonSampler(model)
    ts_network.create_sample()
    
    Y_new = ts_network.query_sample(X_new)
    
    Y_new = model.dag.objective_function(Y_new)
   
    # Outer-max
    Y_maxmin = Y_new.min(0).values
    maxmin_idx = Y_new.min(0).indices
    
    w_final = model.dag.w_combinations[maxmin_idx]    
    
    return w_final.unsqueeze(0), Y_maxmin

###############################################################################
### BONSAI - Bayesian Optimization of Network Systems under uncertAInty #######
###############################################################################

def BONSAI(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           ) -> dict:
    """
    Parameters
    ----------
    x_init : Tensor
        Initial values of features
    y_init : Tensor
        Initial values of mappings
    g : Graph
        Function network graph structure
    objective : Callable
        Function network objective function
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2)  

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} TS(z,w)
            4) Time to compute w_star = min_{w} TS(z_star,w)
            5) Ninit: Number of initial values
            6) T: Evaluation budget

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    Ninit = x_init.size()[0]    
    
    
    # Instantiate requirements
    bounds =torch.tensor([[0]*(nz + nw),[1]*(nz + nw)])
    bounds = bounds.type(torch.float)
    

    time_opt1 = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for t in range(T):   
        
        print('Iteration number', t)        
        model = GaussianProcessNetwork(train_X=X, train_Y=Y, dag=g)
        
        if nw == 0:
            ts_fun = ThompsonSampleFunctionNetwork(model)
            
            t1 = time.time()
           
            z_star, acq_val = optimize_acqf(ts_fun, bounds, q = 1, num_restarts = 100, raw_samples = 1000)      
                  
            
           # Step 2) min TS(z_star, w))       
            w_star = 0   
            
            t2 = time.time()
            
            time_opt1.append(t2 - t1)
            
            X_new = z_star           
            
            
        else:
        
            # 1)  Step 1) max min TS(z, w)
            ts_fun = maxmin_ThompsonSampleFunctionNetwork(model)
            t1 = time.time()
           
            z_star, acq_val = optimize_acqf(ts_fun, bounds, q = 1, num_restarts = 100, raw_samples = 1000)      
            
            # Get the values that are important
            z_star = z_star[..., g.design_input_indices]
                  
            
           # Step 2) min TS(z_star, w))       
            w_star, acq_val = min_W_TS(model = model, z_star = z_star)       
            
            t2 = time.time()
            #print("Time taken for min_{w} min_{eta} LCB = ", t2 - t1) 
            
            time_opt1.append(t2 - t1)
            
            # Store the new point to sample
            X_new[..., design_input] = z_star
            X_new[..., uncertainty_input] = w_star
            
           
        print('Next point to sample', X_new) 
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
        
        print('Objective function here is')
        print(Y_new)
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'Ninit': Ninit, 'T': T}
    
    return output_dict

###### BONSAI Recommender ###########################

def Mean_Recommendor_final(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2),
           T= None
           ) -> dict:

    """
    Recommends design variables at the end of iteration T using GPFN-mean
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T != None:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)  
    
    if g.nw == 0:
        
        Z_test = data['X'][0: Ninit + T, g.design_input_indices]
        Y_test = data['Y'][0: Ninit + T]
        try:
            Y_out_val = g.objective_function(Y_test)
            Z_out = Z_test[Y_out_val.argmax()] 
        except:
            Y_out_val = g.objective_function(Y_test).values
            Z_out = Z_test[Y_out_val.argmax()]             
        
        Y_out = Y_out_val.max()
        
    else:
        
        Z_test = data['X'][0: Ninit + T, g.design_input_indices]
        Y_test = torch.zeros(Z_test.size()[0])
        model = GaussianProcessNetwork(train_X=X[: Ninit + T,:], train_Y=Y[: Ninit + T,:], dag=g)        
        
        # Mean recommender       
        for j in range(Z_test.size()[0]): 
            z_star = Z_test[j]
            z_star = z_star.repeat(g.w_num_combinations,1)        
            N_vals = torch.hstack((z_star, g.w_combinations))   
            X = torch.empty(N_vals.size())
            X[..., design_input] = z_star
            X[..., uncertainty_input] = g.w_combinations
            
            
            posterior = model.posterior(X)
            mean, _ = posterior.mean_sigma
            Y_test[j] = g.objective_function(mean).min().detach()
            
        Y_out = Y_test.max()
        Z_out = Z_test[Y_test[:Ninit + T].argmax()]
        
        print('BONSAI Recommendor suggested best point as', Z_out)
        print('Robust LCB at this recommendation', Y_out)
        
        # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict


def Quantile_Recommendor_final(data, # This is a dictionary
           g: Graph,
           beta = torch.tensor(2),
           T= None
           ) -> dict:

    """
    Recommends design variables at the end of iteration T using GPFN-Quantile
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T != None:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)  
    
    if g.nw == 0:
        
        Z_test = data['X'][0: Ninit + T, g.design_input_indices]
        Y_test = data['Y'][0: Ninit + T]
        try:
            Y_out_val = g.objective_function(Y_test)
            Z_out = Z_test[Y_out_val.argmax()] 
        except:
            Y_out_val = g.objective_function(Y_test).values
            Z_out = Z_test[Y_out_val.argmax()]             
            
        Y_out = Y_out_val.max()
        
    else:
        
        Z_test = data['X'][0: Ninit + T, g.design_input_indices]
        Y_test = torch.zeros(Z_test.size()[0])
        model = GaussianProcessNetwork(train_X=X[: Ninit + T,:], train_Y=Y[: Ninit + T,:], dag=g)        
        
        # Mean recommender       
        for j in range(Z_test.size()[0]): 
            z_star = Z_test[j]
            z_star = z_star.repeat(g.w_num_combinations,1)        
            N_vals = torch.hstack((z_star, g.w_combinations))   
            X = torch.empty(N_vals.size())
            X[..., design_input] = z_star
            X[..., uncertainty_input] = g.w_combinations
            
            #Sample the points
            posterior = model.posterior(X)
            
            N = 100
            
            N_comb = g.w_combinations.size()[0]
            
            lower = torch.empty(N_comb)
            Quant = torch.empty(N,N_comb)  
            sort_vals = torch.empty(N,N_comb)   
            
            for i in range(N):
                 Quant[i] = g.objective_function(posterior.rsample().detach())                
            
            for i in range(N_comb):
                sort_vals[:,i] = torch.sort(Quant[:,i]).values
                lower[i] = sort_vals[:,i][5]          
            
            Y_test[j] = lower.min()
            
        Y_out = Y_test.max()
        Z_out = Z_test[Y_test[:Ninit + T].argmax()]
        
        print('BONSAI Recommendor suggested best point as', Z_out)
        print('Robust LCB at this recommendation', Y_out)
        
        # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict

################################################################################
###################   Nominal mode  ##############################
##################################################################################
def BOFN_nominal_mode(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2),
           acq_fun = 'qEI',
           graph_structure = True,
           nominal_w = None,
           ) -> dict:
    
    """
    In this function, we set some nominal value for the uncertain variable and 
    search over nz dimensional space. Rest of the arguments is the same as the other 
    functions
    """
    
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    Ninit = x_init.size()[0]
    
    time_opt1 = []
    
    # Manipulate the graph to a nominal version of the problem
    
    if not nw == 0:
        g_new = copy.deepcopy(g)
        new_active_input_indices = remove_numbers_from_list_of_lists(g_new.uncertain_input_indices, g_new.active_input_indices)
        g_new.register_active_input_indices(new_active_input_indices)
        g_new.uncertain_input_indices = []
    else:
        g_new = g 
    
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for i in range(T):  
        print('Iteration number', i)
        
        t1 = time.time()
        
        try:
            if graph_structure:
                if nw == 0:
                    z_star = BOFN(X, Y, g_new, objective = None, T = 1, acq_type = 'EI', nominal_mode = True)
                else:
                    z_star = BOFN(X[...,design_input], Y, g_new, objective = None, T = 1, acq_type = 'EI', nominal_mode = True)
                    
            else:
                if nw == 0:
                    z_star = BayesOpt(X, Y, g_new, objective = None, T = 1, acq_type = acq_fun, nominal_mode = True)
                else:
                    z_star = BayesOpt(X[...,design_input], Y, g_new, objective = None, T = 1, acq_type = acq_fun, nominal_mode = True)
        except:
             z_star = X[torch.argmax(g_new.objective_function(Y)), design_input]
            
            
        t2 = time.time()
        time_opt1.append(t2 - t1)
        
        w_star = nominal_w
        
        # Store the new point to sample
        if nw == 0:
            X_new = z_star
        else:
        
            X_new[..., design_input] = z_star
            X_new[..., uncertainty_input] = w_star
        
        print('New point sampled is ', X_new)
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
            
        print('Objective value is ', g.objective_function(Y_new))
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])
    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'Ninit': Ninit, 'T': T}
    
    return output_dict

######### Recommender #########################################
def BOFN_Recommendor_final(data, # This is a pickle folder 
           g: Graph,
           beta = torch.tensor(2),
           T= None
           ) -> dict:

    """
    Recommends design variables at the end of each iteration
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
    bounds_w =torch.tensor([[0]*(nw + n_nodes),[1]*(nw + n_nodes)])
    bounds_w = bounds_w.type(torch.float)   
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T != None:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)
    
    if nw == 0:
        Z_test = data['X'][0: Ninit + T,:]
    else:
        Z_test = data['X'][0: Ninit + T, g.design_input_indices]
    
    Y_test = g.objective_function(data['Y'][0: Ninit + T])
    
    try:   
        Y_out = Y_test.max()
        Z_out = Z_test[Y_test.argmax()]
    except:
        Y_out = Y_test.values.max()
        Z_out = Z_test[Y_test.values.argmax()]
    
    
    print('BOFN Recommendor suggested best point as', Z_out)
    print('Robust LCB at this recommendation', Y_out)
    
    # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict

    

#################################################################################
##########################  ARBO ###############################
################################################################################

def ARBO(x_init: Tensor,
           y_init: Tensor, 
           g: Graph,
           objective: Callable,
           T: float,
           beta = torch.tensor(2)) -> dict:
    """
    Parameters
    ----------
    x_init : Tensor
        Initial values of features
    y_init : Tensor
        Initial values of mappings
    g : Graph
        Function network graph structure
    objective : Callable
        Function network objective function
    T : float
        Budget of function Network
    beta : Tensor, optional
        The confidence bound parameter. The default is torch.tensor(2).

    Returns
    -------
    output_dict: dict
        Constains the following data:
            1) All X values
            2) All Y values
            3) Time to compute z_star = max_{z} min_{w} max_{eta} UCB(z,w,eta)
            4) Time to compute w_star = min_{w} min_{eta} LCB(z_star,w,eta)

    """
    # Extract meta data from graph
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    
    if nw == 0:
        nw = g.w_combinations.size()[1]
        uncertainty_input = [i for i in range(nz,nz + nw)]
        design_input = [i for i in range(nz)]
    
    input_dim = nz + nw
    n_nodes = g.n_nodes
    input_index_info = [design_input,uncertainty_input] # Needed for ARBO acquisition function
    w_combinations = g.w_combinations
    Ninit = x_init.size()[0]

    
    # Instantiate requirements
    bounds_z =torch.tensor([[0]*(nz),[1]*(nz)])
    bounds_z = bounds_z.type(torch.float)
    
    bounds_w =torch.tensor([[0]*(nw),[1]*(nw)])
    bounds_w = bounds_w.type(torch.float)
    
    time_opt1 = []
    time_opt2 = []
    
    # Create a vector for collecting training data
    X = copy.deepcopy(x_init)
    Y = copy.deepcopy(y_init)
    
    X_new = torch.empty(input_dim)
    
    
    for t in range(T):     
        print('Iteration number', t) 
        #covar_module = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=input_dim)) 
        try:
            model = SingleTaskGP(X, g.objective_function(Y).unsqueeze(-1), outcome_transform=Standardize(m=1))
        except:
            model = SingleTaskGP(X, g.objective_function(X,Y).unsqueeze(-1), outcome_transform=Standardize(m=1))
        
        
        mlls = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mlls)
        
        model.eval()
  
        # Alternating bound acquisitions
        # 1) max_{z} min_{w} UCB(z,w)
        
        ucb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info, w_combinations = w_combinations)
        
        t1 = time.time()
        z_star, acq_value = optimize_acqf(ucb_fun, bounds_z, q = 1, num_restarts = 10, raw_samples = 100)
        t2 = time.time()
        #print("Time taken for max_{z} min_{w} UCB(z,w) = ", t2 - t1)
        time_opt1.append(t2 - t1)
        
        lcb_fun = ARBO_UCB( model, beta = beta, input_indices = input_index_info, maximize = False, fixed_variable= z_star, w_combinations= w_combinations)
        t1 = time.time()
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 100)
        
        # Seperate the discrete values
        if w_combinations is None:
            w_star = w_star
        else:
            w_star = round_to_nearest_set(w_star, g.w_sets)
            
        t2 = time.time()
        #print("Time taken for  min_{w} LCB(w; z_star) = ", t2 - t1)
        time_opt2.append(t2 - t1)        
        
        # Store the new point to sample
        X_new[..., design_input] = z_star
        X_new[..., uncertainty_input] = w_star
        
        print('Next point to sample', X_new)
        
        
        if input_dim == 2:
            Y_new = objective(X_new.unsqueeze(0))
        else:
            Y_new = objective(copy.deepcopy(X_new))
            
        print('Value of sample obtained', Y_new)
        
        # Append the new values
        X = torch.vstack([X,X_new])
        Y = torch.vstack([Y,Y_new])      
        
        #############################################################
        # # Test to see if optimizers are looking okay - Check
        
        # test_z = torch.arange(0, 1, (1)/100)
        # test_w = torch.arange(0, 1, (1)/100)
        # # create a mesh from the axis
        # x2, y2 = torch.meshgrid(test_z, test_w)

        # # reshape x and y to match the input shape of fun
        # xy = torch.stack([x2.flatten(), y2.flatten()], axis=1)
        
        
        # posterior = model.posterior(xy)
        # ucb = posterior.mean + posterior.variance.sqrt()
        
        # ucb = ucb.reshape(x2.size())
        
        # fig, ax = plt.subplots(1, 1)
        # plt.set_cmap("jet")
        # contour_plot = ax.contourf(x2,y2,ucb.detach().numpy())
        # fig.colorbar(contour_plot)
        # plt.xlabel('z')
        # plt.ylabel('w')
        
        #######################################################################    
    output_dict = {'X': X, 'Y': Y, 'T1': time_opt1, 'T2': time_opt2 ,'Ninit': Ninit, 'T': T}
    
    return output_dict

################ Recommender ###############################################

def ARBO_Recommendor_final(data, # This is a pickle file 
           g: Graph,
           beta = torch.tensor(0),
           T = None,
           ) -> dict:

    """
    Recommends design variables at the end of each iteration.
    Note: The beta value was set to 0 for the case of GP-Mean recommender
    whereas, it was set to 2 for GP-Quantile
    """
    uncertainty_input = g.uncertain_input_indices
    design_input = g.design_input_indices   
    nz = g.nz
    nw = g.nw
    input_dim = g.nx 
    n_nodes = g.n_nodes
    
    # Instantiate requirements
    
   
    if nw == 0:
        nw = g.w_combinations.size()[1]
        uncertainty_input = [i for i in range(nz,nz + nw)]
        design_input = [i for i in range(nz)]
       
    bounds_w =torch.tensor([[0]*(nw),[1]*(nw)])
    bounds_w = bounds_w.type(torch.float)   
     
    
    
    Ninit = data['Ninit']
    X = data['X']
    Y = data['Y']
    
    if T != None:
        print('Pre-defined T value')
    else:
        T = Y[Ninit:].size()[0]
    
    Y_out = torch.empty(1,1)
    Z_out = torch.empty(1, g.nz)
    
    try:
        model = SingleTaskGP(X[: Ninit + T,:], g.objective_function(Y[: Ninit + T,:]).unsqueeze(-1), outcome_transform=Standardize(m=1))
    except:
        model = SingleTaskGP(X[: Ninit + T,:], g.objective_function(X[: Ninit + T,:],Y[: Ninit + T,:]).unsqueeze(-1), outcome_transform=Standardize(m=1))

    
    Z_test = data['X'][0: Ninit + T, design_input]
    Y_test = torch.zeros(Z_test.size()[0])

    #         
    for j in range(Z_test.size()[0]): 
        z_star = Z_test[j].unsqueeze(0)      
        lcb_fun = ARBO_UCB( model, beta = beta, input_indices = [design_input, uncertainty_input], maximize = False, fixed_variable= z_star, w_combinations= g.w_combinations)
        w_star, acq_value = optimize_acqf(lcb_fun, bounds_w, q = 1, num_restarts = 10, raw_samples = 100)
        w_star = round_to_nearest_set(w_star, g.w_sets)
        
        Y_test[j] = -1*acq_value
        
    Y_out = Y_test.max()
    Z_out = Z_test[Y_test[:Ninit + T].argmax()]
    
    print('BONSAI Recommendor suggested best point as', Z_out)
    print('Robust LCB at this recommendation', Y_out)
    
    # Acquisition function values
    output_dict = {'Z':Z_out , 'Y': Y_out}
    
    return output_dict





















