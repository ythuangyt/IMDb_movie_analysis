# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:03:28 2019

@author: Gentle Deng
"""
import sys
import numpy as np
import scipy

def compute_dLkl_dWmn(k, l, W, sub_W, N_node, layer):
    """Function for computing partial L_kl partial Wmn and sum them up
    inputs: 
        k        :  row index of Lkl
        l        :  column index of Lkl
        W        :  aggregated weight matrix
        sub_W    :  3D matrix stacking all subgraphs in the 2nd demension 
        N_node   :  number of nodes in the graph
        layer    :  the index of Vk we are computing      
    outputs:
        the sum of (dLkl_dWmn * dWmn_dVk)
        
    Formula:
        if (m, n) = (k, l):
            dLkl_dWmn[m,n] = 0.5*Dkk^(-1.5)* Wkl * Dll^(-0.5)
                            - Dkk^(-0.5)*Dll^(-0.5)
                            + 0.5*Dkk^(-0.5) * Wkl * Dll^(-1.5)
        if m = k and n!= l:
            dLkl_dWmn[k,:] = 0.5*Dkk^(-1.5) * Wkl * Dll^(-0.5)
            
        if m!= k and n = l:
            dLkl_dWmn[:,l] = 0.5*Dkk^(-0.5) * Wkl * Dll^(-1.5)
    """
    
    # degree matrix of W
    D = np.diag( np.sum( W, axis = 0) )
    # initialize dLkl_dWmn
    dLkl_dWmn = np.zeros([N_node, N_node])
    # compute kth row of dLkl_dWmn
    # formula: dLkl_dWmn[k,:] = 0.5*Dkk^(-1.5) * Wkl * Dll^(-0.5)
    dLkl_dWmn[k,:]  = np.multiply( sub_W[k, :, layer], np.ones(N_node) * ( 0.5 * D[k,k]**(-1.5) * W[k,l] * D[l,l]**(-0.5) ) )
    # compute lth column of dLkl_dWmn
    # formula: dLkl_dWmn[:,l] = 0.5*Dkk^(-0.5) * Wkl * Dll^(-1.5)
    dLkl_dWmn[:,l]  = np.multiply( sub_W[:, l, layer], np.ones(N_node) * ( 0.5 * D[k,k]**(-0.5) * W[k,l] * D[l,l]**(-1.5) ) )
    # compute kth row and lth column
    # formula: dLkl_dWmn[:,l] = 0.5*Dkk^(-1.5)* Wkl * Dll^(-0.5)
    #                          - Dkk^(-0.5)*Dll^(-0.5)
    #                          + 0.5*Dkk^(-0.5) * Wkl * Dll^(-1.5)
    dLkl_dWmn[k,l]  = (0.5 * D[k,k]**(-1.5) * W[k,l] * D[l,l]**(-0.5)     \
                       - D[k,k]**(-0.5) * D[l,l]*(-0.5)                   \
                       + 0.5 * D[k,k] ** (-0.5) * W[k,l] * D[l,l]**(-1.5) ) * sub_W[k,l,layer]
    return np.sum(dLkl_dWmn)
            

def compute_dHij_dLkl(i, j, t, W, sub_W, L_norm, N_node, layer):
    """Function for computing partial H_ij partial Lkl and sum them up
    inputs: 
        i        :  row index of Lkl
        j        :  column index of Lkl
        t        :  time parameter for heat kernel
        W        :  aggregated weight matrix
        sub_W    :  3D matrix stacking all subgraphs in the 2nd demension
        L_norm   :  Normalized laplacian marix of the graph
        N_node   :  number of nodes in the graph
        layer    :  the index of Vk we are computing      
    outputs:
        the sum of (dHij_dLkl * dLkl_dWmn)
        
    Formula:
        if (k, l) = (i, j):
            dHij_dLkl[k,l] = -t + 0.5*t^2*(Lii + Ljj)
        if k = i and l!= j:
            dHij_dLkl[k,l] = 0.5*t^2*Llj    
        if k!= i and l = j:
            dHij_dLkl[k,l] = 0.5*t^2*Lik
    """
    
    # initialize dHij_dLkl
    dHij_dLkl = np.zeros([N_node, N_node])
    
    # compute ith row of dHij_dLkl
    # formula: dHij_dLkl[k,l] = 0.5*t^2*Llj    
    for ct in range(N_node):
        dHij_dLkl[i,ct] = 0.5*t*t*L_norm[ct,j] * compute_dLkl_dWmn(i, ct, W, sub_W, N_node, layer)

    # compute jth column of dHij_dLkl
    # formula: dHij_dLkl[k,l] = 0.5*t^2*Lik     
    for ct in range(N_node):
        dHij_dLkl[ct,j] = 0.5*t*t*L_norm[i,ct] * compute_dLkl_dWmn(ct, j, W, sub_W, N_node, layer)
        
    # compute ith row and jth column
    # formula: dHij_dLkl[:,l] = dHij_dLkl[k,l] = -t + 0.5*t^2*(Lii + Ljj)
    dHij_dLkl[i,j] = (-t + 0.5*t*t*( L_norm[i,i] + L_norm[j,j] )) * compute_dLkl_dWmn(i, j, W, sub_W, N_node, layer)
    return np.sum(dHij_dLkl)


def compute_dLoss_dvk(Ht, t, signal, error_vec, W_graph, L_norm, subgraphs, layer):
    """
    Function for computing partial Loss partial vk and sum them up
    inputs: 
        Ht         :  row index of Lkl
        t          :  time parameter for heat kernel
        signal     :  signal vector  
        error_vec  :  error vector
        W_graph    :  aggregated weight matrix
        L_norm     :  Normalized laplacian marix of the graph
        subgraphs  :  3D matrix stacking all subgraphs in the 2nd demensio
        layer      :  the index of Vk we are computing      
    outputs:
        the sum of (dLoss_Hij * dHij_dLkl)
        
    Formula:
        dLoss_dvk  = sum( dLoss_Hij[i,j]*dHij_dLkl ) for all (i,j)
        dLoss_Hij[i,j] = error_vect[i] * sigal[j] 
    """
    N_node = len(signal)
    dLoss_Hij = np.zeros([N_node, N_node])
    for i in range(N_node):
        for j in range(N_node):
            dLoss_Hij[i,j] = error_vec[i] * signal[j] * compute_dHij_dLkl(i, j, t, W_graph, subgraphs, L_norm, N_node, layer) 
    return np.sum(dLoss_Hij)
    

def compute_dHt_dt(W_graph, t):
    # degree to -0.5: D^(-0.5)
    d_full_power = np.diag( np.float_power( np.sum(W_graph, axis=0), -0.5) )
    
    # normalized Laplacian
    L_norm = np.eye(W_graph.shape[0], W_graph.shape[1]) - d_full_power.dot(W_graph).dot(d_full_power)
    
    # compute Gradient
    dHt_dt = -L_norm + t*L_norm.dot(L_norm)  # -L + t * L ^2
    return L_norm, dHt_dt


# to be analysis
def compute_Heatgradient(subgraphs, W_graph, V, t):
    dHt_dv = np.zeros( [W_graph.shape[0], W_graph.shape[1], len(V)] )
    
    # degree to -0.5: D^(-0.5)
    d_full_power = np.diag( np.float_power( np.sum(W_graph, axis=0), -0.5 ) )
    
    # normalized Laplacian
    L_norm = np.eye(W_graph.shape[0], W_graph.shape[1]) - d_full_power.dot(W_graph).dot(d_full_power)
    
    # compute Gradient
    dHt_dt = -L_norm + t*L_norm.dot(L_norm)  # -L + t * L ^2
    
    for ct in range(subgraphs.shape[2]):
        
        d_full_pow3  = np.diag( np.float_power( np.sum(W_graph, axis=1), -1.5) )
        d_subgraph_k = np.diag( np.sum(subgraphs[:,:,ct], axis=1) )
        
        dD_dvk = - 0.5 * d_full_pow3.dot(d_subgraph_k)
        
        # release memories
        del d_full_pow3, d_subgraph_k
        
        dL_dvk = - dD_dvk.dot(W_graph).dot(d_full_power) - d_full_power.dot(subgraphs[:,:,ct]).dot(d_full_power) - d_full_power.dot(W_graph).dot(dD_dvk)
        
        dHt_dv[:,:,ct] = - t*dL_dvk + t*t/2*( dL_dvk.dot(L_norm) + L_norm.dot(dL_dvk) )
    return L_norm, dHt_dt, dHt_dv


# to be analysis
def compute_Heatloss(signal, L_norm, t, dHt_dt, dHt_dv, gain = 1, sample_rate = 0.8):
    
    dLoss_dvk = np.zeros(dHt_dv.shape[2])
    
    # heat diffusion operator I - tL + 0.5*t^2 * L^2
#     Ht = np.eye(L_norm.shape[0], L_norm.shape[1]) - t* L_norm + t*t/2*L_norm*L_norm 
    Ht = scipy.linalg.expm(-t*L_norm)
    
    # pick useful signal points and shuffle them 
    pos_ind = np.where(signal != 0)[0]
    np.random.shuffle(pos_ind)
    
    # randomly choose top 'sample_rate' points as input samples
    sample = np.zeros(len(signal))     # create sample
    sample_ind = pos_ind[ : int( np.floor(sample_rate*len(pos_ind)) ) ]  # obtain top 'sample_rate' points' indices
    sample[sample_ind] = signal[sample_ind]
    
    test = np.zeros(len(signal))
    test_ind = pos_ind[ int( np.floor(sample_rate*len(pos_ind)) ) + 1 : ]
    test[test_ind] = signal[test_ind]

    # weight of loss function -- only compute loss from test points
    test_weight = np.zeros(len(signal))
    test_weight[test_ind] = 1
    
    # compute loss
    error_vec = np.multiply(test_weight, Ht.dot(sample) * gain - test )
    error_sca = np.linalg.norm( error_vec, 2 )**2/2/len(test_ind) 
    
    dLoss_dt = gain* error_vec.T.dot(dHt_dt).dot(signal) / len(test_ind)
    for ct in range(dHt_dv.shape[2]):
        dLoss_dvk[ct] = gain* error_vec.T.dot( dHt_dv[:,:,ct] ).dot(signal)     # regardless of the len(test), which is 1/N
    
    return error_sca, dLoss_dt, dLoss_dvk


def compute_Heatloss_elewise(signal, W_graph, L_norm, subgraphs, t, dHt_dt, gain = 1, sample_rate = 0.8):
    dLoss_dvk = np.zeros([subgraphs.shape[2], 1])
    
    # heat diffusion operator
    Ht = scipy.linalg.expm(-t*L_norm)
    
    # pick useful signal points and shuffle them 
    pos_ind = np.where(signal != 0)[0]
    # np.random.shuffle(pos_ind)            # to debug we use the same idxs 
    
    # randomly choose top 'sample_rate' points as input samples
    sample = np.zeros(len(signal))     # create sample
    sample_ind = pos_ind[ : int( np.floor(sample_rate*len(pos_ind)) ) ]  # obtain top 'sample_rate' points' indices
    sample[sample_ind] = signal[sample_ind]
    sample = np.expand_dims(sample, axis = 1)
    
    test   = np.zeros(len(signal))
    test_ind   = pos_ind[ int( np.floor(sample_rate*len(pos_ind)) ) : ]
    test[test_ind] = signal[test_ind]
    test = np.expand_dims(test, axis = 1)

    # weight of loss function -- only compute loss from test points
    test_weight = np.zeros(len(signal))
    test_weight[test_ind] = 1
    test_weight = np.expand_dims(test_weight, axis = 1)

    # compute loss
    error_vec = np.multiply( test_weight, Ht.dot(sample) * gain - test )
    error_sca = np.linalg.norm( error_vec ) / len(test_ind)
    
    dLoss_dt = np.sum( np.multiply( gain * error_vec.dot(sample.T), dHt_dt) ) / len(test_ind)
    for ct in range(subgraphs.shape[2]):
        dLoss_dvk[ct] = compute_dLoss_dvk( Ht, t, test, error_vec, W_graph, L_norm, subgraphs, ct ) / len(test_ind)
        
    return error_sca, dLoss_dt, dLoss_dvk

def validate_weight(weight):
    """Assess weight matrix
    
    If the degree vector of weight matrix contains an entry large/equal to 0, 
    then the weight matrix is not valid. Thus the weight must be modified. A
    simple solution is to set all non-diagnal zero entries in the invalid row
    to a small number such as 1e-5.
    """
    degree_vec = np.sum(weight, axis = 1)
    
    # find invalid rows
    rows = np.where( degree_vec == 0 )[0]
    
    # if the matix is not valid
    if len(rows) != 0:
        print('[Warning! The weight matrix is invalid and we reset it to the correct form]')
        # reset non-positive entries in each row to 1e-5
        for row in rows:
            weight[row, np.where(weight[row,:] <= 0)[0]] = 1e-5
        # make diagnal entries to be 0
        np.fill_diagonal(weight, 0)
    return weight

def Optimize_GraphWeight(W_3d, Vk, signal, t, heat_gain, maxiters = 100, algo_grad = 'matrix', 
                         algo_reg = 'L2', Lambda = 0.01, step_Vk = 0.1, step_t = 1e-8, ada_step = False):
    """Function to iteratively optimize the inter-graph weight

    Elementwise form and matrix form of computing gradient are supported. Both 
    L1 norm regularization and L2 regularization terms are supported. If the 
    adaptive stepsize flag is true, the step size for t and Vk will be generated 
    automatically.

    Args:
        W_3d     ,        3D matrix stacking all subgraphs in the 2nd demension 
        Vk       ,        The inter-graph weight vector which is optimized in this funtion
        signal   ,        Signal vector to be diffused
        t        ,        Time parameter of heat kernel
        heat_gain,        Manually set gain for siganl after diffusion
        maxiters ,        Max iterations, default is 100 
        algo_grad,        Method to compute gradient, defaut is 'matrix'          
        algo_reg ,        Regularization term, default is 'L2'
        Lambda   ,        Regularization parameter, default is 0.01
        step_Vk  ,        Stepsize for optimizing Vk, default is 0.1
        step_t   ,        Stepsize for optimizing t,default is 1e-8
        ada_step ,        Flag of adaptive stepsize

    Returns:
        list: list of t, loss and Vk.
    """
    
    print('\n********Optimization Phase for weights********')
    # list of parameters
    t_list          = [t]
    Vk_list         = [Vk]
    loss_list       = []
    
    # initialize aggregated graph
    Weight = np.zeros(W_3d.shape[0:2])
    for ct in range( len(Vk) ):
        Weight += W_3d[:,:,ct] * Vk[ct]
    Weight = validate_weight(Weight)
    
    
    # iteratively optimize weight vector
    for iters in range(maxiters):
                
        # Compute the main gradient -- L2 norm loss function
        if algo_grad == 'element_wise':
            # use ELEMENT-WISE method
            L_norm, dHt_dt = compute_dHt_dt( Weight, t )   
            loss,   dLoss_dt, dLoss_dvk = compute_Heatloss_elewise(signal, Weight, L_norm, W_3d, t, dHt_dt, gain = 1, sample_rate = 0.8)
        elif algo_grad == "matrix":
            # use MATRIX method
            L_norm, dHt_dt,   dHt_dv    = compute_Heatgradient(W_3d, Weight, Vk, t)   
            loss,   dLoss_dt, dLoss_dvk = compute_Heatloss(signal, L_norm, t, dHt_dt, dHt_dv, gain = heat_gain, sample_rate = 0.8)
        else:
            sys.exit(algo_grad, "is not supported now.")
        
        # Calculate regularization term
        if algo_reg == "L1":
            # use LASSO (i.e. L1 norm regularization) to update Vk and t
            regula_item = Lambda * np.ones(len(Vk))
            regula_item[ Vk < 0 ] = -Lambda
            # Calculate adaptive stepsize. -- different regularization term need defferent step
            # formula: step = 1/(t+1)^r, (0.5 < r < 1)
            if ada_step:
                step_t  = (5e-8)/(iters + 1)**0.75
                step_Vk = 0.05/(iters + 1)**0.75   
                
        elif algo_reg == 'L2':
            # use Redge Regression (i.e. L2 norm regularization) to update Vk and t
            regula_item = Lambda * Vk
            # Calculate adaptive stepsize. -- different regularization term need defferent step
            # formula: step = 1/(t+1)^r, (0.5 < r < 1)
            if ada_step:
                step_t  = (5e-8)/(iters + 1)**0.75
                step_Vk = 0.15/(iters + 1)**0.75   
                
        else:
            sys.exit(algo_reg, "is not supported now.")
        
        # Update Vk and t with regularization term 
        t  = abs(t  - step_t * dLoss_dt)
        Vk = abs(Vk - step_Vk * ( dLoss_dvk + regula_item )) 
        Vk = Vk / np.sum(Vk)
             
        t_list.append(t)
        Vk_list.append(Vk)
        loss_list.append(loss)
        
        # Update aggregated graph
        Weight = np.zeros(W_3d.shape[0:2])
        for ct in range( len(Vk) ):
            Weight += W_3d[:,:,ct] * Vk[ct]
        Weight = validate_weight(Weight)
        
        pct = iters/maxiters*100
        if np.mod(pct, 5) == 0:
            print(f'Now we have finished {pct:.2f}% of the task', end = "\n")
            
    return t_list, Vk_list, loss_list, Weight