# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:59:53 2019

@author: Gentle Deng
"""

import numpy as np

from Lib_graph import construct_graphs
from Lib_prediction import ROI_prediction
from Lib_gradientdecent import Optimize_GraphWeight
from Lib_vis import visual_list, visual_signalongraph

# In[ PART 1. Generate subgraphs ]

# 1.setting file path and choosing features
Genre     = 'Romance'
File_path = r'./data'
features  = ['budget',
             'cast',
             'crew',
             'genres',    # but genre is not used to build a subgraph up to now
             'keywords',
             'popularity',
             'production_companies',
             'vote_average']

# 2.constructing subgraphs
signal, W_3d = construct_graphs( File_path, features, genre = Genre)


# In[ PART 2. Optimize weight vector V ] 

# 1.initialize GD parameters
maxiters        = 200
step_Vk         = 0.2
step_t          = 0.00000005
Lambda          = 0.01     # L1 regularization lambda = 0.01
gain            = 5

algo_reg        = 'L2'
algo_grad       = 'matrix'
adaptive_step   = True

# 2.initialize kernel parameters
# It would be more convincing that weights of these subgraphs start from the same 
# start point (1/number of graphs). In addition, our objective funtion is nonconvex 
# to Vk. Thus if Vk is randomly initialized, it tends to converge to different local minima.
t               = 0.001
Vk              = 1/( len(features) - 1 ) * np.ones( len(features) - 1 )  # genre is not used to build a subgraph up to now

# 3.iteratively optimize Vk
t_list, Vk_list, loss_list, final_weight = Optimize_GraphWeight( W_3d,      Vk,       signal, t,       gain,   maxiters,
                                                                 algo_grad, algo_reg, Lambda, step_Vk, step_t, adaptive_step)
# 4.save weight vector list
np.save(File_path + '/Vk_list_' + Genre, Vk_list)

# In[ PART 3. Visualize the results ] 

# 1.visualze the process of optimization 
legend = features.copy()
legend.remove('genres')
visual_list( loss_list, t_list, Vk_list, Vk_name = legend, save = True)

# 2.visualize the signal on the final graph 
visual_signalongraph(final_weight, signal, labels = np.ones(final_weight.shape[0]), save = True)

# 3.visualize the signal on the original graph
Weight = np.zeros(W_3d.shape[0:2])
for ct in range( len(Vk) ):
    Weight += W_3d[:,:,ct] * Vk[ct]
visual_signalongraph(Weight, signal, labels = np.ones(Weight.shape[0]), save = True)


# In[ PART 4. Predict the ROI of a given new node ]

# 1. define a new node with its 
# Please fill None when there is no data in a feature
La_La_Land = { # La La Land, revenue: 446092357
             'budget':      30000000, 
             'cast':        ['Ryan Gosling', 'Emma Stone', 'Amiée Conn', 'Terry Walters', 'Thom Shelton'], 
             'crew':        'Damien Chazelle', 
             'genres':      None, 
             'keywords':    ['los angeles california', 'pianist', 'aspiring actress', 'musician', 'jazz club'], 
             'popularity':  284, 
             'production_companies':['Summit Entertainment', 'Black Label Media', 'TIK Films', 'Impostor Pictures', 'Gilbert Films', 'Marc Platt Productions'], 
             'vote_average':8.0}

Pacific_Rim = { # revenue: 407602906
            'budget':      180000000, 
             'cast':        ['Stacker Pentecost', 'Raleigh Becket'], 
             'crew':        'Guillermo del Toro', 
             'genres':      None, 
             'keywords':    ['dystopia', 'giant robot', 'giant monster', 'apocalypse', 'imax'], 
             'popularity':  56.523205, 
             'production_companies':['Legendary Pictures', 'Warner Bros.', 'Disney Double Dare You (DDY)', 'Indochina Productions'], 
             'vote_average':6.7
        }

# 2. predict ROI of the movie
# k-nearest neigbor to do prediction -- different genres need different k: Action = 10, Romance = 5, 
k_nn = 10

Predict_ROI = ROI_prediction(La_La_Land, signal, Vk, features, k_nn, File_path, genre=Genre)
print('The prediction on ROI of the given movie is:', Predict_ROI)
