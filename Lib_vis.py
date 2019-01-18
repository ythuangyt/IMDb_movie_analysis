# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:50:18 2019

@author: Gentle Deng
"""
from scipy import sparse

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter3d_signal(signal, evecs_norm, zlimit = [0,7], size = [10,6], colormap = 'bwr', name = 'node scatter',
                     xlabel = 'x', ylabel = 'y', zlabel = 'z', title = 'title'):
    '''Scatter plot signal in 3-D.
    '''
    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')

    x = evecs_norm[:, 1]
    y = evecs_norm[:, 2]

    ax.scatter(x, y, signal, zdir='z', c = signal , cmap = colormap , label = name)

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim( np.min(x)*(1 - 0.1*np.sign(np.min(x))), np.max(x)*(1 + 0.1*np.sign(np.min(x))) )
    ax.set_ylim( np.min(y)*(1 - 0.1*np.sign(np.min(x))), np.max(y)*(1 + 0.1*np.sign(np.min(x))) )
    ax.set_zlim(zlimit)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=15., azim=-0)
    plt.show()
    plt.savefig('3Dsignal.png', dpi = 300)
    return 0


def graph_visual(weight, evecs_norm, signal, labels = None):
    '''
        Visualize graphs.
    '''
    # create a empty plot
    plt.figure()
    
    graph = nx.from_scipy_sparse_matrix(sparse.csr_matrix(weight))
    coords = evecs_norm[:, 1:3]  # Laplacian eigenmaps.
    nx.draw_networkx_nodes(graph, coords, node_size=60, node_color=labels.squeeze())
    nx.draw_networkx_edges(graph, coords, alpha=0.3)
    gain = np.append(np.zeros([signal.shape[0],1]), signal.reshape(signal.shape[0],1), axis=1)
    x = coords + gain
    
    for i in range(len(x)):
        x_cord = [x[i][0], coords[i][0]]
        y_cord = [x[i][1], coords[i][1]]
        plt.plot(x_cord, y_cord, 'b', linewidth = 0.1,linestyle='-')
    plt.savefig('2Dsignal.png', dpi = 300)
    return 0


def calculate_smoothness(f, L):
    """Calculate graph smoothness
        Input: 
           f    :  signal on the graph
           L    :  unnormalized Laplacian matrix of the graph
       
        Return  : float, the smoothness
        
        Formula:
           s = f' * L * f
    """
    return f.T.dot(L).dot(f)


def visual_list(loss = None, t = None, Vk = None, Vk_name = None, save = False):
    if t is not None:
        plt.figure()
        plt.plot(t)
        plt.ylabel('time parameter of heat kernel')
        plt.xlabel('number of iterations')
        if save:
            plt.savefig('time_constant.png', dpi = 300)
        
    if loss is not None:
        plt.figure()
        plt.plot(loss)
        plt.ylabel('loss')
        plt.xlabel('number of iterations')
        if save:
            plt.savefig('loss.png', dpi = 300)
    
    if Vk is not None:
        plt.figure()
        plt.plot(Vk)
        plt.ylabel('weight of subgraphs')
        plt.xlabel('number of iterations')
        if Vk_name is  None:
            print('[Warning: legends of Vk are not given]')
        else:
            plt.legend(Vk_name)
            
        if save:
            plt.savefig('weight_vector.png', dpi = 300)
    return 0


def visual_signalongraph(weight, signal, labels, mode = None, save = False):    
    # Compute laplacian and normalized laplacian
    weight_csr = sparse.csr_matrix(weight)
    degree_csr = sparse.csr_matrix(np.diag( np.sum(weight, axis = 1) ))
    
    laplacian_combinatorial = degree_csr - weight_csr
    laplacian_normalized    = degree_csr.power(-0.5).dot(laplacian_combinatorial).dot(degree_csr.power(-0.5))
    
    laplacian_combinatorial = laplacian_combinatorial.toarray()
    laplacian_normalized    = laplacian_normalized.toarray()
        
    # Compute eigen vectors and eigen values
    evals_norm,   evecs_norm   = np.linalg.eigh(laplacian_normalized)
    evals_unnorm, evecs_unnorm = np.linalg.eigh(laplacian_combinatorial)
    
    # Visualize singal on graph
    if mode == '2D':
        # show signal with 2D space
        graph_visual(weight, evecs_norm, signal, labels)
    elif mode == '3D':
        # show signal with 3D space
        scatter3d_signal(signal, evecs_norm, 
                         colormap = 'winter', size=[16,12], zlimit = [0.9*np.min(signal), 1.1*np.max(signal)], 
                         name = 'nodes', zlabel = 'signal values', title = 'ROI signal distribution')
    else:
        plt.figure()
        plt.stem(evecs_norm[:,1], signal, markerfmt='.')
        plt.xlabel('the smallest eigen vector')
        plt.ylabel('signals')
        if save:
            plt.savefig('signal_on_graph.png', dpi = 300)    
            
#    print("The smoothness of the given graph is {:.2f}".format(calculate_smoothness(signal, laplacian_combinatorial)))
    return 0