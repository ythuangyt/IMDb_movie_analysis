# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:09:47 2019

@author: Gentle Deng
"""
import numpy as np
from Lib_graph import compute_weight_vector


def ROI_prediction(node, Vk, signal, features, k_nn, path_to_file):
    feature_vec = [None] * len(features)
    # 1. filter features
    for key, value in node.items():
        if len(value) == 0:
            continue
        else:
            # if the feature is in the given feature list, save its value to the right place
            try:
                ind = features.index(key)
                feature_vec[ind] = value
            # otherwise continue
            except:
                continue
    # 2. compute subgraphs vectors
    feature_weight = compute_weight_vector( features, feature_vec, path_to_file)
            
    for k in range(len(Vk)):
        weight = feature_weight[:,:,k] * Vk[k]
    ROI_neighbor = (-weight).argsort()[:k_nn]
    
    return (sum(ROI_neighbor)/k_nn)
    