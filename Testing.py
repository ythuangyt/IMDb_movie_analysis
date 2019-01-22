# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:41:55 2019

@author: Gentle Deng
"""
from Lib_graph import construct_ROIsignal
from Lib_prediction import ROI_prediction

Genre     = 'Science Fiction'
File_path = r'./data'

signal = construct_ROIsignal(IMDb = 0, path = File_path)

features  = ['budget',
             'cast',
             'crew',
             'genres',    # but genre is not used to build a subgraph up to now
             'keywords',
             'popularity',
             'production_companies',
             'vote_average']

La_La_Land = { # La La Land, revenue: 446092357
             'budget':      30000000, 
             'cast':        ['Ryan Gosling', 'Emma Stone', 'Amiée Conn', 'Terry Walters', 'Thom Shelton'], 
             'crew':        'Damien Chazelle', 
             'genres':      None, 
             'keywords':    ['los angeles california', 'pianist', 'aspiring actress', 'musician', 'jazz club'], 
             'popularity':  284, 
             'production_companies':['Summit Entertainment', 'Black Label Media', 'TIK Films', 'Impostor Pictures', 'Gilbert Films', 'Marc Platt Productions'], 
             'vote_average':8.0}

Pacific_Rim2 = { # revenue: 290061297
            'budget':      150000000 , 
             'cast':        ['Jake Pentecost', 'Nate Lambert', 'Amara Namani', 'Hermann Gottlieb'], 
             'crew':        'Steven S. DeKnight', 
             'genres':      None, 
             'keywords':    ['sea doo', 'woman with masculine short hair', 'woman with masculine hair', 'chinawoman', 'teen girl with too much make up'], 
             'popularity':  30 , 
             'production_companies':['Clear Angle Studios', 'Dentsu', 'Double Dare You (DDY)', 'Double Negative (DNEG)', 'Fuji Television Network ', 'Legendary Entertainment'], 
             'vote_average':5.6 }


Interstellar = { # revenue: 675120017
             'budget':      165000000 , 
             'cast':        ['Matthew McConaughey', 'Ellen Burstyn', 'Mackenzie Foy', 'John Lithgow'], 
             'crew':        'Christopher Nolan', 
             'genres':      'Science Fiction', 
             'keywords':    ['saving the world', 'artificial intelligence', 'father son relationship', 'single parent', 'nasa'], 
             'popularity':  724.247784 , 
             'production_companies':['Paramount Pictures', 'Legendary Pictures', 'Warner Bros.', 'Syncopy', 'Lynda Obst Productions'], 
             'vote_average':8.1  }


Alien_Covenant = { # revenue: 240891763
             'budget':      97000000 , 
             'cast':        ['Michael Fassbender', 'Katherine Waterston', 'Billy Crudup', 'Danny McBride'], 
             'crew':        'Ridley Scott', 
             'genres':      ['Horror','Science Fiction','Thriller'], 
             'keywords':    ['xenomorph', 'alien', 'android', 'alien technology', 'alien space craft'], 
             'popularity':  463 , 
             'production_companies':['Twentieth Century Fox', 'TSG Entertainment', 'Scott Free Productions', 'Brandywine Productions'], 
             'vote_average':6.4  }


Blade_Runner_2049 = { # revenue: $259344059
             'budget':      150000000 , 
             'cast':        ['Ryan Gosling', 'Dave Bautista', 'Robin Wright', 'Mark Arnold'], 
             'crew':        'Denis Villeneuve', 
             'genres':      ['Drama', 'Science Fiction', 'Mystery'], 
             'keywords':    ['black bra', 'menage a trois', 'short skirt', 'two women one man'], 
             'popularity':  170, 
             'production_companies':['Alcon Entertainment', 'Columbia Pictures Corporation', 'Sony', 'Torridon Films', '16:14 Entertainment', 'Scott Free Productions', 'Babieka'], 
             'vote_average':8.0  }
#
#Ghost_in_the_Shell
#
#Star_Wars_The_Last_Jedi

Vk   = 0
k_nn = 5
Predict_ROI = ROI_prediction(Blade_Runner_2049, signal, Vk, features, k_nn, File_path, genre = Genre)
print('Our ROI prediction on the given movie is', Predict_ROI)