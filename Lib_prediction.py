# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:09:47 2019

@author: Gentle Deng
"""
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from Lib_graph import load_tmdb_movies, load_tmdb_credits

def get_bag_of_word_pred(doc_list):
    docs = []
    for doc in doc_list:
        doc = " ".join(doc)
        docs.append(doc)
    # Create bag of words features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    #In case that one actor plays two roles
    X[X>1] = 1
    return X, vectorizer


#generate distance 
def generate_dist_pred(bag):
    tmp1 = bag@bag.T.toarray()
    tmp2 = np.asarray((bag.T.multiply(bag.T)).sum(axis = 0))
    dist = 1 - tmp1/(tmp2.max())
    return dist, tmp2.max()


#generate adjacency
def generate_adja_pred(dist):
    kernel_width = dist.mean()
    adjacency = np.exp(-dist**2 / kernel_width**2)
    threshold = 0
    #threshold = np.exp(-1 / kernel_width**2)
    adjacency[adjacency <= threshold] = 0
    np.fill_diagonal(adjacency, 0)
    return adjacency, kernel_width


def build_feature_vector(name, name_list):
    feature_vector = np.zeros(len(name_list))
    for i in range(len(name)):
        for j in range(len(name_list)):
            if name[i] == name_list[j]:
                feature_vector[j] = 1
    return feature_vector


def build_weight_vector(feature_vector, bag, kernel_width, norm):
    dist = 1 - bag.toarray().dot(feature_vector)/norm
    weight = np.exp(-dist**2 / kernel_width**2)
    return weight


def construct_peoplevec(features, new_actor, new_director, new_keyword):
    #get cast_name for each movie
    cast_name = []
    for index,i in zip(features.index, features['cast']):
        cast_name_temp = []
        for j in range(len(i)):
            temp = i[j]['name']
            cast_name_temp.append(temp)
        cast_name.append(cast_name_temp)

    cast_name2 = [] 
    for index, i in enumerate(cast_name):
        if len(i)>=4:
            temp = i[:4]#only keep the former four actors
        elif len(i)==0:
            temp = []
        else:
            temp = i
        #replace special characters in names
        temp = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x), temp))
        cast_name2.append(temp)
        
    bag_of_actors, model1 = get_bag_of_word_pred(cast_name2)
    adjacency_actor, kernel_width_actor = generate_adja_pred(generate_dist_pred(bag_of_actors)[0])

    norm_actor = generate_dist_pred(bag_of_actors)[1]

    #extract director for each movie
    crew_name = []
    for item in features.crew:
        tmp2 = []
        tmp3 = []
        for i in item:
            if i['job'] == 'Director':
                tmp2.append(i['name'])
        tmp2 = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x), tmp2))
        #If there are more than 2 directors in a movie, keep the first two
        if len(tmp2)>2:
            tmp3 = tmp2[:2]
        else:
            tmp3 = tmp2
        crew_name.append(tmp3)
                   
    bag_of_directors, model2 = get_bag_of_word_pred(crew_name)
    adjacency_director, kernel_width_dire = generate_adja_pred(generate_dist_pred(bag_of_directors)[0])
    #np.save("adjacency_director", adjacency_director)
    norm_dire = generate_dist_pred(bag_of_directors)[1]
    
    features_keywords = features['keywords'].map(lambda x: [item['name'] for item in x])
    features_keywords = features_keywords.map(lambda x: [re.sub('[^A-Za-z0-9_]+', '_', item) for item in x if len(x)!=0])
    # features_keywords = features_keywords.map(lambda x: [item.replace(' ', '_') for item in x if len(x)!=0])
    features_keywords = list(features_keywords)
    
    bag_of_keywords, model3 = get_bag_of_word_pred(features_keywords)
    adjacency_keyword, kernel_width_key = generate_adja_pred(generate_dist_pred(bag_of_keywords)[0])
    norm_key = generate_dist_pred(bag_of_directors)[1]
    
    actor_list = model1.get_feature_names()
    dire_list = model2.get_feature_names()
    keyword_list = model3.get_feature_names()
    
    new_actor = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x).lower(), new_actor[:4] if len(new_actor)>4 else new_actor))
    new_director = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x).lower(), new_director[:2] if len(new_director)>2 else new_director[:1]))
    new_keyword = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x).lower(), new_keyword))
    
    feature_vector_actor = build_feature_vector(new_actor, actor_list)
    feature_vector_dire = build_feature_vector(new_director, dire_list)
    feature_vector_keyword = build_feature_vector(new_keyword, keyword_list)
    
    weight_actor = build_weight_vector(feature_vector_actor, bag_of_actors, kernel_width_actor, norm_actor)
    weight_dire = build_weight_vector(feature_vector_dire, bag_of_directors, kernel_width_dire, norm_dire)
    weight_keyword = build_weight_vector(feature_vector_keyword, bag_of_keywords, kernel_width_key, norm_key)
    return weight_actor, weight_dire, weight_keyword


def vote_vector(value, features):        
    vote_avg = features.vote_average
    N_node   = vote_avg.shape[0]
    
    print('---> Generating weight vector of votes ...')
    V_vote   = np.zeros([N_node, 1])
    Dis_vote = np.zeros([N_node, 1])

    # compute distance of average votes between each nodes
    Dis_vote = np.abs(vote_avg.values - value)

    # calculate the Wright matrix of the IMDb vote graph
    sigma  = np.nanstd(Dis_vote)
    V_vote = np.exp( (-Dis_vote**2)/(sigma*sigma) )
    V_vote = V_vote.reshape( len(V_vote), 1)
    print("---> Weight matrix of votes has been generated")
    return V_vote
        

def budget_vector(value, features):
    
    print('---> Generating weight matrix of budget ...')
    budget = features['budget'].values
    budget = budget.astype(float)
    length = len(budget)
    budget = budget.reshape(length, 1)
    loc = np.where(budget <= 1000)[0]
    budget[loc] = np.nan
    
    N_node   = budget.shape[0]
    
    print('---> Generating weight vector of budget ...')
    V_budg   = np.zeros([N_node, 1])
    Dis_budg = np.zeros([N_node, 1])

    # compute distance of average votes between each nodes
    Dis_budg = np.abs(budget - value)

    # calculate the Wright matrix of the IMDb vote graph
    sigma  = np.nanmean(Dis_budg)*1.5
    V_budg = np.exp( (-Dis_budg**2)/(sigma*sigma) )
    print('---> Weight matrix of budget has been generated')
    
    return V_budg

    
def popularity_vector(value, features):    
    print('---> Generating weight matrix of popularity ...')
    popularity = features['popularity'].values
    length = len(popularity)
    popularity = popularity.reshape(length,1)

    N_node = popularity.shape[0]

    print('---> Generating weight vector of popularity ...')
    V_pop   = np.zeros([N_node, 1])
    Dis_pop = np.zeros([N_node, 1])

    # compute distance of average votes between each nodes
    Dis_pop = np.abs(popularity - value)

    # calculate the Wright matrix of the IMDb vote graph
    sigma  = Dis_pop.mean()
    V_pop = np.exp( (-Dis_pop**2)/(sigma*sigma) )
    print("---> Weight matrix of popularity has been generated")
    return V_pop

#def company_vector(value, features):
#    companies = features.production_companies 
#    N_node    = companies.shape[0]
#    
#    print('---> Generating weight matrix of companies ...')
#    W_comp   = np.zeros([N_node, N_node])
#    cos_dist = np.zeros([N_node, N_node])
#
#    companies_id   = companies.apply(lambda x: [ (item['id']) for item in x ])
#
#    # calculate cosine distance matrix
#    for ct_setA in range(N_node):
#    for ct_setB in range(N_node):
#        cos_dist[:, ct_setB] = 1 - len(set(companies_id[ct_setA]).intersection(set(companies_id[ct_setB]))) \
#                                     / ( np.sqrt(len(companies_id[ct_setA]) * len(companies_id[ct_setB]))+ 1e-10)
#
#    # restrict diagnal entries to be 0
#    np.fill_diagonal(cos_dist, 0)
#    cos_dist = cos_dist.T + cos_dist
#
#    # calculate weight matrix of production companies
#    sigma  = 0.5
#    W_comp = np.exp( (-cos_dist**2)/(sigma*sigma) )
#    W_comp = W_comp - np.diag(np.ones(N_node))
#    
#    print('---> Weight matrix of companies has been generated')
#    return W_comp


# Genrating weight vectors
def compute_weight_vector(sub_feature, value, path_to_file):
    # files to read
    movies_name = '/tmdb_5000_movies.csv'
    credits_name = '/tmdb_5000_credits.csv'
    
    # read IMDb data
    movies = load_tmdb_movies( path_to_file + movies_name)
    credit = load_tmdb_credits( path_to_file + credits_name)

    # Id has been droped, we must align all data during construction
    IMDb   = credit.merge(movies, left_on = 'movie_id', right_on = 'id') 
    features = IMDb[sub_feature]
    del movies, credit, IMDb
    
    W_vec_3d = np.array([])
    ct_value = 0
    for feature in sub_feature:
        if feature == 'cast':
            new_actor = value[ct_value]
            ct_value += 1
            continue
        elif feature == 'vote_average':
            weight_vec  = vote_vector  ( value[ct_value], features )
        elif feature == 'budget':
            weight_vec  = budget_vector( value[ct_value], features )
        elif feature == 'crew':
            new_director = value[ct_value]
            ct_value += 1
            continue
        elif feature == 'genres':
            print("Since we separate the whole in different genres, we won't use genre as a feature")
            ct_value += 1
            continue
        elif feature == 'keywords':
            new_keyword  = value[ct_value]
            ct_value += 1
            continue
        elif feature == 'popularity':
            weight_vec  = popularity_vector(value[ct_value], features)
        elif feature == 'production_companies':
#            weight_vec  = company_vector(value[ct_value], features )
            weight_vec = np.zeros(features.shape[0])
            weight_vec = weight_vec.reshape( len(weight_vec), 1)
        else:
            print( '[Warning]', feature, 'is not supported now' )
        ct_value += 1
        W_vec_3d = np.dstack([W_vec_3d, weight_vec]) if W_vec_3d.size else weight_vec
        
    W_vec_act, W_vec_dir, W_vec_key = construct_peoplevec(features, new_actor, new_director, new_keyword) 
    W_vec_act = W_vec_act.reshape( len(W_vec_act), 1)
    W_vec_dir = W_vec_dir.reshape( len(W_vec_dir), 1)
    W_vec_key = W_vec_key.reshape( len(W_vec_key), 1)
    W_vec_3d = np.dstack([W_vec_3d, W_vec_act]) if W_vec_3d.size else W_vec_act
    W_vec_3d = np.dstack([W_vec_3d, W_vec_dir]) if W_vec_3d.size else W_vec_dir
    W_vec_3d = np.dstack([W_vec_3d, W_vec_key]) if W_vec_3d.size else W_vec_key
    return W_vec_3d


def check_input(path_to_file, genre):
    # if genre is given, load the Vk if it has been generated
    if genre is not None:
        try:
            Vk = np.load(path_to_file + '/Vk_list_' + genre + '.npy')
            old_Vk = Vk[Vk.shape[0]-1 :]
            print('[The Vk on the give path has been loaded!]')
        except:
            old_Vk = None
            print('[The Vk does not exist in the give path! Try to use the input Vk!]')
    # otherwise, directly use the given Vk
    else:
        old_Vk = None
        print('[Only use the input Vk!]')
    return old_Vk


def ROI_prediction(node, signal, Vk, features, k_nn, path_to_file, genre = None):
    
    old_Vk = check_input(path_to_file, genre)
    if old_Vk is not None:
        Vk = old_Vk
    
    feature_vec = [None] * len(features)
    # 1. filter features
    for key, value in node.items():
        if value is None:
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
        weight = feature_weight[:,:,k] * Vk[0,k]
    ROI_neighbor = (-weight).argsort(axis = 0)[:k_nn]
    
    return (sum(signal[ROI_neighbor])/k_nn)
    