# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:07:09 2019

@author: Gentle Deng
"""
import re
import json
import numpy as np
import pandas as pd
from Lib_keywords import pipe_flatten_names, keywords_inventory, replacement_df_keywords, count_word, get_synonymes, test_keyword, replacement_df_low_frequency_keywords

from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer

pd.options.mode.chained_assignment = None  # default='warn'

# In[]
# using kernal provided by kaggle to load data
def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df

# In[]
# generate distance 
def generate_dist(bag):
    tmp1 = bag@bag.T.toarray()
    tmp2 = np.asarray((bag.T.multiply(bag.T)).sum(axis = 0))
    dist = 1 - tmp1/(tmp2.max())
    return dist

# Create bag of words features
def get_bag_of_word(doc_list):
    docs = []
    for doc in doc_list:
        doc = " ".join(doc)
        docs.append(doc)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    return X

# generate adjacency
def generate_adja(dist):
    kernel_width = dist.mean()
    adjacency = np.exp(-dist**2 / kernel_width**2)
    np.fill_diagonal(adjacency, 0)
    return adjacency

# In[] 
# functions to generate graphs
    
def actor_graph(File_path, features):
    try:
        W_acto = np.load(File_path + '/weight_mat_actor.npy')
        print('---> The previously generated weight matrix of actor/actress has been loaded.')
        
    except:
        
        print('---> Generating weight matrix of actor/actress ...')
        # get cast_name for each movie
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
    
        bag_of_actors = get_bag_of_word(cast_name2)
    
        W_acto = generate_adja(generate_dist(bag_of_actors))    
        np.save(File_path + '/weight_mat_actor', W_acto)
        print('---> Weight matrix of actor/actress has been generated and has been saved to:', File_path+'/weight_mat_actor.npy')  
    return W_acto


def vote_graph(File_path, features):
    try:
        W_vote = np.load(File_path + '/weight_mat_avgvote.npy')
        print('---> The previously generated weight matrix of votes has been loaded.')
        
    except:
        
        vote_avg = features.vote_average
        N_node   = vote_avg.shape[0]
        
        print('---> Generating weight matrix of votes ...')
        W_vote   = np.zeros([N_node, N_node])
        Dis_vote = np.zeros([N_node, N_node])
    
        # compute distance of average votes between each nodes
        for ct in range(N_node):
            Dis_vote[:,ct] = np.abs(vote_avg.values - vote_avg[ct])
    
        # calculate the Wright matrix of the IMDb vote graph
        sigma  = np.nanstd(Dis_vote)
        W_vote = np.exp( (-Dis_vote**2)/(sigma*sigma) )
        W_vote = W_vote - np.diag(np.ones(N_node))
        
        np.save(File_path + '/weight_mat_avgvote', W_vote)
        print('---> Weight matrix of votes has been generated and has been saved to:', File_path+'/weight_mat_avgvote.npy')
    return W_vote
        

def budget_graph(File_path, features):
    try:
        W_budg = np.load(File_path + '/weight_mat_budget.npy')
        print('---> The previously generated weight matrix of budget has been loaded.')
        
    except:
        print('---> Generating weight matrix of budget ...')
        budget = features['budget'].values
        budget = budget.astype(float)
        length = len(budget)
        budget = budget.reshape(length, 1)
        loc = np.where(budget <= 1000)[0]
        budget[loc] = np.nan
        
        # Calculate the budget distance
        distances_budget = pdist(budget, metric='euclidean')
        kernel_width_budget = np.nanmean(distances_budget)*1.5
        # Form the weight matrix of budget
        W_budget_vect = np.exp(-distances_budget**2 / kernel_width_budget**2)
        # Form the adjacency matrix of budget
        W_budg = squareform(W_budget_vect)
        W_budg[np.isnan(W_budg)] = 0
        
        np.save(File_path + '/weight_mat_budget', W_budg)
        print('---> Weight matrix of budget has been generated and has been saved to:', File_path+'/weight_mat_budget.npy')
    return W_budg


def director_graph(File_path, features):
    try:
        W_dire = np.load(File_path + '/weight_mat_director.npy')
        print('---> The previously generated weight matrix of directors has been loaded.')
    
    except:
        #extract director for each movie
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
    
        bag_of_directors = get_bag_of_word(crew_name)
        W_dire = generate_adja(generate_dist(bag_of_directors))
    
        np.save(File_path + '/weight_mat_director', W_dire)
        print('---> Weight matrix of directors has been generated and has been saved to:', File_path+'/weight_mat_director.npy')
    return W_dire


def genre_graph(File_path, features):
    try:
        W_genr = np.load(File_path + '/weight_mat_genre.npy')
        print('---> The previously generated weight matrix of genres has been loaded.')
    
    except:
    
        #generate adjacency for genres
        features_genres = features['genres'].map(lambda x: [item['name'] for item in eval(x)])
        features_genres = features_genres.map(lambda x: [item.replace(' ', '_') for item in x if len(x)!=0])
        features_genres = list(features_genres)
    
        bag_of_genres = get_bag_of_word(features_genres)
        W_genr = generate_adja(generate_dist(bag_of_genres))
      
        np.save(File_path + '/weight_mat_genre', W_genr)
        print('---> Weight matrix of genres has been generated and has been saved to:', File_path+'/weight_mat_genre.npy')
    return W_genr
        

def keyword_graph(File_path, features):
    try:
        W_keyw = np.load(File_path + '/weight_mat_keyword.npy')
        print('---> The previously generated weight matrix of keywords has been loaded.')
    except:
    
        # ---------- Generate sentence list for training Word2Vec model ---------- #
        features_keywords = features['keywords'].map(lambda x: [item['name'] for item in x])
        features_keywords = features_keywords.map(lambda x: [re.sub('[^A-Za-z0-9_]+', '_', item) for item in x if len(x)!=0])
        features_keywords = list(features_keywords)
    
        bag_of_keywords = get_bag_of_word(features_keywords)
        W_keyw = generate_adja(generate_dist(bag_of_keywords))
    
        np.save(File_path + '/weight_mat_keyword', W_keyw)
        print('---> Weight matrix of keywords has been generated and has been saved to:', File_path+'/weight_mat_keyword.npy')
    return W_keyw


def advkeyw_graph(File_path, features):
    try:
        W_advkeyw = np.load(File_path + '/weight_mat_advkeyword.npy')
        print('---> The previously generated weight matrix of cleaned keywords has been loaded.')
    except:
        # Cleaning words share the same root
        # flatten keywords
        IMDb_new = features.copy()
        IMDb_new['keywords'] = IMDb_new['keywords'].apply(pipe_flatten_names)
        keywords, keywords_roots, keywords_select = keywords_inventory(IMDb_new, colomne = 'keywords')
        
        # replace conjugates
        df_keywords_cleaned = replacement_df_keywords(IMDb_new, keywords_select,
                                                       roots = True)
        keyword_occurences, keywords_count = count_word(df_keywords_cleaned, 'keywords', keywords)
        
        # creation of a dictionary to replace keywords by higher frequency synonymes
        keyword_occurences.sort(key = lambda x:x[1], reverse = False)
        key_count = dict()
        for s in keyword_occurences:
            key_count[s[0]] = s[1]
            
        replacement_mot = dict()
        icount = 0
        for index, [mot, nb_apparitions] in enumerate(keyword_occurences):
            if nb_apparitions > 5: continue  # only the keywords that appear less than 5 times
            lemma = get_synonymes(mot)
            if len(lemma) == 0: continue     # case of the plurals
            
            liste_mots = [(s, key_count[s]) for s in lemma
                          if test_keyword(s, key_count, key_count[mot])]
            liste_mots.sort(key = lambda x:(x[1],x[0]), reverse = True)
            if len(liste_mots) <= 1: continue       # no replacement
            if mot == liste_mots[0][0]: continue    # replacement by himself
            icount += 1
            if  icount < 8:
                print('{:<12} -> {:<12} (init: {})'.format(mot, liste_mots[0][0], liste_mots))    
            replacement_mot[mot] = liste_mots[0][0]
            
        # check if there are keywords that appear both in keys and values
        icount = 0
        for s in replacement_mot.values():
            if s in replacement_mot.keys():
                icount += 1
                if icount < 10: print('{:<20} -> {:<20}'.format(s, replacement_mot[s]))
        #successive replacement
        for key, value in replacement_mot.items():
            if value in replacement_mot.keys():
                replacement_mot[key] = replacement_mot[value]
                
        # replacement by synonyms
        df_keywords_synonyms = replacement_df_keywords(df_keywords_cleaned, replacement_mot, roots = False)   
        keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_synonyms, colomne = 'keywords')
                    
        keywords.remove('')
        new_keyword_occurences, new_keywords_count = count_word(df_keywords_synonyms, 'keywords', keywords)
        
        df_keywords_occurence = replacement_df_low_frequency_keywords(df_keywords_synonyms, new_keyword_occurences)
        keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_occurence, colomne = 'keywords')   
        
        # new keywords count
        keywords.remove('')
        new_keyword_occurences, new_keywords_count = count_word(df_keywords_occurence, 'keywords',keywords)
        
        features_keywords = []
        for x in df_keywords_occurence['keywords']:
            tmp = []
            for s in x.split('|'):
                tmp.append(s)
            tmp = list(map(lambda x: re.sub('[^A-Za-z0-9_]+', '_', x), tmp))
            features_keywords.append(tmp)
            
        #create weight matrix based on similarity in terms of keywords
        bag_of_keywords = get_bag_of_word(features_keywords)
        W_advkeyw = generate_adja(generate_dist(bag_of_keywords))    
        np.save(File_path + '/weight_mat_advkeyword', W_advkeyw)
        print('---> Weight matrix of cleaded keywords has been generated and has been saved to:', File_path+'/weight_mat_advkeyword.npy')
    return W_advkeyw


def popularity_graph(File_path, features):
    try:
        W_pop = np.load(File_path + '/weight_mat_pop.npy')
        print('---> The previously generated weight matrix of popularity has been loaded.')
        
    except:
        print('---> Generating weight matrix of popularity ...')
        popularity = features['popularity'].values
        length = len(popularity)
        popularity = popularity.reshape(length,1)
    
        # Calculate the popularity distance
        distances_pop = pdist(popularity, metric='euclidean')
        kernel_width_pop = distances_pop.mean()
        # Form the weight matrix of popularity
        W_pop_vect = np.exp(-distances_pop**2 / kernel_width_pop**2)
        # Form the adjacency matrix of popularity
        W_pop = squareform(W_pop_vect)
        
        np.save(File_path + '/weight_mat_pop', W_pop)
        print('---> Weight matrix of popularity has been generated and has been saved to:', File_path+'/weight_mat_pop.npy')
    return W_pop


def company_graph(File_path, features):
    try:
        W_comp = np.load(File_path + '/weight_mat_companies.npy')
        print('---> The previously generated weight matrix of companies has been loaded.')
        
    except:
    
        companies = features.production_companies 
        N_node    = companies.shape[0]
        
        print('---> Generating weight matrix of companies ...')
        W_comp   = np.zeros([N_node, N_node])
        cos_dist = np.zeros([N_node, N_node])
    
        companies_id   = companies.apply(lambda x: [ (item['id']) for item in x ])
    
        # calculate cosine distance matrix
        for ct_setA in range(N_node):
            for ct_setB in range(ct_setA, N_node):
                cos_dist[ct_setA, ct_setB] = 1 - len(set(companies_id[ct_setA]).intersection(set(companies_id[ct_setB]))) \
                                             / ( np.sqrt(len(companies_id[ct_setA]) * len(companies_id[ct_setB]))+ 1e-10)
    
                if (ct_setA*N_node + ct_setB) % 100000 == 0:
                    pctg = 100 * (ct_setA*N_node + ct_setB)/N_node**2
                    print(f'     {pctg:.2f}% tasks have been finished.', end = '\r')
    
        # restrict diagnal entries to be 0
        np.fill_diagonal(cos_dist, 0)
        cos_dist = cos_dist.T + cos_dist
    
        # calculate weight matrix of production companies
        sigma  = 0.5
        W_comp = np.exp( (-cos_dist**2)/(sigma*sigma) )
        W_comp = W_comp - np.diag(np.ones(N_node))
        
        # save the matrix
        np.save(File_path + '/weight_mat_companies', W_comp)
            
        print('---> Weight matrix of companies has been generated and has been saved to:', File_path+'/weight_mat_avgvote.npy')
    return W_comp
        

def construct_ROIsignal(IMDb, path = None):
    # if the path is given, try to load signal data 
    if path is not None:
        try:
            signal = np.load(path + '/signal.npy')
            print('[The signal on the give path has been loaded!]\n')
            return signal
        except:
            print('[ ! Warning: The signal on the give path is not found, try to generate a signal!]\n')
    else: 
        print("Generating ROI(Return on Investment ) signal ... \n")
        
    budget = IMDb.budget.astype(float)
    budget[budget <= 1000] = np.nan
    
    revenue = IMDb.revenue
    
    # Create the signal
    ROI_signal = (revenue - budget) / budget
    nan_loc = np.where((np.isnan(ROI_signal)) == True)[0]
    ROI_signal[nan_loc] = 0 
    ROI_signal[ROI_signal >= 50] = 0
    
    ROI_signal = ROI_signal.values
    np.save(r'./data/signal', ROI_signal)
    print('Signal has been saved to the default path: ./data/.. \n')
    return ROI_signal

# In[] DATA PICKING
# obtain indices of nodes containing a certain genre
def build_genre_indices(genre, features):
    indices = []
    for index,i in zip(features.index, features['genres']):
        for j in i:
            if j['name'] == genre:
                indices.append(index)
    return indices

# sample the given subgraph according to the given indices 
def build_genre_graph(genre, features, W_3d):
    indices = build_genre_indices(genre, features)
    return indices, W_3d[indices, :, :][:, indices, :]

# In[] main function constructing all subgraphs
def construct_graphs( path_to_file, subgraph_features, genre = None ):
    
    print('********Construction Phase for graphs********')
    
    # files to read
    movies_name = '/tmdb_5000_movies.csv'
    credits_name = '/tmdb_5000_credits.csv'
    
    # read IMDb data
    movies = load_tmdb_movies( path_to_file + movies_name)
    credit = load_tmdb_credits( path_to_file + credits_name)


    # Id has been droped, we must align all data during construction
    IMDb   = credit.merge(movies, left_on = 'movie_id', right_on = 'id') 
    features = IMDb[subgraph_features]
            
    # construct signal and release memories
    signal = construct_ROIsignal(IMDb) # does not give a path --> always generate a new signal
    del movies, credit, IMDb
    
    # construct subgraphs
    print("Generating subgraphs according to the given features ...")
    W_3d = np.array([])
    for feature in subgraph_features:
        if feature == 'cast':
            feature_weight  = actor_graph ( path_to_file, features )
        elif feature == 'vote_average':
            feature_weight  = vote_graph  ( path_to_file, features )
        elif feature == 'budget':
            feature_weight  = budget_graph( path_to_file, features )
        elif feature == 'crew':
            feature_weight  = director_graph(path_to_file, features)
        elif feature == 'genres':
            print("Since we separate the whole in different genres, we won't use genre as a feature")
#            feature_weight  = genre_graph ( path_to_file, features )
            continue
        elif feature == 'keywords':
            feature_weight  = keyword_graph(path_to_file, features )
#            feature_weight = advkeyw_graph( path_to_file, features )
        elif feature == 'popularity':
            feature_weight  = popularity_graph(path_to_file, features)
        elif feature == 'production_companies':
            feature_weight  = company_graph(path_to_file, features )
        else:
            print('[Warning', feature, 'is not supported now]' )
        
        W_3d = np.dstack([W_3d, feature_weight]) if W_3d.size else feature_weight
        
    # set small entries to 0
    W_3d[W_3d <= np.exp(-1)] = 0   # set small weights to zero
    
    # if genre is given, only movies belonging to this genre are analized
    print("Generating genre specified signal and weight ...")
    if genre is not None: 
        indices, W_3d = build_genre_graph(genre, features, W_3d)
        signal = signal[indices]
    
    return signal, W_3d 