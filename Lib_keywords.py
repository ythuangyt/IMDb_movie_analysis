# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:55:37 2019

@author: Gentle Deng

"""

import nltk
import pandas as pd
from nltk.corpus import wordnet

def pipe_flatten_names(keywords):
    return '|'.join([x['name'].replace(' ', '_') for x in keywords])


def keywords_inventory(dataframe, colomne = 'keywords'):
    """
    Returns
    -------
    keywords_roots: dictionary
                    root <-> keywords that share the same root
    keywords_select: dictionary
                    root <-> keyword that we keep for a particular root
    category_keys: list
                    all keywords that we keep
    """
    PS = nltk.stem.PorterStemmer()
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    
    for s in dataframe[colomne]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower() ; racine = PS.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(colomne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select


# Replace keywords in the dataframe, If roots = True, replace conjugates;
#If roots = False, replace synonymes
#----------------------------------------------
def replacement_df_keywords(df, dico_replacement, roots = False):
    PS = nltk.stem.PorterStemmer()
    df_new = df.copy(deep = True)
    for index, row in df_new.iterrows():
        chaine = row['keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'): 
            clef = PS.stem(s) if roots else s
            if clef in dico_replacement.keys():
                nouvelle_liste.append(dico_replacement[clef])
            else:
                nouvelle_liste.append(s)       
        df_new.loc[index, 'keywords'] =  '|'.join(nouvelle_liste)
    return df_new


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count    


#get synonymes for a particular keyword
#-------------------------------------------
def get_synonymes(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            #_______________________________
            # We just get the 'nouns':
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n': lemma.add(w.lower())        
    return lemma   


#test if a keyword has a synonyme of higher frequency
#-------------------------------------------
def test_keyword(mot, key_count, threshold):
    return (False , True)[key_count.get(mot, 0) >= threshold]


#deletion of keywords with low frequencies (appear less than 3 times)
def replacement_df_low_frequency_keywords(df, keyword_occurences):
    df_new = df.copy(deep = True)
    key_count = dict()
    for s in keyword_occurences: 
        key_count[s[0]] = s[1]   
    for index, row in df_new.iterrows():
        chaine = row['keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'): 
            if key_count.get(s, 4) > 3: nouvelle_liste.append(s)
            #4 is the Value to be returned in case key does not exist
        df_new.loc[index, 'keywords'] =  '|'.join(nouvelle_liste)
    return df_new