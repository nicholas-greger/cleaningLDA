#
# cleaningLDA
# 
# [Insert Paper Reference Here]
#
# Written by Nicholas Greger
# 2022 (c)
import logging as log
import gensim
from gensim.models import LdaModel, CoherenceModel, TfidfModel
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
# from gensim.test.utils import datapath
# from gensim.test.utils import common_texts
# from gensim import corpora

# Nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# General Packages
import os
import pickle
import regex as re
import json
import csv
import itertools
import math
from math import nan, isnan
import tokenize
import random
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Pandas, Numpy, Scipy
import pandas as pd
import numpy as np
from numpy import matlib

import scipy
from scipy import stats
from scipy.stats import *
from scipy.spatial import distance

import sklearn 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as hac

def clear():
    os.system("clear")

def run_lda(corpus, 
            dictionary, 
            texts, 
            t_num):
    """Runs the standard implementation of LDA on the entire corpus
    
    Parameters
    ----------
    corpus : iterable of list of (int, float)
        From gensim: Stream of document vectors or sparse matrix of shape (num_documents, num_terms).
    dictionary : dict of (int,str)
        gensim.corpora.dictionary.Dictionary object of corpus.
    texts : list of list of str
        Tokenized texts used for the cohernece model.
    t_num : int
        Number of topics to train the topic model on
    
    Returns
    -------
    Dictionary of the pretrained LDA model and the c_v coherence score for that topic
    """
    alphas_p = [t_num/50 for _ in range(t_num)]
    beta_p = 0.01
    model = LdaModel(corpus, id2word=dictionary,
        alpha=alphas_p,
        eta=beta_p, 
        num_topics=t_num,
        passes=10,
        iterations=1000,
        random_state=1997)

    coherencemodel = CoherenceModel(model=model, 
                                    texts=texts, 
                                    dictionary=dictionary, 
                                    corpus = corpus, 
                                    coherence='c_v', 
                                    window_size = 50)
    coherence_score = coherencemodel.get_coherence()

    # append this to our models to consider best one
    
    return [model, coherence_score]

def pretrain_topic_models(path_to_corpus, topics_to_try, find_best_model = True, verbose = True):
    """Pretrains the LDA topic model on a corpus using a specifed number of topics
    
    Parameters
    ----------
    path_to_corpus : str
        String path to the directory of the corpus. Must be .txt and each line of the file must be your documents per timeperiod. 
    
    topics_to_try : list of int
        List of number of topics to train the LDA topic model on. 
    
    find_best_model : bool
        If True, determines the best topic model for the range of topics using elbow detection.
    
    verbose : bool
        Text output of steps
        
    Returns
    -------
    dict
        Dictionary of the coherence scores and number of topics of the LDA model. 
        
    Saves as .pkl either: the most optimal model given c_v coherence or all possible models in the Step_One_Models folder.
    """
    # Reading in file and tokenizing the document 

    # Reading in the corpus along lines
    with open(path_to_corpus, 'r') as f:
        corpus_untokenized = f.read().splitlines()
    
    # corpus_sentences = corpus_untokenized
    # print(corpus_untokenized[0])
    # Tokenizing the corpus
    tokens = []
    for document in corpus_untokenized:
        tokens.append(word_tokenize(document))
    # tokens = [word_tokenize(t) for t in corpus_untokenized]
 
    # Creating dictionary object
    dictionary = Dictionary(tokens)
    
    # Create corpus
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Generating texts
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
    
    # Initializing coherence_score and models lists 
    coherence_scores = []
    models = []
    
    # Pretraining the LDA model along the number of topics
    for t_num in tqdm(topics_to_try, desc = 'Training LDA on Topics', disable = not verbose):
        model_results = run_lda(corpus, dictionary, texts, t_num)
        
        models.append(model_results[0]) # Appending LDA Models per number of topics for a specific tfidf value
        coherence_scores.append(model_results[1]) # Appending the coherence scores per topic
    attributes = {}
    # Finding the optimal elbow point
    if (find_best_model == True): # If automatic elbow detection is enabled, find it
        best_point_index = bestk_linalg(coherence_scores, topics_to_try)

        best_topic_number = topics_to_try[best_point_index] # Getting the index of 
        pickle.dump(models[best_point_index], open('data/topics_' + str(best_topic_number) + '_pretrained_model.pkl', 'wb'))
        attributes['best_model_index'] = best_point_index
    else: # If automatic detection is not enabled, do not find it and move on
        temp_i = 0
        for model in models:
            pickle.dump(model, open('data/topics_' + str(topics_to_try[temp_i]) + '_pretrained_model.pkl', 'wb'))
            temp_i += 1
    
    attributes['coherence_scores'] = coherence_scores
    attributes['topics'] = topics_to_try
    
    return attributes
def bestk_linalg(cohesion, 
                 t_nums):
    """Finds the elbow of a plot by finding the point furthest away from the line created from the first to last point of a plot.
    
    Parameters
    ----------
    cohesion : list
        List of coherence values.
    t_nums : list
        List of the number of topics. 
    Returns
    -------
    int
        Elbow of the graph.
    """
    points = np.vstack((t_nums, cohesion)).T
    
    lineVec = points[-1] - points[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    
    vecFromFirst = points - points[0]
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, len(points), 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    
    return idxOfBestPoint

def train_topic_models(path_to_corpus,
                       chosen_model):
    """Trains the LDA topic model on the entire corpus given the pretrained model. 
    
    Parameters
    ----------
    path_to_corpus : string
        String path to the directory of the corpus. Thus must be the same corpus as the one used in pretrained_topic_models.
    
    chosen_model : string
        string of the filename of the .pkl file of the pretrained model. 
    
    Returns
    -------
    Generates three files:
        1. The trained topic model in Step_Two_Models as a .pkl file.
        2. Document topics 
        3. Document-Topic distribution
    """
    topics = parse_model_features(chosen_model) # Passing over the model path to get the important features of the model

    # Reading in the document a second time, this time reading in lines for get the weekly topic distributions
    # with open(path_to_corpus, 'r') as read_document: # Loading in the document
    #     documents = read_document.read()
    
    with open(path_to_corpus, 'r') as f:
        corpus_untokenized = f.read().splitlines()
    
    # corpus_sentences = corpus_untokenized
    # print(corpus_untokenized[0])
    # Tokenizing the corpus
    document_tokens = []
    for document in corpus_untokenized:
        document_tokens.append(word_tokenize(document))
    
    
    # document = documents.splitlines() # Splitting along newlines such that each item in list is a week worth of tweets
    # document_tokens = [doc.split(' ') for doc in document] # Tokenizing 

    dic = Dictionary() # Creating a dictionary off of the list of tokenized documents
    dic.add_documents(document_tokens)
    old_corpus = [dic.doc2bow(text) for text in document_tokens] # Creating a corpus

    model_path = chosen_model

    with open(model_path, 'rb') as read_pickle:
        pretrained_lda_model = pickle.load(read_pickle) # Reading in the LDA model

    trained_result = pretrained_lda_model.get_document_topics(old_corpus, minimum_probability = 0.01)
    document_topics = pretrained_lda_model.print_topics(topics, 20)
    # print(document_topics)
    trained_result_as_list = []
    for result in trained_result:
        trained_result_as_list.append(result)
    
    with open('data/topics_' + str(topics) + '_step_two_model.pkl', 'wb') as outfile: # Writing list to pickle
            pickle.dump(trained_result, outfile)

    with open('data/topics_' + str(topics) + '_document_topics.txt', 'a') as outfile:
        outfile.write(str(document_topics))

    with open('data/topics_' + str(topics) + '_document_topic_distributions.pkl', 'wb') as outfile:
        pickle.dump(trained_result_as_list, outfile)
    
    
def parse_model_features(chosen_model):
    """Parses the file name of the model to get model features
    
    Parameters
    ----------
    chosen_model : str
        String of the file name
    
    Returns
    -------
    list
        List of the number of topics
    """
    split_text = chosen_model.split("_")
    
    # Getting facts about the model from the title of the model. 
    topics = int(split_text[1])
    # cutoff = float(split_text[1])
    return topics

def clean_distributions(distribution, num_topics):
    """Cleans the probability distribution by adding missing topic numbers and the associated 0 probability.
    
    Parameters
    ----------
    distribution : list of list
        Probability distribution directly given by gensim topic model.
    
    num_topics : int
        Number of topics trained on the model giving the probability distribution.
        
    Returns
    -------
    list
        Cleaned topic distribution.
    """
    model_master = []
    for q in range(len(distribution)):
        model_temp = distribution[q]
        x = 0
        topic_range = range(0,num_topics)
        model_amended = []
        for z in topic_range:
            try: 
                if (model_temp[x][0] == z):
                    model_amended.append(model_temp[x])
                    x += 1
                else:
                    model_amended.append((z, 0))
            except IndexError:
                model_amended.append((z, 0))
                pass
        model_master.append(model_amended)
    return model_master

def keep_y(distributions):
    """Keeps the y values of a distribution. 
    
    Parameters
    ----------
    distributions : list
        Probability distribution.
        
    Returns
    -------
    list
        List of y values.
    """
    y_values = []
    for distribution in distributions:
        y_values.append([x[1] for x in distribution])
    return y_values

def js(distributions):
    """Calculates the Jensen-Shannon Distance between the probability distributions
    
    Parameters
    ----------
    distributions : list of lists
        List of probability distributions
    
    Returns
    -------
    list
        Jensen-Shannon distances.
    """
    distributions_stripped = keep_y(distributions)
    js_master = []
    for z in range(len(distributions_stripped)):
        js = []
        for p in distributions_stripped:
            dist = distance.jensenshannon(distributions_stripped[z], p)
            js.append(dist)
        js_master.append(js)
    
    return js_master

def determine_changepoint(js_distances, distance_cutoff, num_epochs):
    """Determines the changepoint given the cutoff and number of occurances.
    
    Parameters
    ----------
    js_distances : list
        List of Jensen-Shannon distances. Must be 1 dimensional.
    distance_cutoff : float
        Jensen-Shannon distance threshold.
    num_epochs : int
        If a document's Jensen-Shannon distacne is above distance_cutoff for the number of epochs, detect changepoint. 
    
    Returns
    -------
    int
        Returns the changepoint. 
    """
    occurances_above_score_cutoffs = []
    
    for js_distance in js_distances:
        if (len(occurances_above_score_cutoffs) < num_epochs):
            if (js_distance >= distance_cutoff):
                occurances_above_score_cutoffs.append(js_distance)
            else:
                occurances_above_score_cutoffs = []
    try:
        return js_distances.index(occurances_above_score_cutoffs[0])
    except IndexError:
        return -1

def detect_changepoint(topic_distributions_path, 
                       distance_cutoff = 0.6, 
                       num_epochs = 4):
    """Calculates the changepoint of the first document using the trained LDA model.
    
    Parameters
    ----------
    chosen_trained_model : string
        Path to the trained model
    distance_cutoff : float
        Jensen-Shannon distance threshold.
    num_epochs : int
        If a document's Jensen-Shannon distance is above distance_cutoff for the number of epochs, detect changepoint. 
        
    Returns
    -------
    int
        Changepoint. If no changepoint is detected a -1 is returned. 
    list
        List of Jensen-Shannon distances.
    """
    with open(topic_distributions_path, 'rb') as infile:
        trained_distributions = pickle.load(infile)
        
    topics = parse_model_features(topic_distributions_path)
    
    trained_distributions_cleaned = clean_distributions(trained_distributions, topics)
    js_trained = js(trained_distributions_cleaned)

    first_irrelevant_week = determine_changepoint(js_trained[0], distance_cutoff, num_epochs)
    
    attributes = {}
    
    attributes['changepoint'] = first_irrelevant_week
    attributes['distances'] = js_trained
    
    return attributes 

def visualize(topic_distributions_path, pairwise = False):
    """Visualizes the Jensen-Shannon distances.
    
    Parameters
    ----------
    chosen_trained_model : string
        Path to the trained model
    pairwise : bool
        Visualize either the first week or the entire matrix
    
    Returns
    -------
    
    """
    with open(topic_distributions_path, 'rb') as infile:
        trained_distributions = pickle.load(infile)
        
    topics = parse_model_features(topic_distributions_path)

    trained_distributions_cleaned = clean_distributions(trained_distributions, topics)
    js_trained = js(trained_distributions_cleaned)
    
    if pairwise:
        distances = []
        for dist in js_trained:
            distances.append(np.nan_to_num(dist))
            
        fig, ax = plt.subplots(figsize=(5,5))
        plt.title("Distance Matrix_" + str(topics))
        plt.imshow(js_trained, cmap = 'viridis')
        plt.savefig("data/Distance Matrix_" + str(topics) + '.png')
    else:
        fig, ax = plt.subplots(figsize=(5,5))
        plt.title("Initial Week Distance Matrix_" + str(topics))
        plt.plot(js_trained[0])
        plt.savefig("data/Initial_Week_Distance Matrix_" + str(topics) + '.png')
        plt.show()