# -*- coding: utf-8 -*-
"""
Created on Tues Oct 16 23:33:04 2018

@author: Ken Huang
"""

import codecs
import os
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def flatten(input_list):
    '''
    A function to flatten complex list.
    :param input_list: The list to be flatten
    :return: the flattened list.
    '''

    flat_list = []
    for i in input_list:
        if type(i) == list:
            flat_list += flatten(i)
        else:
            flat_list += [i]

    return flat_list


def common_words(path):
    '''
    A function to read-in the top common words from external .txt document.
    :param path: The path where the common words info is stored.
    :return: A set of the top common words.
    '''

    with codecs.open(path) as f:
        words = f.read()
        words = json.loads(words)

    return set(words)


def read_novel(book_name, path):
    '''
    A function to read-in the novel text from given path.
    :param book_name: The name of the novel.
    :param path: The path where the novel text file is stored.
    :return: the novel text.
    '''

    book_list = os.listdir(path)
    book_list = filter(lambda x: x.find(book_name) >= 0, book_list)
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel


def name_entity_recognition(sentence):
    '''
    A function to retrieve name entities in a sentence.
    :param sentence: the sentence to retrieve names from.
    :return: a name entity list of the sentence.
    '''

    doc = nlp(sentence)
    # retrieve person and organization's name from the sentence
    name_entity = [x for x in doc.ents if x.label_ in ['PERSON', 'ORG']]
    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s","") for x in name_entity]
    # split names into single words ('Harry Potter' -> ['Harry', 'Potter'])
    name_entity = [x.split(' ') for x in name_entity]
    # flatten the name list
    name_entity = flatten(name_entity)
    # remove name words that are less than 3 letters to raise recognition accuracy
    name_entity = [x for x in name_entity if len(x) >= 3]
    # remove name words that are in the set of 4000 common words
    name_entity = [x for x in name_entity if x not in words]

    return name_entity


def iterative_NER(sentence_list, threshold_rate=0.0005):
    '''
    A function to execute the name entity recognition function iteratively. The purpose of this
    function is to recognise all the important names while reducing recognition errors.
    :param sentence_list: the list of sentences from the novel
    :param threshold_rate: the per sentence frequency threshold, if a word's frequency is lower than this
    threshold, it would be removed from the list because there might be recognition errors.
    :return: a non-duplicate list of names in the novel.
    '''

    output = []
    for i in sentence_list:
        name_list = name_entity_recognition(i)
        if name_list != []:
            output.append(name_list)
    output = flatten(output)
    from collections import Counter
    output = Counter(output)
    output = [x for x in output if output[x] >= threshold_rate * len(sentence_list)]

    return output


def top_names(name_list, novel, top_num=20):
    '''
    A function to return the top names in a novel and their frequencies.
    :param name_list: the non-duplicate list of names of a novel.
    :param novel: the novel text.
    :param top_num: the number of names the function finally output.
    :return: the list of top names and the list of top names' frequency.
    '''

    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names


def calculate_matrix(name_list, sentence_list, accompany_rate=0.3):
    '''
    Function to calculate the co-occurrence matrix and sentiment matrix among all the top characters
    :param name_list: the list of names of the top characters in the novel.
    :param sentence_list: the list of sentences in the novel.
    :param accompany_rate: the sentiment intimacy growth rate between two characters, every time two characters
    co-occurr in one sentence, the sentiment intimacy between them will grow by one accompany rate.
    :return: the co-occurrence matrix and sentiment matrix.
    '''

    # calculate a sentiment score for each sentence in the novel
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    # calculate occurrence matrix and sentiment matrix among the top characters
    name_vect = CountVectorizer(vocabulary=name_list, binary=True)
    occurrence_each_sentence = name_vect.fit_transform(sentence_list).toarray()
    cooccurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += accompany_rate * cooccurrence_matrix
    cooccurrence_matrix = np.tril(cooccurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)
    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = cooccurrence_matrix.shape[0]
    cooccurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    return cooccurrence_matrix, sentiment_matrix


def matrix_to_edge_list(matrix, mode, name_list):
    '''
    Function to convert matrix (co-occurrence/sentiment) to edge list of the network graph. It determines the
    weight and color of the edges in the network graph.
    :param matrix: co-occurrence matrix or sentiment matrix.
    :param mode: 'co-occurrence' or 'sentiment'
    :param name_list: the list of names of the top characters in the novel.
    :return: the edge list with weight and color param.
    '''
    edge_list = []
    shape = matrix.shape[0]
    lower_tri_loc = list(zip(*np.where(np.triu(np.ones([shape, shape])) == 0)))
    normalized_matrix = matrix / (np.max(matrix) - np.min(matrix))
    if mode == 'co-occurrence':
        weight = np.log(2000 * normalized_matrix + 1) * 0.7
        color = np.log(2000 * normalized_matrix + 1)
    if mode == 'sentiment':
        weight = np.log(np.abs(1000 * normalized_matrix) + 1) * 0.7
        color = 4000 * normalized_matrix
    for i in lower_tri_loc:
        edge_list.append((name_list[i[0]], name_list[i[1]], {'weight': weight[i], 'color': color[i]}))

    return edge_list


def plot_graph(name_list, name_frequency, matrix, plt_name, mode, path=''):
    '''
    Function to plot the network graph (co-occurrence network or sentiment network).
    :param name_list: the list of top character names in the novel.
    :param name_frequency: the list containing the frequencies of the top names.
    :param matrix: co-occurrence matrix or sentiment matrix.
    :param plt_name: the name of the plot (PNG file) to output.
    :param mode: 'co-occurrence' or 'sentiment'
    :param path: the path to output the PNG file.
    :return: a PNG file of the network graph.
    '''

    label = {i: i for i in name_list}
    edge_list = matrix_to_edge_list(matrix, mode, name_list)
    normalized_frequency = np.array(name_frequency) / np.max(name_frequency)

    plt.figure(figsize=(20, 20))
    G = nx.Graph()
    G.add_nodes_from(name_list)
    G.add_edges_from(edge_list)
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    colors = [G[u][v]['color'] for u, v in edges]

    if mode == 'co-occurrence':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 2000, edge_cmap=plt.cm.Blues,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True, width=weights)
    elif mode == 'sentiment':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 2000,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True,
                width=weights, edge_vmin=-1000, edge_vmax=1000)
    else:
        raise ValueError("mode should be either 'co-occurrence' or 'sentiment'")

    plt.savefig(path + plt_name + '.png')


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    words = common_words('common_words.txt')
    novel_name = 'Harry Potter'
    novel_folder = Path(os.getcwd()) / 'novels'
    novel = read_novel(novel_name, novel_folder)
    sentence_list = sent_tokenize(novel)
    preliminary_name_list = iterative_NER(sentence_list)
    name_frequency, name_list = top_names(preliminary_name_list, novel, 25)
    cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list)
    # plot co-occurrence and sentiment graph for Harry Potter
    plot_graph(name_list, name_frequency, cooccurrence_matrix, novel_name + ' co-occurrence graph', 'co-occurrence')
    plot_graph(name_list, name_frequency, sentiment_matrix, novel_name + ' sentiment graph', 'sentiment')

    # plot network graph by season
    novel_list = [novel_name + ' ' + str(season) for season in range(1, 8)]
    for name in novel_list:
        novel = read_novel(name, novel_folder)
        sentence_list = sent_tokenize(novel)
        cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list)
        plot_graph(name_list, name_frequency, cooccurrence_matrix, name + ' co-occurrence graph', 'co-occurrence')
        plot_graph(name_list, name_frequency, sentiment_matrix, name + ' sentiment graph', 'sentiment')
