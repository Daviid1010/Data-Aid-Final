import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.cluster import KMeans
import SimpSOM as sps
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('LineItemData.csv', encoding='utf-8')

print(data.head())

data = data['Description'].dropna()

print(data.head())

all_lines = data.values.tolist()

print(all_lines[0])


def tokenize_lines(line):
    line = line.lower().replace('\n', ' ')
    for sign in punctuation:
        line = line.replace(sign, '')
    tokens = line.split()
    tokens = [t for t in tokens if t not in STOPWORDS and t != '']
    return tokens


tokenized_lines = [tokenize_lines(poem) for poem in all_lines]


def gimme_glove():
    with open('glove.6B.50d.txt', encoding='utf-8') as glove_raw:
        for line in glove_raw.readlines():
            splitted = line.split(' ')
            yield splitted[0], np.array(splitted[1:], dtype=np.float)


glove = {w: x for w, x in gimme_glove()}


def closest_word(in_vector, top_n=1):
    vectors = glove.values()
    idx = np.argsort([np.linalg.norm(vec - in_vector) for vec in vectors])
    return [glove.keys()[i] for i in idx[:top_n]]


def lines_to_vec(tokens):
    words = [w for w in np.unique(tokens) if w in glove]
    return np.array([glove[w] for w in words])


W = [lines_to_vec(tokenized).mean(axis=0) for tokenized in tokenized_lines]
W = np.array(W)

som_shape = (1000, 1000)
som = MiniSom(som_shape[0], som_shape[1], 50, sigma=0.5, learning_rate=0.5, random_seed=1,
              neighborhood_function='gaussian')
som.random_weights_init(W)
som.train_random(W, 100)
print('Training Model')
print(som.winner(W[0]))
print(som.winner(W[100]))
print(som.winner(W[456]))

print('Training Complete')

##winner_coordinates = np.array([som.winner(x) for x in W])
##print(winner_coordinates[:10])
##
# print(winner_coordinates[-10:])

no_features = 1000

tfidf_vectroiser = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=no_features,
                                   stop_words='english')
tfidf = tfidf_vectroiser.fit_transform(all_lines)
tfidf_feature_names = tfidf_vectroiser.get_feature_names()

som = MiniSom(2, 4, no_features)
D = tfidf.todense().tolist();
som.pca_weights_init(D)
som.train_batch(D, 40000)


top_keywords = 15

weights = som.get_weights()
cnt = 1
for i in range(2):
    for j in range(4):
        keywords_idx = np.argsort(weights[i,j,:])[-top_keywords:]
        keywords = ' '.join([tfidf_feature_names[k] for k in keywords_idx])
        print('Topic', cnt, ':', keywords)
        cnt += 1


