from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids



data = pd.read_csv('LineItemData.csv', encoding='utf-8')

print(data.head())

data = data['Description'].dropna()

print(data.head())

vectoriser = TfidfVectorizer(stop_words='english')

X = vectoriser.fit_transform(data.values.astype('U'))

true_k = 8
kmodel = KMedoids(n_clusters=true_k, random_state=0, init='k-medoids++').fit(X)

print(kmodel.cluster_centers_)
print('Top terms per cluster:')
order_centroids = kmodel.cluster_centers_[:, ::-1]
terms = vectoriser.get_feature_names()
for i in range(true_k):
    print('Cluster %d' % i),
    for ind in order_centroids[i, :10]:
        print(' %s ' % terms[ind]),
    print


