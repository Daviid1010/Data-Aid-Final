##https://pythonprogramminglanguage.com/kmeans-text-clustering/
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA




data = pd.read_csv('LineItemData.csv', encoding='utf-8')

print(data.head())

data = data['Description'].dropna()

print(data.head())

vectoriser = TfidfVectorizer(stop_words='english')

X = vectoriser.fit_transform(data.values.astype('U'))
true_k = 3
kmodel = KMeans(n_clusters=true_k, init='k-means++', max_iter=10000, n_init=1)
kmodel.fit(X)

print('Top terms per cluster:')
order_centroids = kmodel.cluster_centers_.argsort()[:, ::-1]
terms = vectoriser.get_feature_names()
for i in range(true_k):
    print('Cluster %d' % i),
    for ind in order_centroids[i, :10]:
        print(' %s ' % terms[ind]),
    print

y_km = kmodel.fit_predict(X)
print(y_km)
print(kmodel.cluster_centers_)


pca = PCA(n_components=2)
scatterplot_points = pca.fit_transform(X.toarray())
colors = np.arange(17)
x_axis = [o[0] for o in scatterplot_points]
y_axis = [o[1] for o in scatterplot_points]
plt.scatter(x_axis, y_axis, c=[colors[d] for d in kmodel.fit_predict(X)])
plt.show()



sse = {}
for k in range(1,20):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=10000, n_init=1).fit(X)
    data['clusters'] = kmeans.labels_
    sse[k] = kmeans.inertia_ #sum of distances of samples

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()




