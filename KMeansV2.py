import pandas as pd
pd.set_option('display.max_columns',100)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

data = pd.read_csv('LineItemData.csv')
print(data.head())

data = data['Description'].dropna()
print(data.head())

print('Count Vectors')
vec = CountVectorizer()
matrix = vec.fit_transform(data)
print(pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names()).head())

print('TFID Tokeniser')
vec = TfidfVectorizer(use_idf=False, norm='l1')
matrix = vec.fit_transform(data)
print(pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names()).head())

##S Stemming words

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

print('TextBlob')
vec = CountVectorizer(tokenizer=textblob_tokenizer)
matrix = vec.fit_transform(data.tolist())
print(pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names()).head())

vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                      stop_words='english',
                      norm='l1',
                      use_idf=True)
matrix = vec.fit_transform(data)
print(pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names()).head())

from sklearn.cluster import KMeans
cluster_no = 3
km = KMeans(n_clusters=cluster_no)
km.fit(matrix)

print('Top terms per cluster:')
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vec.get_feature_names()
for i in range(cluster_no):
    top_ten_words = [terms[ind] for ind in order_centroids[i :5]]
    print('Cluster {}: {}'.format(i, ' '.join(top_ten_words)))


result = pd.DataFrame()
result['text'] = data
result['category'] = km.labels_
print(result.head())
