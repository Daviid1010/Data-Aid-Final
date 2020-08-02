import os
import re
import gensim
import numpy as np
import pandas as pd
# nltk.download('wordnet')
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer

print(os.getcwd())

data = pd.read_csv('texts.csv', header=None)
print(data.keys())
data = data[1]

documents = data

documents = documents.map(lambda x: re.sub('\d+|-|\.|\(|\)\:\#', ' ', x))
documents = documents.map(lambda x: x.lower())
documents = documents.map(lambda x: re.sub('tax|cash|total|subtotal|thank|you|card|cashier|order|gratuity|tip|change|visa|date|phone',' ', x))

print(len(documents))
print(documents[:5])

np.random.seed(2020)
stemmer = SnowballStemmer('english')


def lemmatze_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatze_stemming(token))
    return result


doc_sample = documents[20]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents.fillna('').astype(str).map(preprocess)
print(processed_docs[:10])

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
"""

###### Make Bigram and Trigrams models
bigram = gensim.models.Phrases(documents, min_count=5, threshold=100) ## higher theshold fewer phrases
trigram = gensim.models.Phrases(bigram[documents], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return  [trigram_mod[bigram_mod[doc]] for doc in texts]
"""

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[20])

bow_doc_4310 = bow_corpus[20]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
                                                     dictionary[bow_doc_4310[i][0]],
                                                     bow_doc_4310[i][1]))

#### TF-IDF

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

# #lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=10, workers=2,
# per_word_topics=True, chunksize=100) #model = gensim.models.ldamodel(bow_corpus)

ldamodel = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=8, id2word=dictionary, passes=100)
print(ldamodel.print_topics(num_topics=3, num_words=3))
for i in ldamodel.show_topics():
    print(i[0], i[1])

pprint(ldamodel.print_topics())
doc_lda = ldamodel[bow_corpus]


ldamodel_tfidf = gensim.models.LdaModel(corpus_tfidf, num_topics=3, id2word=dictionary, passes=100)
print(ldamodel_tfidf.print_topics(num_topics=3, num_words=3))
for i in ldamodel_tfidf.show_topics():
    print(i[0], i[1])


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

countVectoriser = CountVectorizer(stop_words='english')

count_data = countVectoriser.fit_transform(documents.fillna('').astype(str))


def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


plot_10_most_common_words(count_data, countVectoriser)

##panel = pyLDAvis.sklearn.prepare(ldamodel_tfidf, count_data, countVectoriser, mds='tsne')
##pyLDAvis.save_html(panel, 'LDA_Visualization.html')

pprint(ldamodel_tfidf.print_topics())
doc_lda = ldamodel_tfidf[corpus_tfidf]

