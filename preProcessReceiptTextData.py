import os
import re
import pandas as pd
from collections import Counter

import tqdm

receipts = pd.read_csv('LineItemData.csv')
# print(receipts.head())
# print(type(receipts))
# print(receipts.keys())
print(receipts.keys())
receipts['Description'] = receipts['Description'].astype(str)
receipts['text_processed'] = receipts['Description'].map(lambda x: re.sub('[,\.!?^\d+\s|\s\d+\s|\s\d+$\:\/\-]', ' ', x))
receipts['text_processed'] = receipts['text_processed'].map(lambda x: x.lower())
receipts['text_processed'] = receipts['text_processed'].map(
    lambda x: re.sub('total|tax|pm|#|subtotal|table|server|%|you|order|thank|due|check|pay|change|tip|receipt| cashier',
                     ' ', x))

print(receipts.head())

import gensim
from gensim.utils import simple_preprocess


def sent_to_words(sentences):
    for sentence in sentences:
        yield (simple_preprocess(sentence, deacc=True))


data = receipts.text_processed.values.tolist()
data_words = list(sent_to_words(data))
print(data_words[:1])

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')
stop_words.extend(
    ['from', 'subject', 're', 'edu', 'use', 'gratuity', 'thank', 'you', 'server', 'table', 'total', 'subtotal',
     'service', 'due'])


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def stemming(texts):
    texts_out = []
    for sent in texts:
        texts_out.append(stemmer.stem(sent))
    return texts_out


data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# data_stemmed = stemming(data_words)

# counts = Counter(data_lemmatized)

# print(Counter(" ".join(receipts['text_processed']).split()).most_common(25))

print(data_lemmatized[:1])

import gensim.corpora as corpora

id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

# print(corpus[:1])

lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   random_state=100,
                                   chunksize=100,
                                   passes=10,
                                   per_word_topics=True,
                                   num_topics=8)

from pprint import pprint

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)


# hyper param tuning
def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=3,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       alpha=a,
                                       eta=b,
                                       per_word_topics=True
                                       )
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')

    return coherence_model_lda.get_coherence()


import numpy as np

gird = {}
gird['Validation_Set'] = {}

min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

num_of_docs = len(corpus)
corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
    # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
    gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
    corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }

if 1 == 1:
    pbar = tqdm.tqdm(total=1000)

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()
