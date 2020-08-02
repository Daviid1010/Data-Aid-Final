import pandas as pd
import os

print(os.getcwd())
papers = pd.read_csv('LineItemData.csv')

print(papers.head())
papers = papers.drop(columns=['Unnamed: 0','InvoiceNo','UnitPrice', 'Quantity'], axis=1)

#papers = papers.sample(10)

print(papers.head())

import re

## remove punctuation
papers['paper_text_processed'] = papers['Description'].fillna('').astype(str).map(lambda x: re.sub('[,\.!?]', '', x))
## convert to lower case
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

#print first rows
##print(papers.head())

import gensim
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),
                                             deacc=True))


data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
##print(data_words[:1])


##Phrase Modelling: Bi-grams and Tr-grams

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

## remove stopwords, make bigrams, and lemmatize
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


import  spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:1])

## Data Transformation: Corpus and Dictionary

import gensim.corpora as corpora

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])

###Base Model
### Now I will train the LDA model
## chunksize controls how many documents are processed at a time, increasing chunksize will speed up training
## passes controls how often we train the model on the entire corpus

lda_model = gensim.models.LdaModel(corpus= corpus,
                                       id2word=id2word,
                                       num_topics=8,
                                       random_state=100,
                                       chunksize=1000,
                                       passes=10,
                                       per_word_topics=True)

from pprint import pprint
##Print keywords in the topics
pprint(lda_model.print_topics())
docs_lda = lda_model[corpus]

from gensim.models import CoherenceModel

##Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model,
                                     texts=data_lemmatized,
                                     dictionary= id2word,
                                     coherence='u_mass',
                                     )

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)