import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess

papers = pd.read_csv('LineItemData.csv')
papers = papers.drop(columns=['Unnamed: 0','InvoiceNo','UnitPrice', 'Quantity'], axis=1)
papers['paper_text_processed'] = papers['Description'].fillna('').astype(str).map(lambda x: re.sub('[,\.!?]', '', x))
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),
                                             deacc=True))


data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

if isinstance(bigram_mod, gensim.models.phrases.Phraser):
    if isinstance(trigram_mod, gensim.models.phrases.Phraser):
        print('Unit Test Passed: Bigrams and Trigrams successfully created using gensim package')
    else:
        print('Unit Test Failed: Bigrams created successfully created but Trigrams failed')
else:
    print('Unit Test Failed: Bigrams and Trigrams Failed to create')



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


print(data_words[23301])

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_words_nostops[23301])
print(data_words_bigrams[23301])
print(data_lemmatized[23301])




#print(data_lemmatized[:1])
