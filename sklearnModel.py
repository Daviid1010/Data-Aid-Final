import pandas as pd

docs = pd.read_csv('LineItemData.csv', encoding='utf-8')

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
docs['Description'] = docs['Description'].astype(str)
df = cv.fit_transform(docs['Description'])

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=8, random_state=42)
lda.fit(df)
print(type(lda))

for index, topic in enumerate(lda.components_):
    print(f'Top 15 Words for Topic #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')

# high is good
print('Log Liklihood: ', lda.score(df))
## low is good
print('Perpexility: ', lda.perplexity(df))

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

tv = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tv.fit_transform(docs['Description'])

nmf = NMF(n_components=8, random_state=1).fit(tfidf)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


tfidf_fnames = tv.get_feature_names()
print_top_words(nmf, tfidf_fnames, 15)
