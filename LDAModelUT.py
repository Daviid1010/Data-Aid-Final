import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = pd.read_csv('LineItemData.csv', encoding='utf-8')


cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
docs['Description'] = docs['Description'].astype(str)
df = cv.fit_transform(docs['Description'])
print(type(cv))
print(type(df))


lda = LatentDirichletAllocation(n_components=8, random_state=42)
lda.fit(df)
print(type(lda))

if isinstance(cv, sklearn.feature_extraction.text.CountVectorizer):
    if isinstance(df, scipy.sparse.csr.csr_matrix):
        if isinstance(lda, sklearn.decomposition.LatentDirichletAllocation):
            print('Unit Test Passed, LDA model created from Matrix and Count Vectoriser data')
        else:
            print('Falied Unit Test: LDA model failed to create')
    else:
        print('Failed Unit Test: Matrix created from Count Vectoriser')
else:
    print('Failed Unit Test: Count Vectoriser failed to create')

