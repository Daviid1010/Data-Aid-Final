import pandas as pd

data = pd.read_csv('lda_tuning_results.csv', encoding='utf-8')

#print(data.head(n=100))

print(data[data.Coherence == data.Coherence.max()])