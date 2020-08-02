import pandas as pd
import numpy as np

data = pd.read_csv('LineItemData.csv', index_col=False)
data.dropna()
# data['InvoiceNo'] = data['InvoiceNo'].astype(str)
data['Description'] = data['Description'].astype(str)
print(data.head())

data = data.drop(data.columns[[0, 3, 4]], axis=1)

print(data.tail())

data = data.groupby('InvoiceNo')['Description'].apply(' '.join)

print(data.tail(n=20))
print(data.head())
print(data.keys())
data.to_csv(r'joinedLineItems.csv', header=True, encoding='utf-8')