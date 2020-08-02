import pandas as pd

data = pd.read_csv('LineItemData.csv')

if isinstance(data, pd.DataFrame):
    print('Test Passed, dataframe loaded')
    print('\n')
    print(data.head(n=10))
else:
    print('Test Failed, dataframe not loaded')

