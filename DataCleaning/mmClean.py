import pandas as pd

dataset = pd.read_csv('./Melbourne_housing_dataset.csv')
df = pd.DataFrame(dataset)
#  If datatype of column is numeric use mean to populate data; else use mode
for column in df:
    if(df[column].dtype == 'object'):
        column = df.fillna(df.mode(), inplace=True)
    elif(df[column].dtype == 'int64' or df[column].dtype == 'float64'):
        column = df.fillna(df.mean(), inplace=True)
export_csv = df.to_csv('DataCleaning/mmClean.csv', index = None, header = True)
