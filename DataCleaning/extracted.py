import pandas as pd

dataset = pd.read_csv('./Melbourne_housing_dataset.csv')
df = pd.DataFrame(dataset)
export_csv = df[df.isnull().any(axis=1)].interpolate(limit_direction='both').to_csv('DataCleaning/extracted.csv', index = None, header = True)
