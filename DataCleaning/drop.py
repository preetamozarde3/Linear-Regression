import pandas as pd

dataset = pd.read_csv('./Melbourne_housing_dataset.csv')
export_csv = pd.DataFrame(dataset).dropna(axis='columns').to_csv('DataCleaning/drop2.csv', index = None, header = True)

