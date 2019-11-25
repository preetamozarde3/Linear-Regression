import pandas as pd

dataset = pd.read_csv('./Melbourne_housing_dataset.csv')
export_csv = pd.DataFrame(dataset).interpolate(limit_direction='both').to_csv('DataCleaning/LiClean.csv', index = None, header = True)
