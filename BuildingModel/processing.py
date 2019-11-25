import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns 

# matplotlib inline
correlation_dataset = pd.read_csv('DataCleaning/./LiClean.csv')
correlation_matrix = correlation_dataset.corr().round(2)
correlation_plot = sns.heatmap(data=correlation_matrix, annot=True).get_figure()
correlation_plot.savefig('BuildingModel/correlation_matrix_Li.png')
