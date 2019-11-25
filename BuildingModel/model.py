import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import pandas as pd  
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('DataCleaning/./LiClean.csv')
df = pd.DataFrame(dataset)
X = pd.DataFrame(np.c_[df['Distance'], df['Rooms'], df['Bathroom'], df['Car'], df['YearBuilt']], columns = ['Distance','Rooms','Bathroom','Car','YearBuilt'])
Y = df['Price']
lin_model = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lin_model.fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)
pd.DataFrame(data=Y_train, columns=['Price']).to_csv('BuildingModel/train.csv', index = True, header = True) 
# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
pd.DataFrame(data=Y_test, columns=['Price']).to_csv('BuildingModel/test.csv', index = True, header = True) 
pd.DataFrame(data=y_test_predict, columns=['Price']).to_csv('BuildingModel/predict.csv', index = True, header = True)
def linearRegressionModel(df_part, file_name):
    X = pd.DataFrame(np.c_[df_part['Distance'], df_part['Rooms'], df_part['Bathroom'], df_part['Car'], df_part['YearBuilt']], columns = ['Distance','Rooms','Bathroom','Car','YearBuilt'])
    Y = df_part['Price']
    X = scaler.transform(X)
    Y_predict = lin_model.predict(X)
    pd.DataFrame(data=Y_predict, columns=['Price']).to_csv(file_name, index = True, header = True)
df_1 = df.sample(frac=0.3)
linearRegressionModel(df_1, 'BuildingModel/predict_1.csv')
df_2 = df.sample(frac=0.3)
linearRegressionModel(df_2, 'BuildingModel/predict_2.csv')
df_3 = df.sample(frac=0.3)
linearRegressionModel(df_3, 'BuildingModel/predict_3.csv')
