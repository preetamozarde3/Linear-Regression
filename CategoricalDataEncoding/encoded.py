import pandas as pd
import category_encoders as ce

dataset = pd.read_csv('DataCleaning/./mmClean.csv')
df = pd.DataFrame(dataset)
df_CouncilArea = df['CouncilArea']
df['CouncilArea'] = ce.OrdinalEncoder().fit_transform(df_CouncilArea)
export_csv = pd.get_dummies(df, columns=['Regionname'], drop_first=False).to_csv('CategoricalDataEncoding/encoded.csv', index = None, header = True)


# n_ca = df['CouncilArea'].nunique()
# n_ra = df['Regionname'].nunique()
# print(n_ca, n_ra)
