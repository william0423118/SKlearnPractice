from sklearn.datasets import load_boston
import pandas as pd
from sklearn import linear_model
reg = linear_model.LinearRegression()
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print (df.head())
reg.fit(df.iloc[:,:-1], df.target)
coeff = pd.Series(reg.coef_, index = dataset.feature_names)
print('Coefficients: \n', coeff.sort_values())


