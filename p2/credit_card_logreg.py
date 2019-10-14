import numpy as np
import pandas as pd

# Read file and create dataframe
df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values={})
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Create matrix X of explanatory variables (23 features)
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
# target variable: if customer defaults or not
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

print (df.head())
ncols = X.shape[1]

onehot = np.zeros(ncols)
