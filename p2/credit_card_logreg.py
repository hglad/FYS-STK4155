import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Read file and create dataframe
df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values={})
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Create matrix X of explanatory variables (23 features)
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
# target variable: if customer defaults or not
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

print (df.head())
ncols = X.shape[1]

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
).fit_transform(X)

print (X)
