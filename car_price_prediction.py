# This file was auto-generated from your Colab Notebook

rmse(y_train, y_pred)

import pandas as pd
import numpy as np
np.zeros(100)
np.zeros((8,2))

import pandas as pd
import numpy as np
data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'Seden', 2000],
    ['Hyundai','Sonata', 2017, None, 4, 'AUTOMATIC', 'Seden', 27150],
    ['LOTUS', 'ELISE', 2010, 218, 4, 'MANUAL', 'CONVERTIBLE', 54990],
    ['GMC', 'Acadia', 2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340]
]
columns = ['Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Vehicle_Style', 'MSRP']
df = pd.DataFrame(data, columns=columns)
display(df.head())

df[df['Year'] >= 2015]

data = ''

import pandas as pd
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
get_ipython().system('wget $data')

df = pd.read_csv('data.csv')
# Assign the formatted strings back to df.columns
df.columns = df.columns.str.lower().str.replace(' ', '_')
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
df.head()

for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.histplot(df.msrp[df.msrp<100000], bins=50)

price_logs = np.log1p(df.msrp)
sns.histplot(price_logs, bins=50)

#setting up the validation framework
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

df_train = df.iloc[:n_train]
df_val = df.iloc[n_train:n_train+n_val]
df_test = df.iloc[n_train+n_val:]

idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train.head()

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

len(y_train), len(y_val), len(y_test)

#linear regression
df_train.dtypes

df_train.columns

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
x_train = df_train[base].fillna(0).values
w0, w = train_linear_regression(x_train, y_train)
y_pred = w0 + X_train.dot(w)

from numpy._core.multiarray import ALLOW_THREADS
sns.histplot(y_pred, bins=50, alpha=0.5, color='red')
sns.histplot(y_train, bins=50, alpha=0.5, color='blue')

#RMSE
def rmse(y, y_pred):
    error = y_pred - y
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)
    

