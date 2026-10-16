# This file was auto-generated from your Colab Notebook

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

df['Vehicle_Style'].str.lower()
df.groupby('Transmission Type').agg({'MSRP': 'mean'}).round(2)

import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
display(df.head())

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

import pandas as pd
import numpy as np
import os

# Define the URL to load the data directly if the local file does not exist
data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')
else:
    print("data.csv not found locally. Loading directly from URL...")
    df = pd.read_csv(data_url)
    # Optional: Save it locally so it exists next time
    df.to_csv('data.csv', index=False)

df.columns = df.columns.str.lower().str.replace(' ', '_')

# 1. Check for missing variables across all columns
print("Missing values per column:")
print(df.isnull().sum())

# 2. Identify all categorical (string) columns
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

# 3. Clean string values inside categorical columns to keep them consistent
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# 4. Handle missing values: fill numeric missing values with 0
for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna('unknown')

print("\nMissing values after cleaning:")
print(df.isnull().sum())

import pandas as pd
import numpy as np

# Define target column
target_col = 'msrp'

# Select numerical and key categorical features for training
base_features = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
categorical_features = ['make', 'transmission_type', 'vehicle_size']

# Perform one-hot encoding on the chosen categorical columns
df_encoded = pd.get_dummies(df[base_features + categorical_features], columns=categorical_features, drop_first=True)

X = df_encoded.values.astype(np.float64)
y = np.log1p(df[target_col].values)

print(f"Features shape X: {X.shape}")
print(f"Target shape y: {y.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Split data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize and fit the Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 3. Predict on the 30% test set
y_pred = model.predict(X_test)

# 4. Evaluate the model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.4f}")
print(f"R2 Score on Test Set: {r2:.4f}")

import matplotlib.pyplot as plt

# Inverse transform the log-transformed predictions/actuals back to original USD scale
y_test_usd = np.expm1(y_test)
y_pred_usd = np.expm1(y_pred)

# Select a subset of samples (e.g., the first 80) for a clean visual comparison
subset_size = 80
plt.figure(figsize=(15, 6))

# Plot actual values
plt.plot(y_test_usd[:subset_size], label='Actual MSRP', color='blue', marker='o', alpha=0.7, linewidth=1.5)
# Plot predicted values
plt.plot(y_pred_usd[:subset_size], label='Predicted MSRP', color='orange', linestyle='--', marker='x', alpha=0.8, linewidth=1.5)

plt.title(f'Actual vs. Predicted MSRP Comparison (First {subset_size} Test Samples)', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Vehicle Price in USD ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Ensure global X and y are defined from df if missing
globals_dict = globals()
if 'X' not in globals_dict or 'y' not in globals_dict:
    import pandas as pd
    target_col = 'msrp'
    base_features = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    categorical_features = ['make', 'transmission_type', 'vehicle_size']
    df_encoded = pd.get_dummies(df[base_features + categorical_features], columns=categorical_features, drop_first=True)
    X = df_encoded.values.astype(np.float64)
    y = np.log1p(df[target_col].values)
else:
    X = globals_dict['X']
    y = globals_dict['y']

# Safely define the USD variables
try:
    y_test_usd = np.expm1(y_test)
    y_pred_usd = np.expm1(y_pred)
except NameError:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    
    # Split, train, and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_usd = np.expm1(y_test)
    y_pred_usd = np.expm1(y_pred)

# Calculate absolute price difference in USD
price_diff = np.abs(y_test_usd - y_pred_usd)

# Select the same subset of samples for consistency
subset_size = 80

plt.figure(figsize=(15, 5))
plt.plot(price_diff[:subset_size], color='red', marker='o', linestyle='-', alpha=0.8, label='Absolute Error ($)')
plt.fill_between(range(subset_size), price_diff[:subset_size], color='red', alpha=0.15)

plt.title(f'Absolute Prediction Error per Sample (First {subset_size} Test Samples)', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Absolute Error in USD ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

#linear regression
df_train.dtypes

df_train.columns

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
df_train[base]

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
df_train[base].values

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
x = df_train[base].values

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
df_train[base].isnull().sum()

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
df_train[base].fillna(0)

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
x_train = df_train[base].fillna(0).values

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
x_train = df_train[base].fillna(0).values
train_linear_regression(x_train, y_train)

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

df['Vehicle_Style'].str.lower()
df.groupby('Transmission Type').agg({'MSRP': 'mean'}).round(2)

import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
display(df.head())

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

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
x_train = df_train[base].fillna(0).values
train_linear_regression(x_train, y_train)

base = ['engine_hp','engine_cylinders','highway_mpg','city_mpg','popularity']
X_train = df_train[base].values
x_train = df_train[base].fillna(0).values
w0, w = train_linear_regression(x_train, y_train)

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
print('w0:', w0)
print('w:', w)

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
w0 + X_train.dot(w)

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

sns.histplot(y_pred, bins=20, color='red')
sns.histplot(y_train, bins=20, color='blue')

from numpy._core.multiarray import ALLOW_THREADS
sns.histplot(y_pred, bins=50, alpha=0.5 color='red')
sns.histplot(y_train, bins=50, alpha=0.5 color='blue')

from numpy._core.multiarray import ALLOW_THREADS
sns.histplot(y_pred, bins=50, alpha=0.5, color='red')
sns.histplot(y_train, bins=50, alpha=0.5, color='blue')

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

df['Vehicle_Style'].str.lower()
df.groupby('Transmission Type').agg({'MSRP': 'mean'}).round(2)

import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
display(df.head())

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

df['Vehicle_Style'].str.lower()
df.groupby('Transmission Type').agg({'MSRP': 'mean'}).round(2)

import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv")
display(df.head())

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

