#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing needed packages

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[4]:


#Training set
# Path of the file to read. We changed the directory structure to simplify submitting to a competition
path = './train.csv'
home_data = pd.read_csv(path)  # reading the file with panda
y = home_data.SalePrice  # we want to predict the sale prices, so we set y as the sale price
# Create X
# The features of the data that we want to take into account
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]


# In[17]:


# path to file you will use for predictions
test_data_path = './test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features, above!
test_X = test_data[features]


# In[21]:


# To improve accuracy, create a Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X,y)


# In[24]:


# Predicting the sale prices for the test set
est_preds = rf_model_on_full_data.predict(test_X)
print(est_preds)

