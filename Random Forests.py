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

# First we want to see the performance of decision trees.
# As we want to test the performance, We will use the training data which we have the real values for every item.
# We will use 5% of the training data as the test, so when we do the prediction, we can compare the outcome.

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = DecisionTreeRegressor(random_state=1)
# Fit Model
model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


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

