'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module performs the data pre-processing.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reading data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Define features and target variable
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# For imputing missing values using KNN
imputer = KNNImputer()

# Separate numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# Impute missing values for numeric columns using KNNImputer
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Impute missing values for non-numeric columns with the mode
for column in non_numeric_cols:
    X_train[column].fillna(X_train[column].mode()[0], inplace=True)
    X_val[column].fillna(X_val[column].mode()[0], inplace=True)
    test[column].fillna(test[column].mode()[0], inplace=True)

# FIXME: This approach is not correct in later versions of Scikit-Learn...
# OHE returns a sparse matrix and, on top pf that, we should apply it
# to categoricals only and proably using a ColumnTransformer...
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

X_train = ohe.fit_transform(X_train)
X_val = ohe.transform(X_val)
test = ohe.transform(test)
