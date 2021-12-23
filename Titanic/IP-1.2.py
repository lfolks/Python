# -*- coding: utf-8 -*-
"""
@author: Lloyd Folks
@date: 08/11/21
@course: Machine Learning (CS379-2103B-01)
@desc: This is a python program to predict the accuracy of the prediction using K Nearest Neighbor
"""

import numpy as np
import sklearn as sk
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neighbors

# Load the data from Excel into a dataframe named df
df = pd.read_excel('CS379T-Week-1-IP.xls')

# Drop columns from the dataset that I believe do not contribute to the survivablitiy of the passenger
df = df.drop(columns=['name', 'ticket', 'body', 'cabin', 'home.dest', 'embarked'])

# Null values are filled with 0
df = df.fillna(0)

# Used to convert values to integers except for float values
def handle_non_numerical_data(df):
    
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int,df[column]))

    return df

# Convert the dataframe values to integers except for float values
df = handle_non_numerical_data(df)

print('---------------==============================---------------')
print('----------------------Dataframe START-----------------------')
print(df)
print('-----------------------Dataframe END------------------------')
print('---------------==============================---------------')

# Prepare the data for K Nearest Neighbor
X = np.array(df.drop(['survived'], 1))
y = np.array(df['survived'])

# Split the data between X train, X test, y train, and y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Run prediction and then output the accurancy
clf = sk.neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print('-------------------Accuracy of Perdiction-------------------')
print(accuracy)
print('---------------==============================---------------')