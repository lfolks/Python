# -*- coding: utf-8 -*-
"""
@author: Lloyd Folks
@date: 08/11/21
@course: Machine Learning (CS379-2103B-01)
@desc: This is a python program to predict the accuracy of the prediction using KMeans
"""

import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
import pandas as pd

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
                    # creating dict that contains new
                    # id per unique string
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

# Prepare the data for KMeans
X = np.array(df.drop(columns=['survived']))
X = sk.preprocessing.scale(X)
Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

# Set value of the correct variable to 0
correct = 0

# Run prediction and then output the accurancy
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == Y[i]:
        correct += 1

print('-------------------Accuracy of Perdiction-------------------')
print(correct/len(X))
print('---------------==============================---------------')