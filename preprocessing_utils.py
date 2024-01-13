# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:08:33 2023

@author: saksh
"""

# preprocessing_utils.py
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

def preprocessing(data, transformer):
    transfromed_data = transformer.transform(data)
    transformed_df = pd.DataFrame(transfromed_data, columns=transformer.get_feature_names_out())

    #  One-hot encoding removed an index. Let's put it back:
    transformed_df.index = data.index
    # Joining tables
    data = pd.concat([data, transformed_df], axis=1)

    # Dropping old categorical columns
    data.drop(['gender'], axis=1, inplace=True)
    # Rename the one-hot encoded columns to 'Gender_Male' and 'Gender_Female'
    data.rename(columns={'onehotencoder__gender_Female': 'Gender_Female', 'onehotencoder__gender_Male': 'Gender_Male'}, inplace=True)
    # input_df = pd.DataFrame(input_df, columns=feature_names)
    return data