# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:27:00 2023

@author: saksh
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import statsmodels.api as sm
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import json
from preprocessing_utils import preprocessing

app = FastAPI()


class InputData(BaseModel):
    age: int
    experience: int
    completed_trainings: int
    expired_trainings: int
    total_trainings: int
    gender: str
    
#Loading the saved model
with open('C:\\Users\\saksh\\OneDrive\\Desktop\\Capstone\\Python_Code\\OLS_Model.pkl', 'rb') as OLS_model_file:
    ols_model = pickle.load(OLS_model_file)
    
# Load the preprocessing utilities (including preprocess_data function and transformer)
with open("C:\\Users\\saksh\\OneDrive\\Desktop\\Capstone\\Python_Code\\encoding.pkl", 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)

with open('C:\\Users\\saksh\\OneDrive\\Desktop\\Capstone\\Python_Code\\lr_Model.pkl', 'rb') as lr_model_file:
    lr_model = pickle.load(lr_model_file)
    
## Defining a method to take in parameters an

def preprocess_input_data(input_data: InputData, transformer):
    input_df = input_data.json()
    input_dict = json.loads(input_df)
    
    # Assuming input_dict is already defined
    age = input_dict['age']
    experience = input_dict['experience']
    completed = input_dict['completed_trainings']
    expired = input_dict['expired_trainings']
    total = input_dict['total_trainings']
    gender = input_dict['gender']
    
    # Creating a DataFrame from the data
    data = pd.DataFrame({
        'age': [age],
        'experience': [experience],
        'completed_trainings': [completed],
        'expired_trainings': [expired],
        'total_trainings': [total],
        'gender': [gender]
    })
    
    # Preprocess the data using the preprocess_data function
    processed_data = preprocessing(data, transformer)
    
    return processed_data

@app.post("/predict_ols")
def predict_ols(data: InputData):
    input_df = preprocess_input_data(data, transformer)
    
    # Perform the prediction
    prediction = ols_model.predict(sm.add_constant(input_df, has_constant='add'))
    # Customize the response key
    response_key = "Training completion rate prediction using OLS Model"
    response = {response_key: float(prediction[0])}
    
    return response


@app.post("/predict_lr")
def predict_lr(data: InputData):
    input_df = preprocess_input_data(data, transformer)
    
    # Perform the prediction using the LR model
    prediction = lr_model.predict(input_df)
    
    # Customize the response key
    response_key = "Training completion rate prediction using LR Model"
    response = {response_key: float(prediction[0])}
    
    return response
                                                                                              