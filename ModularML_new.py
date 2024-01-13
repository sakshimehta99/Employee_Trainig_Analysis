#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img height=300 width=300 src="NEU.png" />
# <font color="#000099">
#     <b><h1>ALY6980 Capstone</h1></b>
#     <br><b><h2>Northeastern University</h2></b>
#     <br><b><h3>Dr. Hema Seshadri</h3></b>
# <br><b><h3>Group 1</h3></b>
# <br><b><h3>Predictive Modeling and Analysis - Modular ML</h3></b>
# <br><b><h3></h3></b>
# </font>
# </center>

# ### Libraries Used In this Project 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:.2f}'.format


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# ### Read Data
# Function to read a CSV file and display its head
def read_and_display_csv_head(file_path):
    """     
	Reads a CSV file from the given file path and returns the first few rows of the DataFrame.    
 	Parameters:     file_path (str): The path to the CSV file.     
	Returns:    pandas.DataFrame: The first few rows of the CSV file as a DataFrame.  
    """
    
    df = pd.read_csv(file_path)
    return df


# Using the function to read and display the head of the two uploaded files
incident_df = read_and_display_csv_head('incidentANDtype.csv')
incident_df.head()


# ### Preprocess Data 

# Defining the function again due to the reset of the code execution state
def preprocess_data_for_classification_and_regression(df, target_column, features_to_exclude=[], drop_nulls=True, test_size=0.2, random_state=42):
    """
    Prepares a DataFrame for machine learning by handling nulls, encoding categorical variables, 
    and splitting into training and test sets.     
    Parameters:     df (DataFrame): Input data.     
    target_column (str): Column name of the target variable.    
    features_to_exclude (list, optional): Columns to exclude from the input data.    
    drop_nulls (bool, optional): If True, drops rows with null values.     
    test_size (float, optional): Proportion of data to use for the test set.    
    random_state (int, optional): Seed for random operations.     
    Returns:     Tuple: Training and test sets (X_train, X_test, y_train, y_test).
    """
    # Dropping rows with null values if specified
    if drop_nulls:
        df = df.dropna()

    # Separating the target variable and features
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Drop features that should be excluded (e.g., date columns)
    X = X.drop(columns=features_to_exclude, errors='ignore')

    # Encoding categorical columns (excluding date columns as per features_to_exclude)
    X = pd.get_dummies(X, drop_first=True)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# ### Train Model

def get_model_features(df, target_column):
    """
    Identifies the features used for modeling by excluding the target column.
    
    : param df: Pandas DataFrame, the dataset to process
    : param target_column: str, the name of the target column
    :return: list, the names of the features used for modeling
    """

    # Ensuring the target column is in the dataframe
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")

    # Excluding the target column to get the features
    feature_names = df.drop(target_column, axis=1).columns.tolist()

    return feature_names



def train_model(X_train, y_train, model):
    """
    Trains a machine learning model provided by the user.

    :param X_train: Pandas DataFrame or Numpy array, training feature set
    :param y_train: Pandas Series or Numpy array, training target variable
    :param model: Machine learning model instance (classifier or regressor)
    :return: Trained model
    """

    model.fit(X_train, y_train)
    return model


# ### Predictions




def get_predictions(model, X):
    """
    Generates predictions using a trained model.

    :param model: Trained machine learning model
    :param X: Pandas DataFrame or Numpy array, feature set for which predictions are to be made
    :return: Predictions
    """
    return model.predict(X)


# ### Performance Evaluation 



def evaluate_model_performance_table(y_train, y_test, train_predictions, test_predictions, X_train, X_test, model, model_type='classification'):
    """
    Evaluates the performance of a machine learning model and returns the results in a table format.

    :param y_train: Actual true labels/values for training data
    :param y_test: Actual true labels/values for testing data
    :param train_predictions: Predicted labels/values by the model on training data
    :param test_predictions: Predicted labels/values by the model on testing data
    :param X_train: Training feature set
    :param X_test: Testing feature set
    :param model: Trained machine learning model
    :param model_type: Type of the model ('classification' or 'regression')
    :return: Pandas DataFrame containing relevant evaluation metrics
    """
    if model_type == 'classification':
        # Classification metrics
        accuracy_train = accuracy_score(y_train, train_predictions)
        accuracy_test = accuracy_score(y_test, test_predictions)
        report_train = classification_report(y_train, train_predictions, zero_division=0, output_dict=True)
        report_test = classification_report(y_test, test_predictions, zero_division=0, output_dict=True)

        classification_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy-Train', 'Precision-Train', 'Recall-Train', 'F1-Score-Train',
                       'Accuracy-Test', 'Precision-Test', 'Recall-Test', 'F1-Score-Test'],
            'Value': [
                accuracy_train, report_train['weighted avg']['precision'], report_train['weighted avg']['recall'], report_train['weighted avg']['f1-score'],
                accuracy_test, report_test['weighted avg']['precision'], report_test['weighted avg']['recall'], report_test['weighted avg']['f1-score']
            ]
        })
        return classification_metrics_df

    elif model_type == 'regression':
        # Regression metrics
        regression_metrics_df = pd.DataFrame({
            'Metric': ['RMSE-Train', 'RMSE-Test', 'Score-Train', 'Score-Test', 'MedAE-Train', 'MedAE-Test', 'MeanAE-Train', 'MeanAE-Test'],
            'Value': [
                np.sqrt(mean_squared_error(y_train, train_predictions)),
                np.sqrt(mean_squared_error(y_test, test_predictions)),
                model.score(X_train, y_train),
                model.score(X_test, y_test),
                median_absolute_error(y_train, train_predictions),
                median_absolute_error(y_test, test_predictions),
                mean_absolute_error(y_train, train_predictions),
                mean_absolute_error(y_test, test_predictions)
            ]
        })
        return regression_metrics_df

    else:
        raise ValueError("model_type must be 'classification' or 'regression'")


# #### Classification 

# Preprocessing the data for classification using 'IncidentTypeID' as the target variable

target_column_classification = 'IncidentTypeID'
exclude_features = ['Date_of_Joining', 'IncidentCreationDate','Incident_Number','Employee_ID','Employee_Name']  # Replace with actual date column names
X_train_class, X_test_class, y_train_class, y_test_class = preprocess_data_for_classification_and_regression(
    incident_df, target_column_classification, features_to_exclude=exclude_features)

# Displaying the shapes of the generated datasets
X_train_class.head()

# Example usage for classification with RandomForestClassifier
classification_model = RandomForestClassifier(random_state=42)
trained_classifier = train_model(X_train_class, y_train_class, classification_model)


classification_predictions = get_predictions(trained_classifier, X_test_class)
classification_predictions[:5]

train_class_predictions = trained_classifier.predict(X_train_class)
test_class_predictions = trained_classifier.predict(X_test_class)

# Evaluate the classification model
classification_performance_table = evaluate_model_performance_table(
    y_train_class, y_test_class, 
    train_class_predictions, test_class_predictions, 
    X_train_class, X_test_class,  # Make sure to pass the feature sets
    trained_classifier,  # The trained classification model
    'classification'
)

print("Classification Model Performance:\n", classification_performance_table)


# #### Regression 

# Preprocessing the data for regression using 'Employee_Service_Length_in_Years' as the target variable
target_column_regression = 'Employee_Service_Length_in_Years'
exclude_features = ['Date_of_Joining', 'IncidentCreationDate','Incident_Number','Employee_ID','Employee_Name','Incident_ID','Incident_Age_in_days','IncidentTypeID']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess_data_for_classification_and_regression(
    incident_df, target_column_regression, features_to_exclude=exclude_features)

X_train_reg.head()


# Example usage for regression with LinearRegression
regression_model = LinearRegression()
trained_regressor = train_model(X_train_reg, y_train_reg, regression_model)

regression_predictions = get_predictions(trained_regressor, X_test_reg)
regression_predictions[:5]

# For the regression model
train_reg_predictions = trained_regressor.predict(X_train_reg)
test_reg_predictions = trained_regressor.predict(X_test_reg)


# Evaluate the regression model
regression_performance_table = evaluate_model_performance_table(
    y_train_reg, y_test_reg, 
    train_reg_predictions, test_reg_predictions, 
    X_train_reg, X_test_reg,  # Make sure to pass the feature sets
    trained_regressor,  # The trained regression model
    'regression'
)
print("Regression Model Performance:")
regression_performance_table


# ### Generate Pickle Files


import pickle

def save_model_to_pickle(model, file_path):
    """
    Saves a machine learning model to a pickle file.

    :param model: Trained machine learning model
    :param file_path: Path where the pickle file will be saved
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)



# Save the classification model
save_model_to_pickle(trained_classifier, 'trained_classifier.pkl')

# Save the regression model
save_model_to_pickle(trained_regressor, 'trained_regressor.pkl')



# Save the classification metrics
save_model_to_pickle(classification_performance_table, 'classification_metrics.pkl')

# Save the regression metrics
save_model_to_pickle(regression_performance_table, 'regression_metrics.pkl')







