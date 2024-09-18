import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data = data.drop_duplicates()

    for column in data.columns:
        if data[column].dtype == 'object':  # Categorical
            data[column] = data[column].fillna(data[column].mode()[0])
        else:  # Numerical
            data[column] = data[column].fillna(data[column].median())

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    data.drop(['id'], axis=1, inplace=True)

    return data

def split_data(data):
    X = data.drop('stroke', axis=1).values
    y = data['stroke'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    return X_train, X_test, y_train, y_test
