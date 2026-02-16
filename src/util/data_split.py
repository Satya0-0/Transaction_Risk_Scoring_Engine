import pandas as pd
from config import get_config
from src.util.data_loader import get_training_data
from typing import Tuple
import numpy as np


def define_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the original DF into X, and y datasets assuming Traget variable is named "isFraud"
    Args:
        Input DataFrame
    Returns:
        X, and y datasets containing independent and dependent features respectively
    """
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    return X,y


def get_X_y() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the training data, merges the transaction and identity data, and splits it into X and y datasets.
    Returns:
        X, and y datasets containing independent and dependent features respectively
    """
    df = get_training_data()
    X, y = define_X_y(df)

    return X,y


def get_y_metrics() -> Tuple[int, float]:
    """
    Loads the training data, merges the transaction and identity data, and returns the value counts of the target variable.
    Returns:
        A Series containing the value counts of the target variable "isFraud"
    """
    df = get_training_data()

    total_frauds = df['isFraud'].sum().item()
    overall_fraud_rate = round(df["isFraud"].sum()/df.shape[0],3).item()

    return total_frauds, overall_fraud_rate


def train_validation_split(split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the X,and y sets into training and validation sets in 80:20 ratio based on Transaction DateTime.
    The function assumes the data is already sorted according to Transaction DateTime
    
    Retruns:
        X_train, X_val: training and validation sets for X
        y_train, y_val: training and validation sets for y
    """
    
    X, y = get_X_y()

    # Calculate the point at which data needs to be split
    total_training_rows = int(len(X))
    split_point = int(split_ratio*total_training_rows)

    # Splitting based on total length
    X_train = X.iloc[0:split_point,]
    X_val = X.iloc[split_point:total_training_rows,]

    y_train = y.iloc[0:split_point,]
    y_val = y.iloc[split_point:total_training_rows,]
    
    return X_train, X_val, y_train, y_val


def get_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    A wrapper function to get the train and validation sets for X and y
    """
    split_point = get_config("split_point")
    return train_validation_split(split_point)