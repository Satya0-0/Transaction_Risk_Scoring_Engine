from config import get_config
import pandas as pd
import numpy as np
from typing import Tuple
from src.util.data_split import get_y_metrics


def calculate_EV(X_val: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Expected Value (EV) for each transaction in the validation set based on the predicted probabilities of fraud, transaction amount, and inspection cost. It also creates buckets for the transactions based on their EV and RiskScore.
    
    Args:
        X_val: The validation set containing the transaction data.
        y_pred: The predicted probabilities of fraud for the validation set.
    
    Returns:
        A DataFrame containing the validation set with added columns for EV and Bucket.   
    """
    # Reset the 'index' for the valdiation set
    X_val_new = X_val.reset_index(drop=True)
    
    # Add the 'RiskScore' from LGBM to X_val_new
    X_val_pred = X_val_new.join(y_pred['RiskScore'])
    
    # Calculate the 'Expected Value' based on 'RiskScore', 'Transaction Amount', 'Investigation_Charge' and for Validation Data
    X_val_pred['EV'] = (X_val_pred['RiskScore'] * X_val_pred['TransactionAmt']) - get_config("Inspection_Cost")
    
    # Create 'Buckets' column which segregates tranasctions into different pools
    X_val_pred['Bucket'] = np.where(((X_val_pred['EV'] >= 15*50) | (X_val_pred['RiskScore']>=0.9)), 'P0',
                                   np.where(
                                       ((X_val_pred['EV'] < 15*50) & (X_val_pred['EV'] >= 5*50)) | ((X_val_pred['RiskScore']>=0.75) & (X_val_pred['RiskScore']<0.9))
                                       , 'P1', 'P2')
                                  )

    # Converte 'RiskScore' from 'float16' to 'float32' - to support sorting and other Pandas/Numpy operation
    X_val_pred['RiskScore'] = X_val_pred['RiskScore'].astype('float32')
    X_val_pred['EV'] = X_val_pred['EV'].astype('float32')
    
    return X_val_pred



# Below needs to be executed and the 'X_val_pred' needs to be passed to pooling() and validation_set()
# X_val_pred = calculate_EV(X_Val, y_pred)

def pooling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates the three pools (P0, P1, P2) based on the 'Bucket' column in the provided DataFrame. Each pool is sorted by 'RiskScore' and 'EV' in descending order.
    
    Args:
        df: The DataFrame containing the validation set with added columns for EV and Bucket.
    
    Returns:
        Three DataFrames corresponding to the three pools (P0, P1, P2) sorted by 'RiskScore' and 'EV' in descending order.
    """
    # Creating Pools
    p0_bucket = df[df['Bucket'] =='P0'].sort_values(by=['RiskScore','EV'], ascending=False)
    p1_bucket = df[df['Bucket'] =='P1'].sort_values(by=['RiskScore','EV'], ascending=False)
    p2_bucket = df[df['Bucket'] =='P2'].sort_values(by=['RiskScore','EV'], ascending=False)
    
    return p0_bucket, p1_bucket, p2_bucket


# get_y_metrics: total_frauds, overall_fraud_rate

def validation_metrics(X_val_pred: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    """
    Calculates the validation metrics (Precision@1000, Recall@1000, Lift) for each pool (P0, P1, P2) based on the 'isFraud' column in the provided DataFrame.

    Args:
        p0_bucket: The DataFrame for pool P0.
        p1_bucket: The DataFrame for pool P1.
        p2_bucket: The DataFrame for pool P2.
    Returns:
        A DataFrame containing the calculated metrics for each pool (P0, P1, P2).
    """
    
    # Joining with the 'isFraud' column from the original validation set to calculate the metrics later
    y_val_df = pd.DataFrame(y_val, columns = ["isFraud"]).reset_index(drop=True)
    X_val_pred = X_val_pred.join(y_val_df['isFraud'])

    total_frauds, overall_fraud_rate = get_y_metrics()
    
    p0_bucket, p1_bucket, p2_bucket = pooling(X_val_pred)

    # Step-1: Calculate Precision@1000

    def precision1000(df):
        return (df["isFraud"].iloc[0:1000].sum().item())/1000

    p0_precision1000 = precision1000(p0_bucket)
    p1_precision1000 = precision1000(p1_bucket)
    p2_precision1000 = precision1000(p2_bucket)
    
    # Step-2: Recall@1000
    
    def recall1000(df):
        return round((df["isFraud"].iloc[0:1000].sum().item())/total_frauds,3)
        
    p0_recall1000 = recall1000(p0_bucket)
    p1_recall1000 = recall1000(p1_bucket)
    p2_recall1000 = recall1000(p2_bucket)
    
    # Step-3: Lift for each pool
    
    
    p0_lift = round((precision1000(p0_bucket)/overall_fraud_rate),2)
    p1_lift = round((precision1000(p1_bucket)/overall_fraud_rate),2)
    p2_lift = round((precision1000(p2_bucket)/overall_fraud_rate),2)

    # Create a dataframe for all the metrics calculated above
    metrics = pd.DataFrame({'Pool':['P0','P1','P2'],
                           'Precision@1000':[p0_precision1000, p1_precision1000, p2_precision1000],
                            'Recall@1000':[p0_recall1000, p1_recall1000, p2_recall1000],
                            'Lift':[p0_lift, p1_lift, p2_lift]
                           })
    
    return metrics