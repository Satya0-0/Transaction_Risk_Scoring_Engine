from src.util.data_loader import get_sample_data
from src.util.data_loader import merged_columns_all
from src.util.model_save import load_pipe
import pandas as pd
import numpy as np
import random
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from src.util.buckets_validation import calculate_EV, pooling
from config import get_config
from src.util.random_test_record import random_test_record

X_sample = get_sample_data()
X_sample_columns = merged_columns_all(X_sample)

pipe = load_pipe("lgbm_pipeline_v1.joblib")

def main_2(idx: int) -> dict:
    """
    Returns the predicted risk score, expected value, and pool for a given sample record.
    """

    sample_txn = X_sample.iloc[idx].to_frame().T
    sample_txn.reset_index(drop=True, inplace=True)
    risk_score = pipe.predict_proba(sample_txn)[:,1]
    risk_score_df = pd.DataFrame(risk_score, columns=['RiskScore'])

    sample_txn_pred = calculate_EV(sample_txn, risk_score_df)
    
    result = {
            'Transaction Risk Score': round(risk_score.item(), 2), 
            'Expected Value': round(sample_txn_pred['EV'].item(), 2),
            'Pool': sample_txn_pred['Bucket'].item()
    }

    return result