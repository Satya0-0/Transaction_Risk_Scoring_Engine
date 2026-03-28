from src.util.data_loader import get_testing_data
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



X_test = get_testing_data()
X_test_columns = merged_columns_all(X_test)

pipe = load_pipe("lgbm_pipeline_v1.joblib")

def main(idx: int) -> dict:
    """
    Returns the predicted risk score, expected value, and pool for a given test record.
    """

    test_txn = X_test.iloc[idx].to_frame().T
    test_txn.reset_index(drop=True, inplace=True)
    risk_score = pipe.predict_proba(test_txn)[:,1]
    risk_score_df = pd.DataFrame(risk_score, columns=['RiskScore'])

    test_txn_pred = calculate_EV(test_txn, risk_score_df)
    
    result = {
            'Transaction Risk Score': round(risk_score.item(), 2), 
            'Expected Value': round(test_txn_pred['EV'].item(), 2),
            'Pool': test_txn_pred['Bucket'].item()
    }

    return result