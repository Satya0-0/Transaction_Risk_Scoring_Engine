from config import get_config
from src.util.reproducibility import set_seed
from src.util.data_split import train_validation_split
from src.util.feature_engineering import ExtractMonth, SelectFeatures50, TypeConverter
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from src.util.buckets_validation import calculate_EV, pooling, validation_metrics
from typing import Tuple, Union
from src.util.model_save import save_pipe

set_seed()

# Create train and validation sets
X_train, X_val, y_train, y_val = train_validation_split(get_config("split_point"))


def train_and_predict() -> Tuple[pd.DataFrame, Pipeline]:
    """
    Trains the LightGBM model on the training data and predicts the probabilities of fraud on the validation set.
    Returns:
        A tuple containing a DataFrame with the predicted probabilities of fraud for the validation set and the trained model object.
    """
    params = {
        'objective': get_config("lgb_params.objective"), 
        'metric': get_config("lgb_params.metric"),
        'boosting_type': get_config("lgb_params.boosting_type"),
        'min_data_in_leaf': get_config("lgb_params.min_data_in_leaf"),
        'is_unbalance': get_config("lgb_params.is_unbalance"),
        'verbosity': get_config("lgb_params.verbosity"),
        'n_estimators': get_config("lgb_params.n_estimators")
    }
        
    # Create a Pipeline() object for Data Preprocesssing, Feature engineering/ selection
    pipe = Pipeline(steps=[
                    ('get_month', ExtractMonth()),
                    ('select_top50',SelectFeatures50()),
                    ('type_converter',TypeConverter()),
                    ('lgbm_model', lgb.LGBMClassifier(**params))
                    ])
    
    pipe.fit(X_train, y_train)

    fraud_probabilites = pipe.predict_proba(X_val)[:,1]

    y_val_pred = pd.DataFrame(fraud_probabilites, columns =["RiskScore"])
    
    return y_val_pred, pipe


def main():
    """
    Final function to execute the training and validation process. It trains the model, predicts the probabilities of fraud on the validation set, calculates the expected value and buckets for the validation set, and finally calculates the validation metrics.
    """    
    y_pred, pipe = train_and_predict()
    X_val_pred = calculate_EV(X_val, y_pred)
    p0_bucket, p1_bucket, p2_bucket = pooling(X_val_pred)
    validation_metrics_df = validation_metrics(X_val_pred, y_val)
    print(validation_metrics_df)
    pipe_save_name = "lgbm_pipeline_v1.joblib"
    save_pipe(validation_metrics_df, pipe, pipe_save_name)

if __name__ == "__main__":
    set_seed()
    main()