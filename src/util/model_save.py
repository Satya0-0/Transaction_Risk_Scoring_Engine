import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Union
from config import get_config
import os


def save_pipe(metrics_df: pd.DataFrame, pipe_obj: Pipeline, pipe_save_name: str) -> None:
    
    p0_data = metrics_df[metrics_df['Pool']=='P0']
    pipe_save_path = os.path.join(get_config("model_path"), pipe_save_name)

    if p0_data.empty:
        raise ValueError("There's no data for P0 bucket. Pipe NOT saved!")
    # Debugging prints to check the values of Precision@1000 and Lift for P0 bucket
    print(p0_data['Precision@1000'].iloc[0])
    print(p0_data['Lift'].iloc[0])

    if(p0_data['Precision@1000'].iloc[0] > 0.8 and p0_data['Lift'].iloc[0] > 20):
        joblib.dump(pipe_obj, pipe_save_path)
        print("The pipe's Precision@1000 > 0.8 and Lift is > 20.\nThe pipe is saved!")

    else:
        print("The pipe's Precision@1000 and Lift are not up to mark.\nThe pipe is NOT saved!")
     
    return None


def load_pipe(pipe_save_name: str) -> Pipeline:
    pipe_save_path = os.path.join(get_config("model_path"), pipe_save_name)
    return joblib.load(pipe_save_path)