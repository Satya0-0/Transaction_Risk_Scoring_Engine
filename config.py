import os

# Creating a base directory variable to always refer to the root of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CONFIG = {
            'seed' : 42,
            'data': {
                'train_txn_data': os.path.join(BASE_DIR, 'data', 'raw', 'train_transaction.csv'),
                'train_idnty_data': os.path.join(BASE_DIR, 'data', 'raw', 'train_identity.csv'),
                'test_txn_data': os.path.join(BASE_DIR, 'data', 'raw', 'test_transaction.csv'),
                'test_idnty_data': os.path.join(BASE_DIR, 'data', 'raw', 'test_identity.csv'),
                'sample_data': os.path.join(BASE_DIR, 'sample_data', 'sample_data.csv')
            },
            'split_point': 0.8,
            'lgb_params' : {
                'objective': 'binary',
                'metric': ['auc', 'binary_logloss'],
                'boosting_type': 'gbdt',
                'min_data_in_leaf': 100,
                'is_unbalance': True,
                'n_estimators': 100,
                'verbosity': -1
                },
            'Inspection_Cost': 50,
            'model_path': os.path.join(BASE_DIR, 'models')
         }


def get_config(dot_path:str):
    """
    Docstring for get_config
    
    :param dot_path: Description
    :type dot_path: str
    """

    keys = dot_path.split(".")
    val = CONFIG
	
    for key in keys:
        try:
            val = val[key]
        except KeyError:
            raise KeyError(f"Key {key} not found in config")
        

    return val