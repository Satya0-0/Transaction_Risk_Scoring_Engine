import pandas as pd
from config import get_config


def load_data(path:str) -> pd.DataFrame:
    """Loads the training data from the specified path.
 
    Args:
        path: The file path to the CSV data.
    Returns:
        A pandas DataFrame containing the loaded data.
    """
    
    df = pd.read_csv(path)
    return df


def merge_df(txn_df: pd.DataFrame, idnty_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the transaction and identity dataframes.

    Args:
        txn_df: The dataframe containing transaction records.
        idnty_df: The dataframe containing identity records.

    Returns:
        A single merged dataframe joined on 'TransactionID'.
    """
    merged_df = txn_df.merge(idnty_df, on = "TransactionID", how="left")
    return merged_df 

def merged_columns_all(df: pd.DataFrame) -> list:
    """Returns a list of all column names in the merged dataframe.

    Args:
        df: The merged dataframe containing both transaction and identity data.
    Returns:

        A list of column names in the merged dataframe.
    """
    return df.columns.tolist()


def get_training_data() -> pd.DataFrame:
    """Loads and merges the training transaction and identity data.
    Returns:
        A merged dataframe containing both transaction and identity data.
    """
    txn_path = get_config("data.train_txn_data")
    idnty_path = get_config("data.train_idnty_data")

    txn = load_data(txn_path)
    idnty = load_data(idnty_path)

    final_df = merge_df(txn, idnty)
    
    return final_df


def get_testing_data() -> pd.DataFrame:
    """Loads and merges the testing transaction and identity data.
    Returns:
        A merged dataframe containing both transaction and identity data.
    """
    test_txn_path = get_config("data.test_txn_data")
    test_idnty_path = get_config("data.test_idnty_data")

    txn = load_data(test_txn_path)
    idnty = load_data(test_idnty_path)

    final_df = merge_df(txn, idnty)
    
    return final_df
