import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ExtractMonth(BaseEstimator, TransformerMixin):
    """
    Extracts the month of the transaction from the given Date
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        There's no action to perform in here
        """
        
        # Check if the input is  a DataFrame or not
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        
        # Create a 'trailing underscore attribute' to signal 'fitted' 
        self.is_fitted_ = True
        
        return self
        
        
    def transform(self, X, y=None):
        """
        Extracts the month from the given date
        """
        # Check if the data was already fitted
        check_is_fitted(self)
        
        
        # Check if the input is  a DataFrame or not
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
        
        X_out = X.copy()
        try:
            X_out['TransactionDT'] = X_out['TransactionDT'].astype('int64')
            X_out['TransactionDT'] = pd.to_datetime(X_out['TransactionDT'], unit='s', origin='2025-06-01')
            X_out['TransactionMonth'] = X_out['TransactionDT'].dt.month
            X_out.drop(columns=['TransactionDT'], inplace=True)
            print("Month extracted successfully!")
            return X_out
        except Exception as e:
            raise Exception(f"Error! Encountered '{e}' issue while extracting the month")




class SelectFeatures50(BaseEstimator, TransformerMixin):
    """
    Selects the top-50 most important features to determine fraud as per EDA
    """
    
    def __init__(self):
        """
        Nothing to initialize in the constructor
        """
        self.selected_features = ['V258', 'C1', 'C14', 'R_emaildomain', 'V294', 'C13', 
                     'D2', 'V201', 'card2', 'C12', 'TransactionAmt', 'card1', 'V156', 
                     'P_emaildomain', 'addr1', 'V308', 'card6', 'C11', 'D15', 'V91', 
                     'C8', 'TransactionMonth', 'dist1', 'card3', 'D4', 'C2', 'D10', 
                     'C6', 'V70', 'card5', 'V283', 'V62', 'D3', 'C5', 'D8', 'V87', 'D1', 
                     'V45', 'ProductCD', 'V48', 'V149', 'M5', 'V189', 'V130', 'M6', 
                     'V53', 'C9', 'V82', 'M4', 'V313']
    
    def fit(self, X, y=None):
        """
        Used to fit the data. No action performed for this use-case
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided object is not a DataFrame")
        
        self.is_fitted_ = True
        return self
        
    def transform(self,X, y=None):
        """
        Used to select the top-50 columns from provided DataFrame
        """
        # Create a copy of the dataframe so the orginal remains uneffected
        X_copy = X.copy()
        
        # Check if X has already been fitted to the class object
        check_is_fitted(self)
                     
        # Check whether the selected columns are alredy present in the DataFrame
        provided_features = list(X_copy.columns)
                
        missing_columns = [feature for feature in self.selected_features if feature not in provided_features]
        if missing_columns:
            raise ValueError(f"Following columns are missing from the dataframe: {missing_columns}")
        
        # Actual Transformation
        try:
            print("Selecting top-50 features...")
            return X_copy[self.selected_features]
        except Exception as e:
            raise Exception(f"Error! Encountered an issue while selecting top-50 features.\n'{e}'")



class TypeConverter(BaseEstimator, TransformerMixin):
    """
    Converts the data types of the columns to 'category' for object type columns and 'float' for numerical columns. This is done to optimize the memory usage and speed up the training process.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        All the columns with object data type are stored in a list to be used while transforming the data - this is done to avoid any data leakage from the test set. A 'trailing underscore attribute' is created to avoid any type mismatch issues while transforming the data.
        """
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Converts the data types of the columns to 'category' for object type columns and 'float' for numerical columns. 
        This is done to optimize the memory usage and speed up the training process.
        """
        check_is_fitted(self)

        X = X.copy()
        
        # Enforce 'category' on the exact columns identified during training
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # Enforce 'float' for everything else to keep LightGBM happy
        # Single rows often default to 'object' dtype; this forces them to numeric
        numeric_cols = [c for c in X.columns if c not in self.cat_cols]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
        
        print("\n\n")
        for c in X.columns:
            print(c," : ", type(X[c].iloc[0]))
        print("Data types converted successfully!")
        print("\n\n")
        return X