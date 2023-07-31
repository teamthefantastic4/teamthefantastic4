from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarray
        val : np.ndarrary
        test : np.ndarray
    """
 
    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()
 
    # Drop columns categorical for the DataFrame
    categorical_cols = working_train_df.select_dtypes(include=['object']).columns

    working_train_df = working_train_df.drop(categorical_cols, axis=1)
    working_test_df = working_test_df.drop(categorical_cols, axis=1)
    working_val_df = working_val_df.drop(categorical_cols, axis=1)

    # Impute values for all columns with missing data or, just all the columns.
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean.fit(working_train_df)

    t_train_df = imp_mean.transform(working_train_df)
    t_val_df = imp_mean.transform(working_val_df)
    t_test_df = imp_mean.transform(working_test_df)

    # Feature scaling with Standar scaler. Apply this to all the columns.

    scaler = StandardScaler()
    scaler = scaler.fit(t_train_df)

    t_train_df = scaler.transform(t_train_df)
    t_val_df = scaler.transform(t_val_df)
    t_test_df = scaler.transform(t_test_df)

    return t_train_df, t_val_df, t_test_df
