import os
from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split

from src import config

def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Returns:
        train : pd.DataFrame
            Training dataset

        test : pd.DataFrame
            Test dataset
    """

    train = pd.read_csv(config.DATASET_TRAIN)
    test = pd.read_csv(config.DATASET_TRAIN)

    return train, test

def get_feature_target(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Args:
        train : pd.DataFrame
            Training dataset.
        test : pd.DataFrame
            Test dataset.

    Returns:
        X_train : pd.Series
            List reviews for train

        y_train : pd.Series
            List labels for train

        X_test : pd.Series
            List reviews for test

        y_test : pd.Series
            List labels for test
    """
    # TODO
    
    train["hosp1y"] = train["hosp1y"].fillna(0)

    X_train, y_train, X_test, y_test = None, None, None, None

    # Remove the 'target' column for train
    X_train = train.drop("hosply", axis=1)

    # Assign the 'target' column to y_train
    y_train = train["hosply"]

    # Remove the 'target' column for test
    X_test = test.drop("hosply", axis=1)

    # Assign the 'target' column to y_test
    y_test = test["hosply"]
    
    return X_train, y_train, X_test, y_test

def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training dataset in two new sets used for train and validation.

    Arguments:
        X_train : pd.DataFrame
            Original training features
        y_train: pd.Series
            Original training labels/target

    Returns:
        X_train : pd.DataFrame
            Training features
        X_val : pd.DataFrame
            Validation features
        y_train : pd.Series
            Training target
        y_val : pd.Series
            Validation target
    """
    # Features -> X

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_val, y_train, y_val

