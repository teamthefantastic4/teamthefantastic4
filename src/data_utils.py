import os
from typing import Tuple

import gdown
import pandas as pd

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
    test = pd.read_csv(config.DATASET_TEST)

    return train, test


def split_data(
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

    X_train, y_train, X_test, y_test = None, None, None, None

    # Remove the 'target' column for train
    X_train = train.drop("TARGET", axis=1)

    # Assign the 'target' column to y_train
    y_train = train["TARGET"]

    # Remove the 'target' column for test
    X_test = test.drop("TARGET", axis=1)

    # Assign the 'target' column to y_test
    y_test = test["TARGET"]
    
    return X_train, y_train, X_test, y_test


