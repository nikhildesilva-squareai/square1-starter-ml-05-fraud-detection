"""
Data loading, cleaning, and splitting. Keep it leak-free — fit anything that
learns from data on the training split only. Tests define the contract.
"""
from __future__ import annotations
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the transactions CSV into a DataFrame. Treat blank strings as missing."""
    raise NotImplementedError("Implement load_data")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw transaction dirt and return a tidy frame.

    TODO:
      - fill the missing `merchant_category` (e.g. with 'unknown') so there are
        no NaNs left in that column
      - normalise `country_code` to a single case (uppercase) so 'au'/'Au'/'AU'
        collapse to one value
      - (optionally) tame extreme `amount` outliers
      - drop `transaction_id`
    Return a DataFrame with no missing `merchant_category` and upper-cased
    `country_code`.
    """
    raise NotImplementedError("Implement clean")


def stratified_split(df: pd.DataFrame, target: str = "is_fraud",
                     test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Stratified train/test split that preserves the rare-class ratio.

    TODO: split `df` into (train_df, test_df) stratifying on `target` so the
    fraud rate is (almost) identical in both parts. Return (train_df, test_df).
    """
    raise NotImplementedError("Implement stratified_split")
