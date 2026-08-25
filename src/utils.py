"""Cleaning, feature engineering, and Z-score helpers for anomaly detection."""

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate transaction IDs and implausible extreme negative amounts."""
    df = df.drop_duplicates(subset=["tx_id"]).copy()
    df = df[df["amount"] > -1000].copy()
    return df


def feature_engineer(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add calendar features and a per-customer rolling Z-score.

    ``window`` controls the rolling window (in transactions) used for each
    customer's rolling mean/std baseline. Defaults to 7 to match prior
    behavior.
    """
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = df.sort_values(["customer_id", "date"])
    grouped = df.groupby("customer_id")["amount"]
    df["roll_mean_7"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["roll_std_7"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(
        0.0
    )
    df["zscore_7"] = (df["amount"] - df["roll_mean_7"]) / df["roll_std_7"].replace(0, 1e-9)
    return df


def zscore_flags(df: pd.DataFrame, th: float = 3.5) -> pd.Series:
    """Flag rows whose rolling Z-score exceeds ``th`` (abs) or whose amount is <= 0."""
    return (df["zscore_7"].abs() >= th) | (df["amount"] <= 0)
