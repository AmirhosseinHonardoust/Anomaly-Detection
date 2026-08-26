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

    The baseline for a row is computed from that customer's *prior*
    transactions only (via ``shift(1)`` before rolling). Including the
    current row in its own mean/std would let a huge transaction inflate
    its own baseline and dampen its own Z-score. A customer's first
    transaction has no prior history, so its baseline mean falls back to
    its own amount (Z-score 0) and std falls back to 0 rather than NaN.
    """
    df = df.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = df.sort_values(["customer_id", "date"])
    grouped = df.groupby("customer_id")["amount"]
    prior_roll_mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    prior_roll_std = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())
    df["roll_mean_7"] = prior_roll_mean.fillna(df["amount"])
    df["roll_std_7"] = prior_roll_std.fillna(0.0)
    df["zscore_7"] = (df["amount"] - df["roll_mean_7"]) / df["roll_std_7"].replace(0, 1e-9)
    return df


def zscore_flags(df: pd.DataFrame, th: float = 3.5) -> pd.Series:
    """Flag rows whose rolling Z-score exceeds ``th`` (abs) or whose amount is <= 0."""
    return (df["zscore_7"].abs() >= th) | (df["amount"] <= 0)
