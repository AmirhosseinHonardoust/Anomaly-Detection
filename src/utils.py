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


def add_burst_feature(df: pd.DataFrame, day_window: int = 30, clip: float = 50.0) -> pd.DataFrame:
    """Add ``zscore_burst``: a per-category daily-volume Z-score.

    The existing per-row features (``amount``, ``dayofweek``, ``month``,
    ``zscore_7``) are all about a *single transaction*, so none of them can
    see a day where one category suddenly gets far more transactions than
    usual (e.g. a burst of many small grocery purchases). This adds a
    feature for exactly that: for each category, the daily transaction
    count is compared to that category's own rolling count over the prior
    ``day_window`` days (``shift(1)`` before rolling, same prior-only
    design as the per-customer Z-score in ``feature_engineer``, so a burst
    day can't inflate its own baseline). The result is clipped to
    ``+/-clip`` because near-zero historical variance can otherwise produce
    extreme, unstable Z-scores that would distort ``StandardScaler``.

    Measured on a seeded synthetic run (see README "Sample Results"), using
    this as a 4th detector vote roughly doubled recall on ground-truth
    anomalies versus the original 3-detector ensemble, because bursts are
    a volume pattern the per-row detectors structurally cannot see.
    """
    df = df.copy()
    df["_day"] = df["date"].dt.floor("D")
    daily_counts = df.groupby(["_day", "category"]).size().rename("day_count").reset_index()
    daily_counts = daily_counts.sort_values(["category", "_day"])
    grouped = daily_counts.groupby("category")["day_count"]
    prior_mean = grouped.transform(lambda s: s.shift(1).rolling(day_window, min_periods=1).mean())
    prior_std = grouped.transform(lambda s: s.shift(1).rolling(day_window, min_periods=1).std())
    daily_counts["burst_mean"] = prior_mean.fillna(daily_counts["day_count"])
    daily_counts["burst_std"] = prior_std.fillna(0.0)
    z = (daily_counts["day_count"] - daily_counts["burst_mean"]) / daily_counts[
        "burst_std"
    ].replace(0, 1e-9)
    daily_counts["zscore_burst"] = z.clip(-clip, clip)
    df = df.merge(
        daily_counts[["_day", "category", "zscore_burst"]], on=["_day", "category"], how="left"
    )
    return df.drop(columns=["_day"])
