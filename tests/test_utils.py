import pandas as pd
from src.utils import add_burst_feature, clean, feature_engineer, zscore_flags


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": ["a", "a", "b", "c", "d"],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "customer_id": [1, 1, 1, 1, 2],
            "category": ["Home", "Home", "Home", "Home", "Toys"],
            "amount": [10.0, 10.0, 12.0, -5000.0, 20.0],
        }
    )


def test_clean_drops_duplicate_tx_id_and_extreme_negatives():
    df = _sample_df()
    out = clean(df)
    # duplicate tx_id "a" collapses to one row, and the -5000 row is dropped
    assert len(out) == 3
    assert out["tx_id"].is_unique
    assert (out["amount"] > -1000).all()


def test_feature_engineer_adds_expected_columns():
    df = clean(_sample_df())
    out = feature_engineer(df, window=7)
    for col in ["dayofweek", "month", "roll_mean_7", "roll_std_7", "zscore_7"]:
        assert col in out.columns
    assert not out["zscore_7"].isna().any()


def test_zscore_flags_catches_zero_and_negative_amounts():
    df = pd.DataFrame(
        {
            "amount": [10.0, 0.0, -5.0, 12.0],
            "zscore_7": [0.1, 0.1, 0.1, 0.1],
        }
    )
    flags = zscore_flags(df, th=3.5)
    assert list(flags) == [False, True, True, False]


def test_zscore_flags_catches_high_zscore():
    df = pd.DataFrame({"amount": [10.0], "zscore_7": [5.0]})
    assert zscore_flags(df, th=3.5).iloc[0]


def test_feature_engineer_excludes_current_row_from_baseline():
    """Regression test: a transaction's own amount must not feed its own
    rolling mean/std, or a spike dampens its own z-score. Four identical
    $10 prior transactions followed by a $100 spike should compute the
    spike's baseline from the four $10 rows only."""
    df = pd.DataFrame(
        {
            "tx_id": ["t1", "t2", "t3", "t4", "t5"],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            ),
            "customer_id": [1, 1, 1, 1, 1],
            "category": ["Home"] * 5,
            "amount": [10.0, 10.0, 10.0, 10.0, 100.0],
        }
    )
    out = feature_engineer(df, window=7)
    spike_row = out.iloc[-1]
    assert spike_row["roll_mean_7"] == 10.0
    assert spike_row["roll_std_7"] == 0.0
    # A std of 0 among identical prior amounts is replaced with a tiny
    # epsilon, so excluding the spike from its own baseline produces a
    # very large z-score rather than a muted one.
    assert spike_row["zscore_7"] > 1000


def _daily_df(day_category_counts: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Build a transactions frame with the given (date, category, count) rows."""
    rows = []
    tx = 0
    for date, category, count in day_category_counts:
        for _ in range(count):
            rows.append({"tx_id": f"t{tx}", "date": date, "category": category, "amount": 10.0})
            tx += 1
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_add_burst_feature_adds_column_without_dropping_rows():
    df = _daily_df([("2023-01-01", "Grocery", 3), ("2023-01-02", "Grocery", 3)])
    out = add_burst_feature(df)
    assert "zscore_burst" in out.columns
    assert len(out) == len(df)
    assert not out["zscore_burst"].isna().any()


def test_add_burst_feature_excludes_current_day_from_its_own_baseline():
    """Regression test mirroring test_feature_engineer_excludes_current_row_from_baseline:
    a burst day's own count must not feed its own mean/std, or the burst
    dampens its own Z-score."""
    normal_days = [(f"2023-01-{d:02d}", "Grocery", 5) for d in range(1, 6)]
    burst_day = [("2023-01-06", "Grocery", 50)]
    df = _daily_df(normal_days + burst_day)
    out = add_burst_feature(df, day_window=30)
    burst_rows = out[out["date"] == "2023-01-06"]
    # Baseline should be ~5 (from the 5 prior days), not inflated by the
    # burst day's own 50, so the burst gets a large positive Z-score.
    assert (burst_rows["zscore_burst"] > 5).all()


def test_add_burst_feature_clips_extreme_values():
    """Near-zero historical variance can otherwise blow up the raw Z-score;
    clipping keeps it from distorting StandardScaler if ever added as a
    model feature."""
    df = _daily_df([("2023-01-01", "Grocery", 5), ("2023-01-02", "Grocery", 500)])
    out = add_burst_feature(df, day_window=30, clip=10.0)
    assert out["zscore_burst"].abs().max() <= 10.0
