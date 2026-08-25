import pandas as pd
from src.utils import clean, feature_engineer, zscore_flags


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
