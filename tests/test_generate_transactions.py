import subprocess
import sys
from pathlib import Path

import pandas as pd

from data.generate_transactions import generate

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "data" / "generate_transactions.py"


def test_generate_includes_is_anomaly_ground_truth_column():
    df = generate(start="2023-01-01", end="2023-02-01", seed=3, n_customers=20)
    assert "is_anomaly" in df.columns
    assert df["is_anomaly"].dtype == bool
    # The generator always injects at least some anomalies over a month.
    assert df["is_anomaly"].sum() > 0
    assert (~df["is_anomaly"]).sum() > 0


def test_same_seed_is_fully_deterministic():
    # Regression test for a bug where one anomaly branch used the global,
    # unseeded np.random instead of the local seeded rng, breaking
    # reproducibility for a given --seed.
    df1 = generate(start="2023-01-01", end="2023-06-01", seed=7, n_customers=50)
    df2 = generate(start="2023-01-01", end="2023-06-01", seed=7, n_customers=50)
    pd_testing_equal = df1.equals(df2)
    assert pd_testing_equal


def test_different_seeds_produce_different_data():
    df1 = generate(start="2023-01-01", end="2023-03-01", seed=1, n_customers=20)
    df2 = generate(start="2023-01-01", end="2023-03-01", seed=2, n_customers=20)
    assert not df1.equals(df2)


def test_cli_omits_labels_by_default(tmp_path):
    out_path = tmp_path / "tx.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-10",
            "--seed",
            "1",
            "--n-customers",
            "10",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out_path)
    assert "is_anomaly" not in df.columns


def test_cli_include_labels_keeps_ground_truth_column(tmp_path):
    out_path = tmp_path / "tx_labeled.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--start",
            "2023-01-01",
            "--end",
            "2023-01-10",
            "--seed",
            "1",
            "--n-customers",
            "10",
            "--out",
            str(out_path),
            "--include-labels",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out_path)
    assert "is_anomaly" in df.columns
    assert df["is_anomaly"].dtype == bool
