import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.detect_anomalies import load, run_models
from src.utils import clean, feature_engineer

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "detect_anomalies.py"


def _tiny_transactions(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "tx_id": [f"tx{i}" for i in range(n)],
            "date": dates,
            "customer_id": rng.integers(1, 5, size=n),
            "category": rng.choice(["Home", "Toys"], size=n),
            "amount": rng.normal(50, 10, size=n).round(2),
        }
    )
    # inject a couple of obvious outliers
    df.loc[0, "amount"] = 5000.0
    df.loc[1, "amount"] = -10.0
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


def test_pipeline_runs_end_to_end_and_flags_something(tmp_path):
    csv_path = _tiny_transactions(tmp_path)
    df = load(str(csv_path))
    df = clean(df)
    df = feature_engineer(df, window=7)

    iso_labels, iso_score, lof_labels, lof_score, z_flags = run_models(
        df, contamination=0.05, lof_n_neighbors=5
    )

    assert len(iso_labels) == len(df)
    assert len(lof_labels) == len(df)
    assert z_flags.sum() >= 1  # the injected outliers should trip the z-score flag


def test_load_rejects_missing_columns(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"tx_id": ["a"], "amount": [1.0]}).to_csv(bad_path, index=False)
    try:
        load(str(bad_path))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_cli_runs_end_to_end_and_writes_outputs(tmp_path):
    """Exercise the actual CLI entry point (main/argparse), not just the
    underlying functions, so a broken flag or wiring bug in main() is caught.
    """
    csv_path = _tiny_transactions(tmp_path)
    outdir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(csv_path),
            "--outdir",
            str(outdir),
            "--contamination",
            "0.05",
            "--lof-n-neighbors",
            "5",
            "--rolling-window",
            "5",
            "--zscore-threshold",
            "3.0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (outdir / "anomalies.csv").exists()
    assert (outdir / "fig_amount_time.png").exists()
    assert (outdir / "fig_amount_hist.png").exists()

    anomalies = pd.read_csv(outdir / "anomalies.csv")
    assert len(anomalies) >= 1
    assert "severity" in anomalies.columns
    assert "votes" in anomalies.columns


def test_cli_reports_error_for_missing_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(tmp_path / "does_not_exist.csv"),
            "--outdir",
            str(tmp_path / "outputs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_reports_clear_error_when_nothing_survives_cleaning(tmp_path):
    """clean() drops amount <= -1000; an all-extreme-negative input leaves
    an empty DataFrame, which should raise a clear error instead of an
    opaque crash deeper in sklearn."""
    df = pd.DataFrame(
        {
            "tx_id": ["a", "b"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "customer_id": [1, 2],
            "category": ["Home", "Toys"],
            "amount": [-5000.0, -6000.0],
        }
    )
    csv_path = tmp_path / "empty_after_clean.csv"
    df.to_csv(csv_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(csv_path),
            "--outdir",
            str(tmp_path / "outputs"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No transactions remain after cleaning" in result.stderr


def test_cli_reports_clear_error_for_too_few_rows_for_lof_neighbors(tmp_path):
    csv_path = _tiny_transactions(tmp_path)  # 200 rows
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(csv_path),
            "--outdir",
            str(tmp_path / "outputs"),
            "--lof-n-neighbors",
            "500",  # exceeds the 200 rows in the fixture
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Not enough rows" in result.stderr


def test_cli_writes_burst_columns_and_votes_out_of_four_by_default(tmp_path):
    csv_path = _tiny_transactions(tmp_path)
    outdir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(csv_path),
            "--outdir",
            str(outdir),
            "--contamination",
            "0.05",
            "--lof-n-neighbors",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    anomalies = pd.read_csv(outdir / "anomalies.csv")
    assert "zscore_burst" in anomalies.columns
    assert "burst_label" in anomalies.columns
    assert (anomalies["votes"] <= 4).all()


def test_cli_disable_burst_vote_flag_falls_back_to_three_detectors(tmp_path):
    """--disable-burst-vote should reproduce the pre-burst-detector 3-vote
    behavior: votes possible range is 0-3, not 0-4."""
    csv_path = _tiny_transactions(tmp_path)
    outdir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(csv_path),
            "--outdir",
            str(outdir),
            "--contamination",
            "0.05",
            "--lof-n-neighbors",
            "5",
            "--disable-burst-vote",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    anomalies = pd.read_csv(outdir / "anomalies.csv")
    assert "burst_label" in anomalies.columns  # still computed/reported
    assert (anomalies["votes"] <= 3).all()  # but excluded from the vote total


def test_run_models_scales_features_before_fitting(monkeypatch, tmp_path):
    """StandardScaler().fit_transform should be called on the raw feature
    matrix before it reaches IsolationForest/LOF, so unscaled 'amount'
    doesn't dominate LOF's distance calculation."""
    csv_path = _tiny_transactions(tmp_path)
    df = load(str(csv_path))
    df = clean(df)
    df = feature_engineer(df, window=7)

    import src.detect_anomalies as detect_anomalies_module

    calls = []
    original_fit_transform = detect_anomalies_module.StandardScaler.fit_transform

    def spy_fit_transform(self, X, *a, **kw):
        result = original_fit_transform(self, X, *a, **kw)
        calls.append(result)
        return result

    monkeypatch.setattr(detect_anomalies_module.StandardScaler, "fit_transform", spy_fit_transform)
    run_models(df, contamination=0.05, lof_n_neighbors=5)

    assert len(calls) == 1
    scaled = calls[0]
    # Scaled amount column should be roughly standardized (mean ~0, std ~1),
    # unlike the raw amount column which spans hundreds/thousands.
    assert abs(scaled[:, 0].mean()) < 1e-6
    assert abs(scaled[:, 0].std() - 1.0) < 1e-6
