import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detect_anomalies import load, run_models  # noqa: E402
from utils import clean, feature_engineer  # noqa: E402

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
