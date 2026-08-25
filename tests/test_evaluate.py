import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import score

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "evaluate.py"


def test_score_computes_precision_recall_f1():
    labeled = pd.DataFrame(
        {
            "tx_id": ["a", "b", "c", "d", "e"],
            "is_anomaly": [True, True, False, False, True],
        }
    )
    # Flags a, b (true positives), c (false positive); misses e (false negative).
    anomalies = pd.DataFrame({"tx_id": ["a", "b", "c"]})

    result = score(labeled, anomalies)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_score_handles_no_flags_and_no_ground_truth():
    labeled = pd.DataFrame({"tx_id": ["a", "b"], "is_anomaly": [False, False]})
    anomalies = pd.DataFrame({"tx_id": []})
    result = score(labeled, anomalies)
    assert result == {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_score_raises_without_is_anomaly_column():
    labeled = pd.DataFrame({"tx_id": ["a"]})
    anomalies = pd.DataFrame({"tx_id": ["a"]})
    with pytest.raises(ValueError, match="is_anomaly"):
        score(labeled, anomalies)


def test_cli_end_to_end(tmp_path):
    labeled_path = tmp_path / "labeled.csv"
    anomalies_path = tmp_path / "anomalies.csv"
    pd.DataFrame(
        {"tx_id": ["a", "b", "c"], "is_anomaly": [True, False, True]}
    ).to_csv(labeled_path, index=False)
    pd.DataFrame({"tx_id": ["a"]}).to_csv(anomalies_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--labeled",
            str(labeled_path),
            "--anomalies",
            str(anomalies_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Precision:" in result.stdout
    assert "Recall:" in result.stdout
    assert "F1:" in result.stdout
