"""Score detector output against injected ground-truth anomaly labels.

This closes a gap in the pipeline: ``data/generate_transactions.py`` knows
exactly which rows it injected as anomalies, but that information is
normally discarded so it doesn't leak into a "real" input file. This script
lets you measure precision/recall/F1 of the vote-based flags in
``anomalies.csv`` against that ground truth, using a labeled CSV produced
with ``--include-labels``.

Usage:
    python data/generate_transactions.py --start 2023-01-01 --end 2023-06-01 \\
        --seed 42 --out data/labeled.csv --include-labels
    python src/detect_anomalies.py --input data/labeled.csv --outdir outputs
    python src/evaluate.py --labeled data/labeled.csv --anomalies outputs/anomalies.csv
"""

import argparse
import sys

import pandas as pd


def score(labeled: pd.DataFrame, anomalies: pd.DataFrame) -> dict[str, float]:
    """Compute precision/recall/F1 of flagged rows against ``is_anomaly``.

    ``labeled`` must contain ``tx_id`` and ``is_anomaly`` (the full,
    unfiltered dataset). ``anomalies`` must contain ``tx_id`` (the flagged
    subset written by ``detect_anomalies.py``); only rows that survived
    ``clean()`` can be scored, since flagged rows are a subset of cleaned
    input.
    """
    if "is_anomaly" not in labeled.columns:
        raise ValueError(
            "labeled input has no 'is_anomaly' column; regenerate it with "
            "generate_transactions.py --include-labels"
        )

    flagged_ids = set(anomalies["tx_id"])
    true_ids = set(labeled.loc[labeled["is_anomaly"], "tx_id"])
    scored_ids = set(labeled["tx_id"])

    # Only score against rows the detector actually saw (i.e. survived
    # cleaning); flagged_ids is already a subset of that by construction.
    true_ids &= scored_ids

    tp = len(flagged_ids & true_ids)
    fp = len(flagged_ids - true_ids)
    fn = len(true_ids - flagged_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labeled", required=True, help="labeled transactions CSV (is_anomaly column)"
    )
    ap.add_argument(
        "--anomalies", required=True, help="anomalies.csv written by detect_anomalies.py"
    )
    args = ap.parse_args()

    try:
        labeled = pd.read_csv(args.labeled)
        anomalies = pd.read_csv(args.anomalies)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        result = score(labeled, anomalies)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"True positives:  {result['true_positives']}")
    print(f"False positives: {result['false_positives']}")
    print(f"False negatives: {result['false_negatives']}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall:    {result['recall']:.3f}")
    print(f"F1:        {result['f1']:.3f}")


if __name__ == "__main__":
    main()
