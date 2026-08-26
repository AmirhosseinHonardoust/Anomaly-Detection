"""Run Isolation Forest, LOF, Z-score, and burst anomaly detection on transactions."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Explicitly add this script's own directory to sys.path so `from utils import
# ...` resolves regardless of how this file is invoked (as `python
# src/detect_anomalies.py`, `python -m src.detect_anomalies`, or imported by
# a test). Python only adds the script's directory automatically for the
# first form, so we do it ourselves rather than relying on that behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotting import save_report_figures  # noqa: E402
from utils import add_burst_feature, clean, feature_engineer, zscore_flags  # noqa: E402

REQUIRED_COLUMNS = {"tx_id", "date", "customer_id", "category", "amount"}


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load(path: str) -> pd.DataFrame:
    """Load the transactions CSV and validate its schema."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    return df


def run_models(
    df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
    lof_n_neighbors: int = 35,
    zscore_threshold: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Fit Isolation Forest, LOF, and the Z-score baseline; return labels/scores."""
    feats = df[["amount", "dayofweek", "month", "zscore_7"]].to_numpy()
    # LOF is distance-based, so unscaled "amount" (which can run into the
    # thousands) would dominate its Euclidean neighbor distances over
    # dayofweek/month/zscore_7 (single digits). Standardize all detector
    # inputs so each feature contributes comparably.
    scaled_feats = StandardScaler().fit_transform(feats)

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    iso_labels = iso.fit_predict(scaled_feats)
    iso_score = -iso.decision_function(scaled_feats)

    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=contamination)
    lof_labels = lof.fit_predict(scaled_feats)
    lof_score = -lof.negative_outlier_factor_

    z_flags = zscore_flags(df, th=zscore_threshold).astype(int)
    return iso_labels, iso_score, lof_labels, lof_score, z_flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to transactions.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument(
        "--rolling-window",
        type=int,
        default=7,
        help="rolling window (in transactions) for the per-customer Z-score baseline",
    )
    ap.add_argument(
        "--lof-n-neighbors",
        type=int,
        default=35,
        help="n_neighbors passed to sklearn's LocalOutlierFactor",
    )
    ap.add_argument(
        "--zscore-threshold",
        type=float,
        default=3.5,
        help="absolute rolling Z-score above which a row is flagged",
    )
    ap.add_argument(
        "--burst-day-window",
        type=int,
        default=30,
        help="rolling window (in days) for the per-category daily-volume burst baseline",
    )
    ap.add_argument(
        "--burst-threshold",
        type=float,
        default=2.0,
        help="absolute burst Z-score above which a day/category is flagged as a volume burst",
    )
    ap.add_argument(
        "--disable-burst-vote",
        action="store_true",
        help=(
            "vote using only Isolation Forest, LOF, and the per-customer Z-score "
            "(pre-burst-detector 3-vote behavior), instead of the default 4-vote "
            "ensemble that also votes on per-category daily-volume bursts"
        ),
    )
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = load(args.input)
    df = clean(df)
    if df.empty:
        raise ValueError(
            "No transactions remain after cleaning (clean() drops duplicate "
            "tx_id rows and amounts <= -1000); check --input."
        )
    if len(df) <= args.lof_n_neighbors:
        raise ValueError(
            f"Not enough rows ({len(df)}) for --lof-n-neighbors={args.lof_n_neighbors}; "
            "reduce --lof-n-neighbors or provide more data."
        )
    df = feature_engineer(df, window=args.rolling_window)
    df = add_burst_feature(df, day_window=args.burst_day_window)

    iso_labels, iso_score, lof_labels, lof_score, z_flags = run_models(
        df,
        contamination=args.contamination,
        lof_n_neighbors=args.lof_n_neighbors,
        zscore_threshold=args.zscore_threshold,
    )

    out = df.copy()
    out["iso_label"] = (iso_labels == -1).astype(int)
    out["lof_label"] = (lof_labels == -1).astype(int)
    out["zscore_label"] = z_flags
    out["burst_label"] = (out["zscore_burst"].abs() >= args.burst_threshold).astype(int)

    vote_cols = ["iso_label", "lof_label", "zscore_label"]
    if not args.disable_burst_vote:
        vote_cols.append("burst_label")
    out["votes"] = out[vote_cols].sum(axis=1)

    s_iso = (iso_score - iso_score.min()) / max(1e-9, (iso_score.max() - iso_score.min()))
    s_lof = (lof_score - lof_score.min()) / max(1e-9, (lof_score.max() - lof_score.min()))
    out["severity"] = (s_iso + s_lof) / 2.0

    anomalies = out[out["votes"] >= 2].sort_values(["severity"], ascending=False)
    anomalies.to_csv(os.path.join(args.outdir, "anomalies.csv"), index=False)

    save_report_figures(out, args.outdir)

    print("[OK] Anomaly detection complete.")
    print(f"Flagged anomalies: {len(anomalies):,}")
    print(f"Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
