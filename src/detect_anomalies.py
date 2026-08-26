"""Run Isolation Forest, LOF, and Z-score anomaly detection on transactions."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from utils import clean, feature_engineer, zscore_flags  # noqa: E402

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
    out["votes"] = out[["iso_label", "lof_label", "zscore_label"]].sum(axis=1)
    s_iso = (iso_score - iso_score.min()) / max(1e-9, (iso_score.max() - iso_score.min()))
    s_lof = (lof_score - lof_score.min()) / max(1e-9, (lof_score.max() - lof_score.min()))
    out["severity"] = (s_iso + s_lof) / 2.0

    anomalies = out[out["votes"] >= 2].sort_values(["severity"], ascending=False)
    anomalies.to_csv(os.path.join(args.outdir, "anomalies.csv"), index=False)

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    df_plot = out.sample(n=min(20000, len(out)), random_state=42).sort_values("date")
    ax1.plot(df_plot["date"], df_plot["amount"])
    ax1.set_title("Transaction Amount over Time (sample)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Amount")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, "fig_amount_time.png"), dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(out["amount"], bins=60)
    ax2.set_title("Amount Distribution")
    ax2.set_xlabel("Amount")
    ax2.set_ylabel("Count")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "fig_amount_hist.png"), dpi=160)
    plt.close(fig2)

    print("[OK] Anomaly detection complete.")
    print(f"Flagged anomalies: {len(anomalies):,}")
    print(f"Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
