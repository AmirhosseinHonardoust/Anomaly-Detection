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

# Centralized defaults so the CLI flags and the underlying function
# signatures can't drift out of sync with each other.
DEFAULT_CONTAMINATION = 0.02
DEFAULT_ROLLING_WINDOW = 7
DEFAULT_LOF_N_NEIGHBORS = 35
DEFAULT_ZSCORE_THRESHOLD = 3.5

# Cap on how many rows are sampled for the time-series plot, to keep large
# datasets readable and fast to render.
PLOT_SAMPLE_CAP = 20000


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
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = 42,
    lof_n_neighbors: int = DEFAULT_LOF_N_NEIGHBORS,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Fit Isolation Forest, LOF, and the Z-score baseline; return labels/scores.

    Features are standardized (zero mean, unit variance) before fitting.
    This has no effect on Isolation Forest (its splits are per-feature and
    scale-invariant) but matters for LOF, which is distance-based: without
    scaling, ``amount`` (range ~0-5000) would dominate ``dayofweek``/``month``
    (range 0-12), effectively hiding the calendar features from LOF's notion
    of a neighborhood.
    """
    feats = df[["amount", "dayofweek", "month", "zscore_7"]].to_numpy()
    feats_scaled = StandardScaler().fit_transform(feats)

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    iso_labels = iso.fit_predict(feats_scaled)
    iso_score = -iso.decision_function(feats_scaled)

    lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, contamination=contamination)
    lof_labels = lof.fit_predict(feats_scaled)
    lof_score = -lof.negative_outlier_factor_

    z_flags = zscore_flags(df, th=zscore_threshold).astype(int)
    return iso_labels, iso_score, lof_labels, lof_score, z_flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to transactions.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    ap.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="rolling window (in transactions) for the per-customer Z-score baseline",
    )
    ap.add_argument(
        "--lof-n-neighbors",
        type=int,
        default=DEFAULT_LOF_N_NEIGHBORS,
        help="n_neighbors passed to sklearn's LocalOutlierFactor",
    )
    ap.add_argument(
        "--zscore-threshold",
        type=float,
        default=DEFAULT_ZSCORE_THRESHOLD,
        help="absolute rolling Z-score above which a row is flagged",
    )
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    try:
        df = load(args.input)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as exc:
        print(f"[ERROR] Could not parse input CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    df = clean(df)
    if df.empty:
        print("[ERROR] No transactions remain after cleaning; check input data.", file=sys.stderr)
        sys.exit(1)
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
    df_plot = out.sample(n=min(PLOT_SAMPLE_CAP, len(out)), random_state=42).sort_values("date")
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
