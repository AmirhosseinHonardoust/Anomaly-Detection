"""Generate a synthetic transactions dataset with injected anomalies."""

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CATEGORIES = ["Electronics", "Home", "Beauty", "Toys", "Grocery"]

# Category mix for ordinary background transactions (must align with CATEGORIES).
CATEGORY_PROBS = [0.25, 0.22, 0.2, 0.18, 0.15]

# (mu, sigma) of the log-normal amount distribution per category for background
# transactions: amount = exp(Normal(mu, sigma)).
CATEGORY_LOGNORMAL_PARAMS = {
    "Electronics": (4.5, 0.6),
    "Home": (3.5, 0.5),
    "Beauty": (3.0, 0.4),
    "Toys": (3.1, 0.45),
    "Grocery": (2.2, 0.3),
}

# Average number of background transactions generated per day (Poisson lambda).
BASE_DAILY_LAMBDA = 300

# Anomaly type 1: a same-day burst of many small Grocery purchases.
GROCERY_BURST_DAILY_PROB = 0.05
GROCERY_BURST_SIZE_RANGE = (30, 80)  # inclusive-low/exclusive-high, per rng.integers
GROCERY_BURST_LOGNORMAL_PARAMS = (1.9, 0.25)  # (mu, sigma), same log-normal shape as background

# Anomaly type 2: a same-day burst of unusually large Electronics purchases.
ELECTRONICS_BURST_DAILY_PROB = 0.03
ELECTRONICS_BURST_SIZE_RANGE = (3, 10)
ELECTRONICS_BURST_AMOUNT_RANGE = (1500.0, 5000.0)

# Anomaly type 3: a same-day handful of negative/zero-amount entries (refunds,
# data-entry errors, etc.), spread across any category.
NEGATIVE_ZERO_DAILY_PROB = 0.02
NEGATIVE_ZERO_SIZE_RANGE = (2, 6)
NEGATIVE_ZERO_AMOUNT_RANGE = (1.0, 50.0)


def generate(start: str, end: str, seed: int, n_customers: int) -> pd.DataFrame:
    """Generate synthetic daily transactions between ``start`` and ``end``.

    Injects three kinds of anomalies (all driven by the seeded ``rng`` so the
    output is fully reproducible for a given seed): grocery bursts, extreme
    electronics purchases, and negative/zero-amount entries.

    The returned frame always carries an ``is_anomaly`` column marking which
    rows were injected as one of those three anomaly types (``True``) versus
    ordinary background traffic (``False``). This ground truth is what lets
    ``src/evaluate.py`` score how well the detectors actually find the
    anomalies; ``main()`` strips the column by default so it doesn't leak
    into a "real" input CSV unless explicitly requested via
    ``--include-labels``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []
    for d in dates:
        base_n = rng.poisson(lam=BASE_DAILY_LAMBDA)
        for _ in range(base_n):
            tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
            cust = int(rng.integers(1, n_customers + 1))
            cat = rng.choice(CATEGORIES, p=CATEGORY_PROBS)
            mu_sigma = CATEGORY_LOGNORMAL_PARAMS[cat]
            amount = float(np.round(np.exp(rng.normal(*mu_sigma)), 2))
            rows.append([tx_id, d.date(), cust, cat, amount, False])

        if rng.random() < GROCERY_BURST_DAILY_PROB:
            for _ in range(rng.integers(*GROCERY_BURST_SIZE_RANGE)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers + 1))
                cat = "Grocery"
                amount = float(np.round(np.exp(rng.normal(*GROCERY_BURST_LOGNORMAL_PARAMS)), 2))
                rows.append([tx_id, d.date(), cust, cat, amount, True])

        if rng.random() < ELECTRONICS_BURST_DAILY_PROB:
            for _ in range(rng.integers(*ELECTRONICS_BURST_SIZE_RANGE)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers + 1))
                cat = "Electronics"
                amount = float(np.round(rng.uniform(*ELECTRONICS_BURST_AMOUNT_RANGE), 2))
                rows.append([tx_id, d.date(), cust, cat, amount, True])

        if rng.random() < NEGATIVE_ZERO_DAILY_PROB:
            for _ in range(rng.integers(*NEGATIVE_ZERO_SIZE_RANGE)):
                tx_id = f"{d.strftime('%Y%m%d')}-{rng.integers(1_000_000, 9_999_999)}"
                cust = int(rng.integers(1, n_customers + 1))
                cat = rng.choice(CATEGORIES)
                amount = float(rng.choice([0.0, -rng.uniform(*NEGATIVE_ZERO_AMOUNT_RANGE)]))
                rows.append([tx_id, d.date(), cust, cat, amount, True])

    df = pd.DataFrame(
        rows, columns=["tx_id", "date", "customer_id", "category", "amount", "is_anomaly"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="start date, e.g. 2023-01-01")
    ap.add_argument("--end", required=True, help="end date, e.g. 2024-12-31")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--n-customers", type=int, default=500)
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument(
        "--include-labels",
        action="store_true",
        help=(
            "keep the 'is_anomaly' ground-truth column in the output CSV "
            "(useful for scoring detector quality with src/evaluate.py; "
            "omitted by default so the file matches a 'real' unlabeled input)"
        ),
    )
    ap.add_argument("-q", "--quiet", action="store_true", help="only log warnings and errors")
    ap.add_argument("-v", "--verbose", action="store_true", help="log debug-level detail")
    args = ap.parse_args()

    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    df = generate(args.start, args.end, args.seed, args.n_customers)
    if not args.include_labels:
        df = df.drop(columns=["is_anomaly"])
    df.to_csv(args.out, index=False)
    logger.info("[OK] wrote %s with %s rows", args.out, f"{len(df):,}")


if __name__ == "__main__":
    main()
