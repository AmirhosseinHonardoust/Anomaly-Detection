<div align="center">

# Anomaly Detection (Transactions & Sales)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest%20%2B%20LOF-orange)
![Detection](https://img.shields.io/badge/Detection-Z--Score%20%2B%20Ensemble%20Voting-green)
![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/actions/workflows/ci.yml)

</div>

A transaction-anomaly detection project that combines **Isolation Forest**, **Local Outlier Factor (LOF)**, and a **per-customer rolling Z-score** into a single **majority-vote ensemble**, with a **synthetic data generator**, **feature scaling**, **severity ranking**, **visual reports**, and **precision/recall/F1 evaluation against ground truth**.

> **Important:** This project runs on **synthetic transaction data** and is an **educational demo**, not a production fraud/anomaly system.
>
> The generator injects known anomaly patterns (bursts, extreme purchases, negative/zero entries) so the pipeline's behavior can be inspected and evaluated against ground truth. Detector defaults (contamination, thresholds, neighbor counts) are illustrative starting points, not tuned for any real dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Generating Synthetic Data](#generating-synthetic-data)
- [Running Anomaly Detection](#running-anomaly-detection)
- [How Anomalies Are Scored](#how-anomalies-are-scored)
- [Evaluating Against Ground Truth](#evaluating-against-ground-truth)
- [Sample Results](#sample-results)
- [Data Schema](#data-schema)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Anomaly detection is often presented as a single model producing a single score. In practice, one detector's blind spots are another's strength: distance-based methods struggle with unscaled features, tree-based methods can miss local density anomalies, and a simple statistical baseline can't see multivariate patterns at all.

This project runs three different detectors over the same transaction data — Isolation Forest, LOF, and a per-customer rolling Z-score — and reports a transaction as anomalous only when at least two of the three agree. It includes a synthetic data generator with known injected anomalies, feature scaling so no single detector is dominated by raw transaction amount, visual reports, and a standalone evaluation script that scores flagged anomalies against ground-truth labels.

The goal is to show a small, honest multi-model detection workflow, including the modeling pitfalls (like a rolling baseline leaking the current row into its own statistics) that are easy to introduce and easy to miss.

---

## What This Project Does

This project can:

- Generate synthetic daily transactions with reproducible, seeded anomalies
- Engineer a per-customer rolling Z-score from **prior transactions only**
- Fit Isolation Forest and LOF on standardized features
- Vote across all three detectors and flag rows with 2+ votes
- Rank flagged rows by a blended 0–1 severity score
- Export flagged rows to CSV and render two summary charts
- Score a set of flagged anomalies against labeled ground truth (precision/recall/F1)
- Fail with a clear error on empty or too-small input, instead of a raw stack trace
- Run automated tests and a GitHub Actions CI workflow (ruff, black, mypy, pytest)

---

## What This Project Does Not Do

This project does **not**:

- Detect fraud or anomalies in real financial data out of the box
- Tune detector thresholds for any specific dataset or business
- Provide real-time/streaming detection — it's a batch CLI
- Guarantee the injected "anomalies" match what a real-world system would flag
- Replace a reviewed, monitored production anomaly-detection pipeline

A production system would need real labeled data, threshold tuning against business cost, drift monitoring, and human review of flagged cases.

---

## Key Features

- **Synthetic transaction generator** with three injected anomaly types: grocery bursts, extreme electronics purchases, and negative/zero-amount entries
- **Per-customer rolling Z-score** computed from prior transactions only, so a spike can't inflate its own baseline
- **Feature scaling** (`StandardScaler`) before Isolation Forest/LOF, so raw `amount` doesn't dominate LOF's distance calculation
- **2-of-3 ensemble voting** across Isolation Forest, LOF, and the Z-score baseline
- **Severity ranking** blending normalized Isolation Forest and LOF scores
- **Clear input-validation errors** for empty-after-cleaning data and too-few-rows-for-LOF-neighbors
- **Standalone evaluation script** for precision/recall/F1 against ground-truth labels
- **Deterministic seeding** throughout the generator and detectors
- **Unit tests and GitHub Actions CI** (ruff, black, mypy, pytest)

---

## System Workflow

```text
Synthetic transaction generator (seeded)
        ↓
Load + clean (drop dup tx_id, extreme negatives)
        ↓
Feature engineering (calendar features + prior-only rolling Z-score)
        ↓
Feature scaling (StandardScaler)
        ↓
Isolation Forest  +  Local Outlier Factor  +  Z-score baseline
        ↓
2-of-3 vote  →  flagged anomalies
        ↓
Severity ranking (blended Isolation Forest / LOF score)
        ↓
anomalies.csv + charts  →  optional scoring against ground truth
```

---

## Project Structure

```text
Anomaly-Detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── .gitkeep
│   └── generate_transactions.py
│
├── src/
│   ├── detect_anomalies.py
│   ├── evaluate.py
│   └── utils.py
│
├── tests/
│   ├── test_detect_anomalies.py
│   ├── test_evaluate.py
│   ├── test_generate_transactions.py
│   └── test_utils.py
│
├── outputs/
│   ├── anomalies.csv
│   ├── fig_amount_time.png
│   └── fig_amount_hist.png
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

`outputs/` is not committed — it's generated locally by the commands below.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Anomaly-Detection.git
cd Anomaly-Detection
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (pytest, ruff, black, mypy):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Generate synthetic data, then run detection:

```bash
python data/generate_transactions.py --start 2023-01-01 --end 2024-12-31 --seed 42 --n-customers 500 --out data/transactions.csv
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02
```

---

## Generating Synthetic Data

No sample data ships with the repo — generate it locally first (this also lets you pick any date range or size):

```bash
python data/generate_transactions.py \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --seed 42 \
  --n-customers 500 \
  --out data/transactions.csv
```

The generator injects three anomaly patterns via a seeded RNG, so the same `--seed` always reproduces the same dataset:

<div align="center">

| Pattern | Description |
|---|---|
| Grocery bursts | ~5% of days get an extra 30–80 grocery transactions |
| Extreme electronics purchases | ~3% of days get 3–10 electronics purchases in the $1,500–$5,000 range |
| Negative/zero-amount entries | ~2% of days get 2–6 rows at $0 or a small negative amount |

</div>

---

## Running Anomaly Detection

```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02
```

Generated outputs:

```text
outputs/anomalies.csv        # flagged rows, votes, and severity score
outputs/fig_amount_time.png  # transaction amounts over time
outputs/fig_amount_hist.png  # amount distribution histogram
```

All detector parameters have working defaults matching the values above:

```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs \
  --contamination 0.02 --rolling-window 7 --lof-n-neighbors 35 --zscore-threshold 3.5
```

---

## How Anomalies Are Scored

Each row gets a vote from three detectors — Isolation Forest, LOF, and a per-customer rolling Z-score. A row is reported as an anomaly if **at least 2 of the 3** detectors flag it. `severity` blends the Isolation Forest and LOF scores (each min-max normalized to [0, 1]) into a single 0–1 ranking score.

> The rolling Z-score (`zscore_7`) is also one of the four features fed to Isolation Forest and LOF, so the three votes aren't fully independent — the Z-score baseline informs the ML models too, not just its own standalone vote. The baseline itself only looks at a customer's **prior** transactions (never the current row), so a large transaction can't inflate its own baseline and dampen its own score.

Isolation Forest and LOF features are standardized (zero mean, unit variance) before fitting, so `amount` (which can run into the thousands) doesn't dominate LOF's distance-based neighbor calculation over the smaller-scale `dayofweek` / `month` / `zscore_7` features.

Bad input is rejected early with a clear message rather than a raw sklearn stack trace — an input that's empty after cleaning, or has fewer rows than `--lof-n-neighbors`, raises a `ValueError` explaining why.

---

## Evaluating Against Ground Truth

`src/evaluate.py` scores a set of flagged anomalies against labeled ground truth:

```bash
python src/evaluate.py --labeled data/labeled.csv --anomalies outputs/anomalies.csv
```

`--labeled` expects a CSV with `tx_id` and a boolean `is_anomaly` column; `--anomalies` expects any CSV with a `tx_id` column (e.g. `outputs/anomalies.csv`). The script prints precision, recall, and F1 computed from true/false positives and false negatives matched by `tx_id`.

---

## Sample Results

### Transaction Amount Distribution

Shows most transactions are small (0–300 units). A few very large amounts (thousands) appear as outliers.

<div align="center">
<img width="1280" height="640" alt="Transaction amount distribution" src="https://github.com/user-attachments/assets/3aa896cc-9b3a-4a69-b96d-c20feaaaed2c" />
</div>

### Transaction Amounts Over Time

Transactions are generally stable, but occasional spikes (extreme purchases or errors) appear and are flagged as anomalies.

<div align="center">
<img width="1920" height="640" alt="Transaction amounts over time" src="https://github.com/user-attachments/assets/fba4e248-dcf6-4c62-9e96-94d607fa5a88" />
</div>

---

## Data Schema

<div align="center">

| Column | Description |
|---|---|
| `tx_id` | Unique transaction ID |
| `date` | Timestamp (daily resolution) |
| `customer_id` | Customer identifier |
| `category` | Product category |
| `amount` | Transaction amount (float) |

</div>

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Lint, format-check, and type-check:

```bash
ruff check --select E,F,I,B,SIM,UP --line-length 100 .
black --line-length 100 --check .
mypy --ignore-missing-imports src data
```

The GitHub Actions workflow runs, in order: dependency installation, ruff, black (`--check`), mypy, and pytest.

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

<div align="center">

| Module | Purpose |
|---|---|
| `data/generate_transactions.py` | Seeded synthetic transaction generator with injected anomalies |
| `src/utils.py` | Cleaning, feature engineering, and the rolling Z-score baseline |
| `src/detect_anomalies.py` | Isolation Forest / LOF / Z-score ensemble, CLI, and report generation |
| `src/evaluate.py` | Precision/recall/F1 scoring of flagged anomalies against ground truth |

</div>

Tooling is configured through `pyproject.toml` (ruff, black, mypy, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- All data is synthetic — real transaction data has different statistical properties
- Detector defaults (contamination, thresholds, neighbor counts) are illustrative, not tuned
- The Z-score baseline feature is shared with the ML detectors' inputs, so the three votes aren't fully independent (see [How Anomalies Are Scored](#how-anomalies-are-scored))
- `severity` is a relative, per-run ranking (min-max normalized), not a calibrated probability
- No streaming/online detection — this is a batch CLI over a static CSV

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- pytest
- ruff
- black
- mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

MIT, see [LICENSE](LICENSE).
