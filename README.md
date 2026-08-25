<div align="center">
 
# Anomaly Detection (Transactions & Sales)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest%20%2B%20LOF-orange)
![Statistics](https://img.shields.io/badge/Statistics-Rolling%20Z--Score-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/actions/workflows/ci.yml)

</div>

A transaction-fraud-style anomaly detection project that combines **Isolation Forest**, **Local Outlier Factor (LOF)**, and a **per-customer rolling Z-score baseline** into a single voting pipeline, with a **reproducible synthetic data generator**, **severity-ranked reporting**, and **audit-ready visualizations**.

> **Important:** This project is a **synthetic anomaly-detection demo**, not a production fraud system.
>
> All transaction data is generated locally with a seeded random number generator. The goal is to demonstrate an honest, multi-model detection workflow with clear scoring logic, not to model real customer or payment data.

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
- [Data Schema](#data-schema)
- [Sample Results](#sample-results)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Anomaly detection is often presented as a single model producing a single score. In practice, no one detector is reliable on its own: Isolation Forest and LOF catch different shapes of outliers, and a simple statistical baseline can catch things both models miss (or flag things neither should). A detector is only useful if its output can support a defensible review action:

- flag rows multiple independent methods agree on
- rank flagged rows by severity instead of treating them all the same
- stay reproducible so results can be regenerated and audited
- expose the data and the decision logic, not just a black-box label

This project demonstrates an end-to-end anomaly-detection workflow on synthetic transaction data: data generation with injected anomalies, multi-model detection, a voting and severity system, and exportable charts and reports for audit.

The goal is to show how independent detectors can be combined into a **transparent, review-ready anomaly report**, not just a single model's outlier flag.

---

## What This Project Does

This project can:

- Generate a fully synthetic, reproducible transactions dataset for any date range
- Inject three kinds of anomalies: grocery order bursts, extreme electronics purchases, and negative/zero-amount entries
- Engineer calendar features and a per-customer rolling Z-score baseline
- Run Isolation Forest and Local Outlier Factor detectors alongside the Z-score baseline
- Combine all three into a single vote-based anomaly flag
- Rank flagged rows with a blended 0–1 severity score
- Export flagged rows to CSV with model votes and severity
- Generate time-series and distribution charts for audit
- Run automated tests and a GitHub Actions CI workflow

---

## What This Project Does Not Do

This project does **not**:

- Use or require any real transaction, customer, or payment data
- Guarantee that flagged rows are actual fraud or errors
- Replace a real-time fraud-detection or payments-risk system
- Tune itself automatically, thresholds are set via CLI flags
- Perform any network calls, database access, or external lookups

A real fraud-detection system would need live transaction streams, case management, labeled ground truth, and human investigator review.

---

## Key Features

- **Synthetic transaction generator** with a seeded RNG for full reproducibility
- **Three injected anomaly types**: grocery bursts, extreme electronics purchases, negative/zero entries
- **Calendar feature engineering**: day-of-week, month, and a per-customer rolling Z-score
- **Multi-model detection**: Isolation Forest, Local Outlier Factor, and a rolling Z-score baseline
- **Vote-based flagging**: a row is reported only if at least 2 of 3 detectors agree
- **Blended severity score** combining normalized Isolation Forest and LOF scores
- **Configurable detectors**: contamination, rolling window, LOF neighbors, and Z-score threshold are all CLI flags
- **Audit visualizations**: transaction amounts over time and amount-distribution histogram
- **Unit tests and GitHub Actions CI**

---

## System Workflow

```text
Synthetic transaction generator (seeded)
        ↓
Cleaning (dedupe, drop implausible amounts)
        ↓
Feature engineering (calendar features, rolling Z-score)
        ↓
Isolation Forest   +   Local Outlier Factor   +   Z-score baseline
        ↓
Vote-based flagging (>= 2 of 3 detectors agree)
        ↓
Blended severity score (0-1)
        ↓
anomalies.csv + time-series & distribution charts
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
│   ├── generate_transactions.py
│   └── .gitkeep
│
├── outputs/
│   ├── anomalies.csv
│   ├── fig_amount_time.png
│   └── fig_amount_hist.png
│
├── src/
│   ├── detect_anomalies.py
│   └── utils.py
│
├── tests/
│   ├── test_detect_anomalies.py
│   ├── test_generate_transactions.py
│   └── test_utils.py
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

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

For development tools (pytest, Ruff, Black, mypy):

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Generate synthetic data:

```bash
python data/generate_transactions.py --start 2023-01-01 --end 2024-12-31 --seed 42 --n-customers 500 --out data/transactions.csv
```

Run anomaly detection:

```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02
```

---

## Generating Synthetic Data

No sample data ships with the repo, generate it locally first, which also lets you pick any date range or size:

```bash
python data/generate_transactions.py \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --seed 42 \
  --n-customers 500 \
  --out data/transactions.csv
```

The generator is fully reproducible for a given `--seed` and injects three kinds of anomalies alongside normal daily activity: grocery order bursts, extreme electronics purchases, and negative/zero-amount entries.

---

## Running Anomaly Detection

```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02
```

Generated outputs include:

```text
outputs/anomalies.csv
outputs/fig_amount_time.png
outputs/fig_amount_hist.png
```

**Outputs** (written to `outputs/`, not committed, regenerate with the command above)

- `outputs/anomalies.csv` – flagged rows with anomaly scores and model votes
- `outputs/fig_amount_time.png` – transaction amounts over time with spikes
- `outputs/fig_amount_hist.png` – amount distribution histogram

**Configuring the detectors** | all of these have working defaults matching the values above:

```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs \
  --contamination 0.02 --rolling-window 7 --lof-n-neighbors 35 --zscore-threshold 3.5
```

---

## How Anomalies Are Scored

Each row gets a vote from three independent detectors:

<div align="center">

| Detector | What it catches |
|---|---|
| Isolation Forest | Rows that are easy to isolate across all engineered features |
| Local Outlier Factor | Rows in locally sparse regions relative to their neighbors |
| Rolling Z-score | Amounts that deviate sharply from a customer's own recent history, plus non-positive amounts |

</div>

A row is reported as an anomaly if **at least 2 of the 3 detectors** flag it. `severity` blends the Isolation Forest and LOF scores, each min-max normalized to `[0, 1]`, into a single 0–1 ranking score, so reviewers can triage the most severe rows first.

> Vote-based flagging is a transparency choice: a single model's outlier score can be noisy, but agreement across independently-behaving detectors is a stronger signal.

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

## Sample Results

### Transaction Amount Distribution

Shows most transactions are small (0–300 units). A few very large amounts (thousands) appear as outliers.

<div align="center">
<img width="1280" height="640" alt="fig_amount_hist" src="https://github.com/user-attachments/assets/3aa896cc-9b3a-4a69-b96d-c20feaaaed2c" />
</div>

---

### Transaction Amounts Over Time

Transactions are generally stable, but occasional spikes (extreme purchases or errors) appear and are flagged as anomalies.

<div align="center">
<img width="1920" height="640" alt="fig_amount_time" src="https://github.com/user-attachments/assets/fba4e248-dcf6-4c62-9e96-94d607fa5a88" />
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

The GitHub Actions workflow checks:

- dependency installation
- linting with Ruff
- formatting with Black
- type-checking with mypy
- unit tests with pytest

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `data/generate_transactions.py` | Seeded synthetic transaction generator with injected anomalies |
| `src/utils.py` | Cleaning, calendar feature engineering, and the rolling Z-score baseline |
| `src/detect_anomalies.py` | Runs all three detectors, computes votes and severity, writes reports and charts |

</div>

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- All data is synthetic, no real transaction or customer data is used or required
- The anomaly types are hand-designed, so real-world fraud patterns may look different
- Thresholds (contamination, Z-score cutoff, LOF neighbors) are set manually, not learned
- No ground-truth labels exist, so precision/recall against "true" anomalies can't be computed
- The vote-based rule treats all three detectors as equally trustworthy, which may not hold on other data

The project is strongest as a portfolio demonstration of a multi-model, vote-based anomaly-detection workflow.

---

## Future Improvements

Potential next improvements:

- Add labeled or semi-labeled anomalies to measure precision/recall
- Add an autoencoder or other deep-learning detector to the vote
- Add per-category and per-customer severity breakdowns
- Add a Streamlit dashboard for interactive review of flagged rows
- Add configurable anomaly-injection rates to the data generator
- Add Docker support for a fully reproducible run
- Explore streaming/incremental anomaly detection

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- pytest
- Ruff
- Black
- mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License see [LICENSE](LICENSE) for details.

If you use or modify this project, please keep the synthetic-data notice and limitations clear.
