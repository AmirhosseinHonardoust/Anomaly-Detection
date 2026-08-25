# Anomaly Detection (Transactions & Sales)
![CI](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/actions/workflows/ci.yml/badge.svg)

Detect anomalies in synthetic transaction data using Isolation Forest, Local Outlier Factor (LOF), and a Z-score baseline. The project generates data, injects anomalies, runs detectors, and exports flagged rows and charts for audit.

---

## Features
- Synthetic transaction generator with anomalies (bursts, extreme purchases, negative/zero entries)
- Multi-model detection: Isolation Forest, LOF, Z-score
- Unified anomaly report with model votes and severity score
- Clean visualizations: time-series spikes and amount distribution
- Reproducible scripts with deterministic seeding

---

## Project Structure
```
anomaly-detection/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  └─ generate_transactions.py
├─ src/
│  ├─ detect_anomalies.py
│  └─ utils.py
└─ outputs/
   └─ figures & reports
```

---

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Generate Synthetic Data
No sample data ships with the repo — generate it locally first (this also lets you pick any date range/size):
```bash
python data/generate_transactions.py --start 2023-01-01 --end 2024-12-31 --seed 42 --n-customers 500 --out data/transactions.csv
```

---

## Run Anomaly Detection
```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02
```

**Outputs** (written to `outputs/`, not committed — regenerate with the command above)
- `outputs/anomalies.csv` – flagged rows with anomaly scores & model votes  
- `outputs/fig_amount_time.png` – transaction amounts over time with spikes  
- `outputs/fig_amount_hist.png` – amount distribution histogram  

**How anomalies are scored**
Each row gets a vote from three independent detectors (Isolation Forest, LOF, and a per-customer rolling Z-score). A row is reported as an anomaly if at least 2 of the 3 detectors flag it. `severity` blends the Isolation Forest and LOF scores (each min-max normalized to [0, 1]) into a single 0–1 ranking score.

**Configuring the detectors** — all of these have working defaults matching the values above:
```bash
python src/detect_anomalies.py --input data/transactions.csv --outdir outputs \
  --contamination 0.02 --rolling-window 7 --lof-n-neighbors 35 --zscore-threshold 3.5
```

---

## Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check --select E,F,I,B,SIM,UP --line-length 100 .
black --line-length 100 .
mypy --ignore-missing-imports src data
pytest
```

---

## Sample Results

### Transaction Amount Distribution
Shows most transactions are small (0–300 units). A few very large amounts (thousands) appear as outliers.

<img width="1280" height="640" alt="fig_amount_hist" src="https://github.com/user-attachments/assets/3aa896cc-9b3a-4a69-b96d-c20feaaaed2c" />

---

### Transaction Amounts Over Time
Transactions are generally stable, but occasional spikes (extreme purchases or errors) appear and are flagged as anomalies.

<img width="1920" height="640" alt="fig_amount_time" src="https://github.com/user-attachments/assets/fba4e248-dcf6-4c62-9e96-94d607fa5a88" />

---

## Data Schema
| column      | description                      |
|-------------|----------------------------------|
| tx_id       | unique transaction ID            |
| date        | timestamp (daily resolution)     |
| customer_id | customer identifier              |
| category    | product category                 |
| amount      | transaction amount (float)       |
