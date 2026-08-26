.PHONY: install lint format test cover run clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check --select E,F,I,B,SIM,UP --line-length 100 .
	black --line-length 100 --check .
	mypy --ignore-missing-imports src data

format:
	ruff check --select E,F,I,B,SIM,UP --line-length 100 --fix .
	black --line-length 100 .

test:
	pytest -v

cover:
	pytest -v --cov=src --cov=data --cov-report=term-missing

# Generate synthetic data, then run the detection pipeline against it.
run:
	python data/generate_transactions.py --start 2023-01-01 --end 2024-12-31 --seed 42 --n-customers 500 --out data/transactions.csv
	python src/detect_anomalies.py --input data/transactions.csv --outdir outputs --contamination 0.02

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +
