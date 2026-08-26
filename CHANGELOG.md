# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project doesn't cut versioned releases, so entries are grouped by PR under
"Unreleased" as they land on `main`.

## [Unreleased]

### Added
- `SECURITY.md` — vulnerability reporting policy, scoped to what this project
  actually is (offline, synthetic data, no deployed service).
- `.github/CODEOWNERS` — default reviewer for PRs.
- This changelog.

### Added (packaging/tooling PR)
- `src/__init__.py`, `tests/__init__.py`, and a `[project]`/`[build-system]`
  table in `pyproject.toml` so `pip install -e .` works.
- `.pre-commit-config.yaml` mirroring the CI gate (ruff, black, mypy).
- `.github/dependabot.yml` for weekly pip and GitHub Actions update checks.
- `CONTRIBUTING.md` documenting the local dev/gate workflow.
- README "Tuning Guide" section under Sample Results, with measured (not
  estimated) precision/recall/F1 across `--contamination` and
  `--zscore-threshold` values.

### Changed
- `data/generate_transactions.py`: magic numbers (Poisson lambda, category
  probabilities/log-normal params, burst probabilities/ranges) extracted to
  named module-level constants. Verified byte-identical CSV output for the
  same seed before/after.
- `data/generate_transactions.py`, `src/detect_anomalies.py`: `print()`
  replaced with `logging` (+ `-q`/`-v` flags). `src/evaluate.py` deliberately
  left on `print()`/stdout, since its precision/recall/F1 output is the
  tool's actual result and is asserted on stdout by
  `tests/test_evaluate.py::test_cli_end_to_end`.

### Fixed
- CI matrix briefly included Python 3.11, which can't install the pinned
  `numpy==2.5.2` (requires >=3.12). Reverted to a single 3.12 job, matching
  the existing pins and `[tool.mypy] python_version = "3.12"`. Also corrected
  `pyproject.toml`'s `requires-python` from `>=3.11` to `>=3.12` for
  consistency.
