# Contributing

Thanks for considering a contribution. This is a small, educational project,
so the bar is: keep it honest, keep it green.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
make install
```

## Before opening a PR

Run the same gate CI runs:

```bash
make lint   # ruff + black --check + mypy
make test   # pytest with coverage
```

Or install [pre-commit](https://pre-commit.com) to run `ruff`, `black`, and
`mypy` automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Guidelines

- **Minimal churn.** Prefer the smallest change that achieves the goal; avoid
  renaming or moving files unless necessary.
- **Match existing patterns.** Type hints on all `src`/`data` functions,
  docstrings that explain *why* (not just what) for anything non-obvious,
  tests for new behavior and for any bug fix (as a regression test).
- **Prove refactors are behavior-preserving.** If you touch detection logic,
  show before/after output on the same seeded input (see the
  ["Sample Results"](README.md#sample-results) reproduction command).
- **Update the README** if you change a CLI flag, default, or documented
  number.

## Reporting issues

Open a GitHub issue with the command you ran, the input size/shape, and the
full traceback. Since all data here is synthetic, feel free to include exact
CSVs.
