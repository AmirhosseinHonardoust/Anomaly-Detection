# Security Policy

## Scope

This is a small, educational anomaly-detection project. It runs entirely
offline on synthetic data generated locally (`data/generate_transactions.py`);
it does not accept network input, does not connect to any external service,
and does not process real user or financial data. There is no deployed
instance, API, or credential surface to attack.

Realistic concerns in scope for this repo:
- A dependency (pandas, numpy, matplotlib, scikit-learn, or a dev tool) with
  a known CVE that this project's usage pattern could actually trigger.
- A code path that could cause unsafe file writes/reads outside the intended
  `data/`/`outputs/` directories (e.g. via a crafted `--input`/`--outdir`
  path).
- Anything that would let a maliciously crafted (but otherwise well-formed)
  CSV cause more than a clean error (per `src/detect_anomalies.py`'s existing
  validation of required columns, empty-after-clean, and row-count checks).

## Supported Versions

This project does not maintain multiple released versions or long-term
support branches. Only the latest commit on `main` is supported.

## Reporting a Vulnerability

Please do not open a public GitHub issue for a suspected security problem.
Instead, report it privately via GitHub's
["Report a vulnerability"](https://github.com/AmirhosseinHonardoust/Anomaly-Detection/security/advisories/new)
feature, or contact [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)
directly.

Please include:
- A clear description of the issue and its potential impact.
- Steps to reproduce, including the exact command and input (synthetic data
  only, please — do not send real financial data).

As a single-maintainer educational project, there's no guaranteed response
SLA, but reports will be acknowledged and addressed as soon as reasonably
possible. Dependency vulnerabilities are also tracked automatically via
`.github/dependabot.yml`.
