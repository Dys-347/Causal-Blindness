# Reproducibility Guide

This document explains how to inspect and rerun the artifact. The goal is to make
the causal-response diagnostics transparent without requiring reviewers to
download large checkpoints or raw training logs.

## 1. Environment

The standalone analysis scripts require a standard Python scientific stack:

```bash
pip install -r requirements.txt
```

GPU-backed training scripts additionally require the dependencies of the external
forecasting backbone being evaluated, such as Time-Series-Library or DUET.

## 2. Quick Inspection Without Training

The repository includes compact result summaries. These files are sufficient to
check the numerical evidence reported by the experiments:

```text
results/priority_20260504/summary.md
results/lookback_20260504/summary.md
results/ett_augmented_20260504/summary.md
results/causal_r1_linear_baselines_20260504/linear_causal_baselines_report.md
```

For a visual sanity check, inspect:

```text
results/duet_crr_20260503/duet_baseline_vs_rir_curve.png
results/delta_response_itransformer_crr_curve_20260503/delta_response_curve.png
results/gradient_sensitivity_20260504/report.md
```

## 3. Standalone Diagnostics

The following scripts are designed to be readable and runnable with local paths:

```bash
python scripts/linear_causal_baselines.py --help
python scripts/counterfactual_h1_response_eval.py --help
python scripts/evaluate_delta_response_curve.py --help
python scripts/gradient_input_sensitivity.py --help
```

They implement the response metrics, delta-response curves, linear sanity
baselines, and functional sensitivity diagnostics used in the artifact.

## 4. Backbone-Dependent Experiments

Some experiments depend on external forecasting implementations. The
project-specific code is included here, but the external libraries themselves are
not vendored.

### Time-Series-Library Based Runs

Relevant scripts:

```text
scripts/train_itransformer_crr_h1.py
scripts/train_itransformer_etth1_rir_side_effect.py
scripts/train_synthetic_dlinear_timemixer.sh
scripts/train_synthetic_timemixer_only.sh
```

These scripts should be run in, or pointed to, a Time-Series-Library checkout.
Dataset paths can be provided through command-line arguments. For the synthetic
CSV, the scripts also support:

```bash
export CAUSAL_R1_SYNTHETIC_CSV=/path/to/synthetic_multivariate.csv
```

### DUET Based Runs

Relevant scripts:

```text
scripts/train_eval_duet_synthetic_causal.py
scripts/train_eval_duet_crr_synthetic.py
scripts/train_eval_duet_baseline_curve.py
scripts/run_lookback_experiments_20260504.sh
```

The shell wrappers use environment variables for local configuration:

```bash
export DUET_DIR=/path/to/DUET-main
export TSL_DIR=/path/to/Time-Series-Library
export LOG_ROOT=./causal_r1_runs
```

## 5. What Is Intentionally Excluded

The artifact excludes:

- submission write-up files,
- raw third-party datasets,
- model checkpoints,
- raw training logs,
- private server paths or credentials,
- local Python and LaTeX cache files.

The included CSV/Markdown summaries and selected plots are intended to support
transparent review of the reported diagnostics while keeping the repository small
and safe to inspect.
