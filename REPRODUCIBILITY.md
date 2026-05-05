# Reproducibility Guide

This document explains how to inspect and rerun the artifact. The goal is to make
the causal-response diagnostics transparent while keeping the repository compact.

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
python scripts/generate_synthetic_mechanism_family.py --help
python scripts/linear_causal_baselines.py --help
python scripts/train_tsl_seeded_synthetic.py --help
python scripts/evaluate_tsl_seeded_synthetic.py --help
python scripts/counterfactual_h1_response_eval.py --help
python scripts/evaluate_delta_response_curve.py --help
python scripts/gradient_input_sensitivity.py --help
```

They implement the response metrics, delta-response curves, linear sanity
baselines, and functional sensitivity diagnostics used in the artifact.

To generate the controlled mechanism-family extension used for reviewer-defense
experiments:

```bash
python scripts/generate_synthetic_mechanism_family.py \
  --output-dir dataset/causal_r1_mechanism_family \
  --mechanisms linear_one_lag linear_multi_lag nonlinear_sin
```

Each generated CSV has a companion `.meta.json` file that defines the exact
horizon-1 response label. This is important for nonlinear mechanisms, where the
correct response is sample-dependent rather than a single constant slope.

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

For the main failure multi-seed rerun and the seed-specific iTransformer repair
rerun, the recommended wrappers are:

```text
scripts/run_main_multiseed_20260505.sh
scripts/run_itransformer_repair_multiseed_20260505.sh
```

The baseline wrapper trains DLinear, PatchTST, iTransformer, Crossformer, and
TimeMixer with seed-specific settings and then evaluates each checkpoint with the
same response metrics used in the paper. The iTransformer repair wrapper reuses
the seed-specific iTransformer baseline checkpoint as initialization before
applying response regularization.

`evaluate_tsl_seeded_synthetic.py` also writes `window_h1_response.csv` for each
checkpoint. This file stores the per-window predicted H1 response, analytic
expected response, absolute error, and sign-correctness indicator, so reviewers can
check that near-zero average response is not an artifact of positive and negative
windows cancelling each other out.

After copying baseline and repair outputs back into `results/`, the per-window
distribution can be visualized with:

```bash
python scripts/plot_window_response_distribution.py \
  --input-root results/main_multiseed_20260505 \
  --input-root results/itransformer_repair_multiseed_20260505 \
  --label iTransformer \
  --label "iTransformer + RIR FT01"
```

### DUET Based Runs

Relevant scripts:

```text
scripts/train_eval_duet_synthetic_causal.py
scripts/train_eval_duet_crr_synthetic.py
scripts/train_eval_duet_baseline_curve.py
scripts/run_lookback_experiments_20260504.sh
scripts/run_v3_mechanism_family_20260504.sh
```

The shell wrappers use environment variables for local configuration:

```bash
export DUET_DIR=/path/to/DUET-main
export TSL_DIR=/path/to/Time-Series-Library
export LOG_ROOT=./causal_r1_runs
```

For the v3 mechanism-family extension:

```bash
export DUET_DIR=/path/to/DUET-main
bash scripts/run_v3_mechanism_family_20260504.sh
```

After copying the generated DUET output directory back into this artifact, aggregate
it with:

```bash
python scripts/summarize_v3_mechanism_family.py \
  --input-root results/duet_v3_mechanism_family_20260504 \
  --output-dir results/v3_mechanism_family_20260504
```

## 5. Artifact Scope

The repository stores project-specific code, compact CSV/Markdown summaries, and
selected diagnostic plots. Raw benchmark downloads, model checkpoints, and full
training logs are treated as generated artifacts: they should be downloaded,
regenerated, or produced by rerunning the corresponding scripts in the local
backbone environment.

This layout keeps the artifact small enough to inspect while preserving the code
paths used to compute the response metrics and aggregate result summaries.
