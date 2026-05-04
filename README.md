# Causal Blindness: Code and Reproducibility Artifact

This repository contains the project-specific code for diagnosing **causal
blindness** in multivariate time-series forecasting.

The artifact includes training adapters, evaluation scripts, response-curve
analysis, functional sensitivity analysis, linear sanity baselines, augmented-ETT
diagnostics, and compact result summaries. 

## What The Code Tests

The central diagnostic asks whether a forecasting model responds correctly when a
known driver is intervened on. In the controlled synthetic benchmark, one target
is generated from one cause:

```text
Y[t] = 2 * C[t-1] + noise
```

Therefore a last-step intervention on `C` has an analytically known horizon-1
effect on `Y`. This makes it possible to measure not only observational error
(`MSE`/`MAE`), but also response correctness:

- horizon-1 interventional response error,
- response slope across intervention magnitudes,
- small-delta/local response behavior,
- functional input sensitivity,
- negative-control distractor sensitivity.

## Code Availability

All project-specific code for the diagnostics is provided under `scripts/`.

Some training scripts are adapters for existing forecasting implementations:

- Time-Series-Library for iTransformer, DLinear, PatchTST, Crossformer, and
  TimeMixer style experiments.
- DUET for DUET-Mix and DUET-related experiments.

The full external libraries are not vendored into this repository. This keeps the
artifact focused on the proposed diagnostics and avoids redistributing
third-party code. The scripts expose command-line arguments and environment
variables so that local dataset and backbone paths can be set explicitly.

## Repository Layout

```text
scripts/              Project-specific training, evaluation, and plotting code
results/              Compact result summaries, CSVs, and selected diagnostic plots
requirements.txt      Minimal dependencies for standalone analysis scripts
REPRODUCIBILITY.md    Step-by-step artifact inspection guide
CODE_STRUCTURE.md     Script-by-script map of the codebase
```

## Artifact Scope

Large generated artifacts such as model checkpoints, raw benchmark downloads, raw
training logs, and local cache files are not versioned. The repository instead
contains the code needed to regenerate the diagnostics and compact result files
that make the reported numbers auditable.

## Main Result Summaries

The fastest way to inspect the numerical evidence is:

- `results/priority_20260504/summary.md`
- `results/lookback_20260504/summary.md`
- `results/ett_augmented_20260504/summary.md`
- `results/causal_r1_linear_baselines_20260504/linear_causal_baselines_report.md`
- `results/duet_crr_20260503/duet_baseline_vs_rir_curve.png`

See `results/README.md` for a more complete result map.

## Reproduction Entry Points

Install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

Standalone analysis scripts can be inspected directly:

```bash
python scripts/linear_causal_baselines.py --help
python scripts/counterfactual_h1_response_eval.py --help
python scripts/evaluate_delta_response_curve.py --help
python scripts/gradient_input_sensitivity.py --help
```

Backbone-dependent training scripts should be placed inside the corresponding
external project checkout, or called with paths pointing to that checkout:

```bash
# DUET synthetic baseline / RIR diagnostics
python scripts/train_eval_duet_synthetic_causal.py --help
python scripts/train_eval_duet_crr_synthetic.py --help

# iTransformer RIR and augmented-ETT diagnostics
python scripts/train_itransformer_crr_h1.py --help
python scripts/train_itransformer_etth1_rir_side_effect.py --help
```

More detailed instructions are in `REPRODUCIBILITY.md`.

## Scope Notes

Raw Traffic, Electricity, and ETT datasets provide observed futures but do not
provide ground-truth `do`-intervention labels. For that reason, causal response
evaluation here uses:

- a controlled synthetic benchmark with exact counterfactual response labels,
- augmented ETT diagnostics where a semi-synthetic causal target is appended to
  real benchmark covariates.

Original benchmark targets such as `OT` in ETT are used only as side-effect
diagnostics, not as causal ground truth.
