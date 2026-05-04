# Causal Blindness Reproducibility Package

This repository is a **reproducibility evidence package** for the project
**Causal Blindness in Multivariate Time-Series Forecasting**.

It is intentionally organized around code, diagnostic scripts, compact result
summaries, and selected plots. The anonymous manuscript is not included here; the
repository is meant to let reviewers inspect how the reported diagnostics were
computed and where the numerical evidence comes from.

## Core Idea

The project studies a failure mode of multivariate long-term time-series
forecasting (MLTSF): a model can obtain low observational MSE/MAE while producing
near-zero response to an intervention on the true driver of the target. We call
this failure mode **causal blindness**.

The controlled synthetic benchmark contains one cause `C`, one target `Y`, and
19 distractors. The target obeys:

```text
Y[t] = 2 * C[t-1] + noise
```

This gives an analytically known horizon-1 response to a last-step cause
intervention.

## What Is Included

- `scripts/`:
  Training, evaluation, response-curve, functional-sensitivity, linear sanity
  baseline, DUET, iTransformer, and augmented-ETT scripts.
- `results/`:
  Compact CSV/Markdown summaries and selected plots used to audit the reported
  results.
- `requirements.txt`:
  Minimal Python dependencies for the standalone analysis scripts.

Large model checkpoints, raw third-party datasets, private server logs, local
cache files, and manuscript files are intentionally excluded.

## Main Diagnostics

The repository includes scripts and results for:

- observational target MSE/MAE,
- horizon-1 interventional response error,
- response slope across intervention magnitudes,
- small-delta/local response checks,
- functional central-difference input sensitivity,
- AR/ARX/VARX linear sanity baselines,
- DUET-Mix look-back window robustness,
- augmented ETTh1/ETTh2 side-effect diagnostics.

## Method

The repository includes **Randomized Intervention-Response Regularization (RIR)**,
a lightweight objective that aligns the forecast response to known interventions.
RIR uses:

- an ordinary prediction loss,
- a response loss for a known driver-target pair,
- a distractor stability loss for negative-control variables.

RIR does not perform causal discovery. It assumes that at least one intervention
response label is available from domain knowledge, a simulator, a controlled
intervention, or benchmark construction.

## Reproducing Core Analyses

The scripts are lightweight adapters around the Time-Series-Library and DUET
codebases used in the experiments. Install dependencies from `requirements.txt`,
place the scripts inside the corresponding backbone repository when needed, and
adjust dataset paths using command-line arguments or environment variables.

Examples:

```bash
# Linear sanity baselines and small-delta response checks
python scripts/linear_causal_baselines.py --help

# Horizon-1 counterfactual response evaluation
python scripts/counterfactual_h1_response_eval.py --help

# Delta-response curves
python scripts/evaluate_delta_response_curve.py --help

# Functional central-difference sensitivity
python scripts/gradient_input_sensitivity.py --help
```

Some scripts target external implementations:

- `train_itransformer_crr_h1.py` and
  `train_itransformer_etth1_rir_side_effect.py` are designed for the
  Time-Series-Library experiment structure.
- `train_eval_duet_synthetic_causal.py`,
  `train_eval_duet_crr_synthetic.py`, and
  `train_eval_duet_baseline_curve.py` are designed for the DUET codebase.

## Key Result Summaries

See `results/README.md` for a compact map from result files to diagnostics. The
most important reviewer-facing summaries are:

- `results/priority_20260504/summary.md`
- `results/lookback_20260504/summary.md`
- `results/ett_augmented_20260504/summary.md`
- `results/causal_r1_linear_baselines_20260504/linear_causal_baselines_report.md`

## Notes For Reviewers

Standard real-world MLTSF datasets such as Traffic, Electricity, and raw ETT
provide observed futures but not ground-truth `do`-intervention labels. Therefore,
causal response evaluation in this repository uses:

- a controlled synthetic benchmark with exact counterfactual response labels,
- augmented ETT diagnostics where a semi-synthetic causal target is appended to
  real benchmark covariates.

Raw benchmark targets such as `OT` in ETT are used only as side-effect diagnostics,
not as causal evidence.
