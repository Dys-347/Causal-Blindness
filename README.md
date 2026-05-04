# Causal Blindness in Multivariate Time-Series Forecasting

This repository contains the code, paper draft, generated figures, and compact
result summaries for:

**Low Error, Wrong Response: Causal Blindness in Multivariate Time-Series
Forecasting**

The project studies a failure mode of multivariate long-term time-series
forecasting (MLTSF): a model can obtain low observational MSE/MAE while producing
near-zero response to an intervention on the true driver of the target. We call
this failure mode **causal blindness**.

The repository is organized for reviewer inspection. It focuses on scripts and
diagnostic outputs rather than large checkpoints or raw benchmark datasets.

## What Is Included

- `paper/`:
  NeurIPS-style draft, bibliography, checklist, generated PDF, and publication
  figures.
- `scripts/`:
  Training, evaluation, response-curve, functional-sensitivity, and figure
  generation scripts.
- `results/`:
  Compact CSV/Markdown summaries and selected plots used by the paper.

Large model checkpoints, raw third-party datasets, private server logs, and local
cache files are intentionally excluded.

## Main Diagnostics

The controlled synthetic benchmark contains one cause `C`, one target `Y`, and
19 distractors. The target obeys:

```text
Y[t] = 2 * C[t-1] + noise
```

This gives an analytically known horizon-1 response to a last-step cause
intervention. The paper reports:

- observational target MSE/MAE,
- horizon-1 interventional response error,
- response slope across intervention magnitudes,
- functional input sensitivity,
- distractor and target-history sensitivity checks.

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

# Generate paper figures
python scripts/make_neurips_figures.py
```

Some scripts target external implementations:

- `train_itransformer_crr_h1.py` and
  `train_itransformer_etth1_rir_side_effect.py` are designed for the
  Time-Series-Library experiment structure.
- `train_eval_duet_synthetic_causal.py`,
  `train_eval_duet_crr_synthetic.py`, and
  `train_eval_duet_baseline_curve.py` are designed for the DUET codebase.

## Key Result Summaries

See `results/README.md` for a compact map from result files to paper tables and
figures. The most important reviewer-facing summaries are:

- `results/priority_20260504/summary.md`
- `results/lookback_20260504/summary.md`
- `results/ett_augmented_20260504/summary.md`
- `paper/neurips_draft_v1.pdf`

## Notes For Reviewers

Standard real-world MLTSF datasets such as Traffic, Electricity, and raw ETT
provide observed futures but not ground-truth `do`-intervention labels. Therefore,
causal response evaluation in this repository uses:

- a controlled synthetic benchmark with exact counterfactual response labels,
- augmented ETT diagnostics where a semi-synthetic causal target is appended to
  real benchmark covariates.

Raw benchmark targets such as `OT` in ETT are used only as side-effect diagnostics,
not as causal evidence.
