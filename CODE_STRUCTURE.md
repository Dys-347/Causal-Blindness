# Code Structure

This file maps each public script to its role in the artifact.

## Core Evaluation Scripts

| Script | Purpose |
|---|---|
| `linear_causal_baselines.py` | AR, ARX, and VARX sanity baselines under the same response-evaluation pipeline. |
| `generate_synthetic_mechanism_family.py` | Generates controlled one-lag, multi-lag, and nonlinear synthetic mechanism families with metadata for exact H1 response labels. |
| `synthetic_mechanism_utils.py` | Shared metadata loader and response-label utility for synthetic mechanism-family experiments. |
| `counterfactual_h1_response_eval.py` | Horizon-1 intervention-response evaluation for trained forecasting models. |
| `evaluate_delta_response_curve.py` | Delta-response curves across positive and negative intervention magnitudes. |
| `gradient_input_sensitivity.py` | Functional central-difference sensitivity to cause, target-history, and distractor inputs. |
| `clean_causal_sensitivity.py` | Broad all-history perturbation diagnostics. |

## RIR / Response-Regularization Training

| Script | Purpose |
|---|---|
| `train_itransformer_crr_h1.py` | iTransformer response-regularized training and fine-tuning on the synthetic benchmark. |
| `train_itransformer_etth1_rir_side_effect.py` | Augmented ETT side-effect diagnostic with an appended semi-synthetic causal target. |
| `train_eval_duet_crr_synthetic.py` | DUET response-regularized synthetic benchmark training and evaluation. |

## DUET Baselines And Robustness

| Script | Purpose |
|---|---|
| `train_eval_duet_synthetic_causal.py` | DUET synthetic benchmark baseline evaluation. |
| `train_eval_duet_baseline_curve.py` | DUET baseline delta-response curve evaluation. |
| `run_lookback_experiments_20260504.sh` | DUET-Mix look-back window robustness wrapper. |
| `run_v3_mechanism_family_20260504.sh` | DUET-Mix baseline/RIR wrapper for multi-lag and nonlinear controlled mechanism-family robustness. |

## Experiment Wrappers And Summaries

| Script | Purpose |
|---|---|
| `run_priority_experiments_20260504.sh` | Multi-run wrapper for priority reviewer-defense experiments. |
| `run_ett_augmented_20260504.sh` | Augmented ETTh2/ETTm1-style diagnostic wrapper. |
| `summarize_priority_20260504.py` | Aggregates priority multi-seed results into compact CSV/Markdown summaries. |
| `summarize_lookback_20260504.py` | Aggregates DUET look-back robustness results. |
| `summarize_v3_mechanism_family.py` | Aggregates mechanism-family baseline/RIR runs into compact CSV/Markdown summaries. |

## Visualization

| Script | Purpose |
|---|---|
| `make_neurips_figures.py` | Recreates the polished concept, workflow, sensitivity, and response-curve figures from compact numbers. |

## Notes

The scripts are intentionally explicit rather than hidden behind a large
framework. This makes it easier to inspect how each response metric is computed.
External backbone repositories are required only for experiments that train or
evaluate those models directly.
