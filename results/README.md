# Result Map

This directory contains compact summaries and selected plots for reproducibility
inspection. Large checkpoints, raw logs, and raw datasets are not included in the
public release.

## Reviewer-Facing Summaries

- `priority_20260504/summary.md`:
  three-seed DUET-Mix and augmented ETTh1 priority results.
- `lookback_20260504/summary.md`:
  DUET-Mix look-back window robustness for `T in {48, 96, 192, 336}`.
- `ett_augmented_20260504/summary.md`:
  three-seed augmented ETTh2 diagnostic.
- `causal_r1_linear_baselines_20260504/linear_causal_baselines_report.md`:
  AR/ARX/VARX sanity checks under the same standardization and intervention
  pipeline.
- `delta_response_itransformer_crr_curve_20260503/`:
  iTransformer response-curve repair diagnostics.
- `duet_crr_20260503/duet_baseline_vs_rir_curve.png`:
  DUET-Mix baseline-vs-RIR response curve used in the appendix.

## Interpretation

The CSV and Markdown files are intended to make paper tables auditable without
requiring reviewers to rerun every training job. Paths inside older raw JSON files
may reflect the original experiment machine and are not required for reproduction.
