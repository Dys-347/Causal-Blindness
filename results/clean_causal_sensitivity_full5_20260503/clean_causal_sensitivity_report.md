# Clean Causal Sensitivity Evaluation

Generated at: `20260503_185013`

## Channel Map

- channel 0: `Cause_Var`
- channels 1-19: distractors
- channel 20 / -1: `Target_Var`

## Summary

| Model | Target MSE | Target MAE | Cause shift MPD | Distractor shift MPD | CSR | Target-zero / Cause-shift |
|---|---:|---:|---:|---:|---:|---:|
| DLinear | 0.545448 | 0.622045 | 0.000000 | 0.000000 | 0.000000 | 562089312963.076904 |
| PatchTST | 0.296033 | 0.427234 | 0.000000 | 0.000000 | 0.000000 | 730165069393.181396 |
| iTransformer | 0.060812 | 0.190868 | 0.000002 | 0.000006 | 0.373839 | 385685.685956 |
| Crossformer | 0.015730 | 0.099358 | 0.003551 | 0.034936 | 0.101635 | 247.367861 |
| TimeMixer | 0.010938 | 0.083087 | 0.000000 | 0.000000 | 0.000000 | 857566416983.216431 |

Detailed rows are in `clean_causal_sensitivity_summary.csv`.
