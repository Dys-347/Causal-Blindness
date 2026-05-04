# Clean Causal Sensitivity Evaluation

Generated at: `20260503_200647`

## Channel Map

- channel 0: `Cause_Var`
- channels 1-19: distractors
- channel 20 / -1: `Target_Var`

## Summary

| Model | Target MSE | Target MAE | Cause shift MPD | Distractor shift MPD | CSR | Target-zero / Cause-shift |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer | 0.060812 | 0.190868 | 0.000002 | 0.000006 | 0.373839 | 385685.685956 |
| iTransformer_CRR_FT01 | 0.105430 | 0.255280 | 0.000003 | 0.000009 | 0.362760 | 257205.394277 |
| iTransformer_CRR_FT03 | 0.146527 | 0.302024 | 0.000003 | 0.000008 | 0.389664 | 247331.168892 |
| iTransformer_CRR | 0.530811 | 0.591046 | 0.000002 | 0.000007 | 0.286268 | 311816.730992 |

Detailed rows are in `clean_causal_sensitivity_summary.csv`.
