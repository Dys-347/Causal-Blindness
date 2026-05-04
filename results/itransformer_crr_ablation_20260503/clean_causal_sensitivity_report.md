# Clean Causal Sensitivity Evaluation

Generated at: `20260503_201443`

## Channel Map

- channel 0: `Cause_Var`
- channels 1-19: distractors
- channel 20 / -1: `Target_Var`

## Summary

| Model | Target MSE | Target MAE | Cause shift MPD | Distractor shift MPD | CSR | Target-zero / Cause-shift |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer | 0.060812 | 0.190868 | 0.000002 | 0.000006 | 0.373839 | 385685.685956 |
| iTransformer_FT_PREDONLY | 0.082554 | 0.224174 | 0.000003 | 0.000008 | 0.352935 | 311408.397658 |
| iTransformer_CRR_FT01_RESPONLY | 0.109450 | 0.259257 | 0.000003 | 0.000008 | 0.383795 | 278550.039397 |
| iTransformer_CRR_FT01 | 0.105430 | 0.255280 | 0.000003 | 0.000009 | 0.362760 | 257205.394277 |

Detailed rows are in `clean_causal_sensitivity_summary.csv`.
