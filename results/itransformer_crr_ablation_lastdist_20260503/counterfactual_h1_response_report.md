# Counterfactual Horizon-1 Response Evaluation

Generated at: `20260503_201643`

The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. Therefore, an intervention on the last historical cause value has a known counterfactual effect on horizon 1.

| Model | Intervention | Expected | Predicted | IRE MAE | Slope | Sign Acc. |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | cause_last_shift_plus_delta | 5.000384 | 0.032361 | 4.993866 | 0.001304 | 0.553806 |
| iTransformer | cause_shift_plus_delta | 5.000384 | 0.000002 | 5.000384 | -0.000000 | 0.337008 |
| iTransformer | cause_zero | 0.854292 | 0.029782 | 0.857482 | -0.002299 | 0.497638 |
| iTransformer | cause_flip | 1.708583 | 0.029249 | 1.706396 | 0.001841 | 0.558005 |
| iTransformer | cause_flip_x5 | 5.125750 | 0.029249 | 5.123481 | 0.000614 | 0.558005 |
| iTransformer | target_zero | 0.000000 | 0.869898 | 0.869898 | nan | nan |
| iTransformer_FT_PREDONLY | cause_last_shift_plus_delta | 5.000384 | 0.061599 | 4.986578 | 0.002761 | 0.555381 |
| iTransformer_FT_PREDONLY | cause_shift_plus_delta | 5.000384 | 0.000002 | 5.000384 | -0.000000 | 0.350656 |
| iTransformer_FT_PREDONLY | cause_zero | 0.854292 | 0.057922 | 0.862046 | -0.008040 | 0.512861 |
| iTransformer_FT_PREDONLY | cause_flip | 1.708583 | 0.056383 | 1.709070 | -0.000249 | 0.545932 |
| iTransformer_FT_PREDONLY | cause_flip_x5 | 5.125750 | 0.056383 | 5.126020 | -0.000083 | 0.545932 |
| iTransformer_FT_PREDONLY | target_zero | 0.000000 | 0.881977 | 0.881977 | nan | nan |
| iTransformer_CRR_FT01_RESPONLY | cause_last_shift_plus_delta | 5.000384 | 4.277363 | 0.755437 | 0.855407 | 1.000000 |
| iTransformer_CRR_FT01_RESPONLY | cause_shift_plus_delta | 5.000384 | 0.000006 | 5.000384 | 0.000000 | 0.529134 |
| iTransformer_CRR_FT01_RESPONLY | cause_zero | 0.854292 | 0.253466 | 0.797789 | 0.102331 | 0.613123 |
| iTransformer_CRR_FT01_RESPONLY | cause_flip | 1.708583 | 0.271327 | 1.622805 | 0.060581 | 0.627822 |
| iTransformer_CRR_FT01_RESPONLY | cause_flip_x5 | 5.125750 | 0.271327 | 5.029470 | 0.020194 | 0.627822 |
| iTransformer_CRR_FT01_RESPONLY | target_zero | 0.000000 | 0.741792 | 0.741792 | nan | nan |
| iTransformer_CRR_FT01 | cause_last_shift_plus_delta | 5.000384 | 4.149043 | 0.871410 | 0.829745 | 1.000000 |
| iTransformer_CRR_FT01 | cause_shift_plus_delta | 5.000384 | 0.000007 | 5.000383 | 0.000000 | 0.536483 |
| iTransformer_CRR_FT01 | cause_zero | 0.854292 | 0.293344 | 0.773140 | 0.134615 | 0.631496 |
| iTransformer_CRR_FT01 | cause_flip | 1.708583 | 0.308498 | 1.594673 | 0.074982 | 0.646194 |
| iTransformer_CRR_FT01 | cause_flip_x5 | 5.125750 | 0.308498 | 4.999933 | 0.024994 | 0.646194 |
| iTransformer_CRR_FT01 | target_zero | 0.000000 | 0.744640 | 0.744640 | nan | nan |
