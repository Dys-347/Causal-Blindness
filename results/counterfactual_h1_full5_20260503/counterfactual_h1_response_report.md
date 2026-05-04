# Counterfactual Horizon-1 Response Evaluation

Generated at: `20260503_184850`

The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. Therefore, an intervention on the last historical cause value has a known counterfactual effect on horizon 1.

| Model | Intervention | Expected | Predicted | IRE MAE | Slope | Sign Acc. |
|---|---|---:|---:|---:|---:|---:|
| DLinear | cause_shift_plus_delta | 5.000384 | 0.000000 | 5.000384 | 0.000000 | 0.000000 |
| DLinear | cause_zero | 0.854292 | 0.000000 | 0.854292 | 0.000000 | 0.000000 |
| DLinear | cause_flip | 1.708583 | 0.000000 | 1.708583 | 0.000000 | 0.000000 |
| DLinear | cause_flip_x5 | 5.125750 | 0.000000 | 5.125750 | 0.000000 | 0.000000 |
| DLinear | target_zero | 0.000000 | 0.648138 | 0.648138 | nan | nan |
| PatchTST | cause_shift_plus_delta | 5.000384 | 0.000000 | 5.000384 | 0.000000 | 0.000000 |
| PatchTST | cause_zero | 0.854292 | 0.000000 | 0.854292 | 0.000000 | 0.000000 |
| PatchTST | cause_flip | 1.708583 | 0.000000 | 1.708583 | 0.000000 | 0.000000 |
| PatchTST | cause_flip_x5 | 5.125750 | 0.000000 | 5.125750 | 0.000000 | 0.000000 |
| PatchTST | target_zero | 0.000000 | 0.798587 | 0.798587 | nan | nan |
| iTransformer | cause_shift_plus_delta | 5.000384 | 0.000002 | 5.000384 | -0.000000 | 0.337008 |
| iTransformer | cause_zero | 0.854292 | 0.029782 | 0.857482 | -0.002299 | 0.497638 |
| iTransformer | cause_flip | 1.708583 | 0.029249 | 1.706396 | 0.001841 | 0.558005 |
| iTransformer | cause_flip_x5 | 5.125750 | 0.029249 | 5.123481 | 0.000614 | 0.558005 |
| iTransformer | target_zero | 0.000000 | 0.869898 | 0.869898 | nan | nan |
| Crossformer | cause_shift_plus_delta | 5.000384 | 0.004354 | 4.996057 | 0.000865 | 0.973753 |
| Crossformer | cause_zero | 0.854292 | 0.001743 | 0.853838 | 0.000593 | 0.629921 |
| Crossformer | cause_flip | 1.708583 | 0.003062 | 1.706174 | 0.001342 | 0.770079 |
| Crossformer | cause_flip_x5 | 5.125750 | 0.006173 | 5.120486 | 0.000943 | 0.858268 |
| Crossformer | target_zero | 0.000000 | 0.884160 | 0.884160 | nan | nan |
| TimeMixer | cause_shift_plus_delta | 5.000384 | 0.000000 | 5.000384 | 0.000000 | 0.000000 |
| TimeMixer | cause_zero | 0.854292 | 0.000000 | 0.854292 | 0.000000 | 0.000000 |
| TimeMixer | cause_flip | 1.708583 | 0.000000 | 1.708583 | 0.000000 | 0.000000 |
| TimeMixer | cause_flip_x5 | 5.125750 | 0.000000 | 5.125750 | 0.000000 | 0.000000 |
| TimeMixer | target_zero | 0.000000 | 0.837711 | 0.837711 | nan | nan |
