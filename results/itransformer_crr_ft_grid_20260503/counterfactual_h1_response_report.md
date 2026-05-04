# Counterfactual Horizon-1 Response Evaluation

Generated at: `20260503_200624`

The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. Therefore, an intervention on the last historical cause value has a known counterfactual effect on horizon 1.

| Model | Intervention | Expected | Predicted | IRE MAE | Slope | Sign Acc. |
|---|---|---:|---:|---:|---:|---:|
| iTransformer | cause_last_shift_plus_delta | 5.000384 | 0.032361 | 4.993866 | 0.001304 | 0.553806 |
| iTransformer | cause_shift_plus_delta | 5.000384 | 0.000002 | 5.000384 | -0.000000 | 0.337008 |
| iTransformer | cause_zero | 0.854292 | 0.029782 | 0.857482 | -0.002299 | 0.497638 |
| iTransformer | cause_flip | 1.708583 | 0.029249 | 1.706396 | 0.001841 | 0.558005 |
| iTransformer | cause_flip_x5 | 5.125750 | 0.029249 | 5.123481 | 0.000614 | 0.558005 |
| iTransformer | target_zero | 0.000000 | 0.869898 | 0.869898 | nan | nan |
| iTransformer_CRR_FT01 | cause_last_shift_plus_delta | 5.000384 | 4.149043 | 0.871410 | 0.829745 | 1.000000 |
| iTransformer_CRR_FT01 | cause_shift_plus_delta | 5.000384 | 0.000007 | 5.000383 | 0.000000 | 0.536483 |
| iTransformer_CRR_FT01 | cause_zero | 0.854292 | 0.293344 | 0.773140 | 0.134615 | 0.631496 |
| iTransformer_CRR_FT01 | cause_flip | 1.708583 | 0.308498 | 1.594673 | 0.074982 | 0.646194 |
| iTransformer_CRR_FT01 | cause_flip_x5 | 5.125750 | 0.308498 | 4.999933 | 0.024994 | 0.646194 |
| iTransformer_CRR_FT01 | target_zero | 0.000000 | 0.744640 | 0.744640 | nan | nan |
| iTransformer_CRR_FT03 | cause_last_shift_plus_delta | 5.000384 | 4.205089 | 0.815873 | 0.840953 | 1.000000 |
| iTransformer_CRR_FT03 | cause_shift_plus_delta | 5.000384 | 0.000007 | 5.000384 | 0.000000 | 0.526509 |
| iTransformer_CRR_FT03 | cause_zero | 0.854292 | 0.327534 | 0.771618 | 0.154335 | 0.649869 |
| iTransformer_CRR_FT03 | cause_flip | 1.708583 | 0.345461 | 1.589435 | 0.085023 | 0.657743 |
| iTransformer_CRR_FT03 | cause_flip_x5 | 5.125750 | 0.345461 | 4.991083 | 0.028341 | 0.657743 |
| iTransformer_CRR_FT03 | target_zero | 0.000000 | 0.667752 | 0.667752 | nan | nan |
| iTransformer_CRR | cause_last_shift_plus_delta | 5.000384 | 4.793955 | 0.243046 | 0.958717 | 1.000000 |
| iTransformer_CRR | cause_shift_plus_delta | 5.000384 | 0.000002 | 5.000384 | 0.000000 | 0.586352 |
| iTransformer_CRR | cause_zero | 0.854292 | 0.098032 | 0.824220 | 0.037063 | 0.618898 |
| iTransformer_CRR | cause_flip | 1.708583 | 0.090287 | 1.693462 | 0.005099 | 0.558530 |
| iTransformer_CRR | cause_flip_x5 | 5.125750 | 0.090287 | 5.109070 | 0.001699 | 0.558530 |
| iTransformer_CRR | target_zero | 0.000000 | 0.412794 | 0.412794 | nan | nan |
