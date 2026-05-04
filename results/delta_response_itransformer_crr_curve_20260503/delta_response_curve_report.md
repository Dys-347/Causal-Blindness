# Delta-Response Curve Evaluation

Generated at: `20260503_213537`

The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. We intervene on the last historical cause value and measure the horizon-1 target response.

## Model-Level Summary

| Model | Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |
|---|---:|---:|---:|---:|
| iTransformer | -0.000843 | -0.227592 | 2.591262 | -0.009014 |
| iTransformer_CRR_FT01 | 0.983589 | 0.983902 | 0.544662 | 1.221568 |
| iTransformer_CRR_FT03 | 0.986726 | 0.986175 | 0.509646 | 1.198287 |

## Per-Delta Response

| Model | Delta | Expected | Predicted | Ratio | IRE |
|---|---:|---:|---:|---:|---:|
| iTransformer | -5.000 | -5.000384 | 0.000305 | -0.000061 | 5.000689 |
| iTransformer | -4.000 | -4.000308 | 0.003046 | -0.000761 | 4.003353 |
| iTransformer | -3.000 | -3.000231 | 0.007192 | -0.002397 | 3.007422 |
| iTransformer | -2.000 | -2.000154 | 0.014998 | -0.007498 | 2.015152 |
| iTransformer | -1.000 | -1.000077 | 0.022676 | -0.022674 | 1.022753 |
| iTransformer | -0.500 | -0.500038 | 0.014922 | -0.029842 | 0.514961 |
| iTransformer | 0.500 | 0.500038 | -0.012650 | -0.025299 | 0.512689 |
| iTransformer | 1.000 | 1.000077 | -0.016980 | -0.016979 | 1.017057 |
| iTransformer | 2.000 | 2.000154 | -0.009017 | -0.004508 | 2.009171 |
| iTransformer | 3.000 | 3.000231 | -0.001010 | -0.000337 | 3.001241 |
| iTransformer | 4.000 | 4.000308 | 0.003520 | 0.000880 | 3.996788 |
| iTransformer | 5.000 | 5.000384 | 0.006518 | 0.001304 | 4.993866 |
| iTransformer_CRR_FT01 | -5.000 | -5.000384 | -4.164257 | 0.832787 | 0.847925 |
| iTransformer_CRR_FT01 | -4.000 | -4.000308 | -3.943097 | 0.985698 | 0.353082 |
| iTransformer_CRR_FT01 | -3.000 | -3.000231 | -3.508844 | 1.169525 | 0.554108 |
| iTransformer_CRR_FT01 | -2.000 | -2.000154 | -2.708868 | 1.354330 | 0.721101 |
| iTransformer_CRR_FT01 | -1.000 | -1.000077 | -1.473432 | 1.473319 | 0.484218 |
| iTransformer_CRR_FT01 | -0.500 | -0.500038 | -0.752714 | 1.505313 | 0.259618 |
| iTransformer_CRR_FT01 | 0.500 | 0.500038 | 0.756140 | 1.512165 | 0.261540 |
| iTransformer_CRR_FT01 | 1.000 | 1.000077 | 1.483115 | 1.483001 | 0.491435 |
| iTransformer_CRR_FT01 | 2.000 | 2.000154 | 2.720538 | 1.360165 | 0.734940 |
| iTransformer_CRR_FT01 | 3.000 | 3.000231 | 3.508824 | 1.169518 | 0.572334 |
| iTransformer_CRR_FT01 | 4.000 | 4.000308 | 3.933315 | 0.983253 | 0.384234 |
| iTransformer_CRR_FT01 | 5.000 | 5.000384 | 4.149043 | 0.829745 | 0.871410 |
| iTransformer_CRR_FT03 | -5.000 | -5.000384 | -4.221804 | 0.844296 | 0.795896 |
| iTransformer_CRR_FT03 | -4.000 | -4.000308 | -3.970231 | 0.992482 | 0.341709 |
| iTransformer_CRR_FT03 | -3.000 | -3.000231 | -3.496679 | 1.165470 | 0.546690 |
| iTransformer_CRR_FT03 | -2.000 | -2.000154 | -2.655657 | 1.327726 | 0.678758 |
| iTransformer_CRR_FT03 | -1.000 | -1.000077 | -1.416000 | 1.415891 | 0.436724 |
| iTransformer_CRR_FT03 | -0.500 | -0.500038 | -0.718319 | 1.436527 | 0.230760 |
| iTransformer_CRR_FT03 | 0.500 | 0.500038 | 0.721409 | 1.442708 | 0.230778 |
| iTransformer_CRR_FT03 | 1.000 | 1.000077 | 1.425016 | 1.424906 | 0.438474 |
| iTransformer_CRR_FT03 | 2.000 | 2.000154 | 2.666174 | 1.332984 | 0.685266 |
| iTransformer_CRR_FT03 | 3.000 | 3.000231 | 3.496699 | 1.165477 | 0.555535 |
| iTransformer_CRR_FT03 | 4.000 | 4.000308 | 3.960384 | 0.990020 | 0.359291 |
| iTransformer_CRR_FT03 | 5.000 | 5.000384 | 4.205089 | 0.840953 | 0.815873 |
