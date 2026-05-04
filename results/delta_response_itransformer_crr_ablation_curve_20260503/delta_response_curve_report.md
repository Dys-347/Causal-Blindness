# Delta-Response Curve Evaluation

Generated at: `20260503_213823`

The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. We intervene on the last historical cause value and measure the horizon-1 target response.

## Model-Level Summary

| Model | Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |
|---|---:|---:|---:|---:|
| iTransformer | -0.000843 | -0.227592 | 2.591262 | -0.009014 |
| iTransformer_FT_PREDONLY | -0.000056 | -0.012579 | 2.590553 | -0.010300 |
| iTransformer_CRR_FT01_RESPONLY | 0.996859 | 0.986529 | 0.511245 | 1.217759 |
| iTransformer_CRR_FT01 | 0.983589 | 0.983902 | 0.544662 | 1.221568 |

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
| iTransformer_FT_PREDONLY | -5.000 | -5.000384 | -0.003050 | 0.000610 | 4.997335 |
| iTransformer_FT_PREDONLY | -4.000 | -4.000308 | 0.000561 | -0.000140 | 4.000868 |
| iTransformer_FT_PREDONLY | -3.000 | -3.000231 | 0.006527 | -0.002175 | 3.006757 |
| iTransformer_FT_PREDONLY | -2.000 | -2.000154 | 0.016407 | -0.008203 | 2.016560 |
| iTransformer_FT_PREDONLY | -1.000 | -1.000077 | 0.026771 | -0.026769 | 1.026848 |
| iTransformer_FT_PREDONLY | -0.500 | -0.500038 | 0.018588 | -0.037174 | 0.518627 |
| iTransformer_FT_PREDONLY | 0.500 | 0.500038 | -0.015774 | -0.031545 | 0.515812 |
| iTransformer_FT_PREDONLY | 1.000 | 1.000077 | -0.019961 | -0.019960 | 1.020038 |
| iTransformer_FT_PREDONLY | 2.000 | 2.000154 | -0.008303 | -0.004151 | 2.008457 |
| iTransformer_FT_PREDONLY | 3.000 | 3.000231 | 0.002458 | 0.000819 | 2.997773 |
| iTransformer_FT_PREDONLY | 4.000 | 4.000308 | 0.009326 | 0.002331 | 3.990981 |
| iTransformer_FT_PREDONLY | 5.000 | 5.000384 | 0.013806 | 0.002761 | 4.986578 |
| iTransformer_CRR_FT01_RESPONLY | -5.000 | -5.000384 | -4.276543 | 0.855243 | 0.749087 |
| iTransformer_CRR_FT01_RESPONLY | -4.000 | -4.000308 | -3.996317 | 0.999002 | 0.335524 |
| iTransformer_CRR_FT01_RESPONLY | -3.000 | -3.000231 | -3.505861 | 1.168531 | 0.553232 |
| iTransformer_CRR_FT01_RESPONLY | -2.000 | -2.000154 | -2.676341 | 1.338068 | 0.691066 |
| iTransformer_CRR_FT01_RESPONLY | -1.000 | -1.000077 | -1.452846 | 1.452735 | 0.464241 |
| iTransformer_CRR_FT01_RESPONLY | -0.500 | -0.500038 | -0.741572 | 1.483030 | 0.247789 |
| iTransformer_CRR_FT01_RESPONLY | 0.500 | 0.500038 | 0.745014 | 1.489914 | 0.248630 |
| iTransformer_CRR_FT01_RESPONLY | 1.000 | 1.000077 | 1.461097 | 1.460984 | 0.466272 |
| iTransformer_CRR_FT01_RESPONLY | 2.000 | 2.000154 | 2.683769 | 1.341781 | 0.693148 |
| iTransformer_CRR_FT01_RESPONLY | 3.000 | 3.000231 | 3.508160 | 1.169297 | 0.561517 |
| iTransformer_CRR_FT01_RESPONLY | 4.000 | 4.000308 | 3.996762 | 0.999114 | 0.368999 |
| iTransformer_CRR_FT01_RESPONLY | 5.000 | 5.000384 | 4.277363 | 0.855407 | 0.755437 |
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
