# Functional Input Sensitivity

All values are central-difference absolute sensitivities of the horizon-1 target prediction with respect to small standardized input perturbations.

| Model | Cause last | Target last | Cause/Target last | Cause history | Target history | Cause/Target history |
|---|---:|---:|---:|---:|---:|---:|
| iTransformer | 0.03176947 | 0.23854906 | 0.133178 | 0.00000417 | 1.00000031 | 0.000004 |
| iTransformer_MSE_FT | 0.03958463 | 0.20145304 | 0.196496 | 0.00000675 | 0.99999905 | 0.000007 |
| iTransformer_RIR_FT01 | 1.50753835 | 1.34168955 | 1.123612 | 0.00002018 | 1.00000399 | 0.000020 |
| iTransformer_RIR_FT03 | 1.43441542 | 1.40193146 | 1.023171 | 0.00001814 | 1.00000030 | 0.000018 |
| PatchTST | 0.00000000 | 0.53616806 | 0.000000 | 0.00000000 | 0.99999986 | 0.000000 |
| Crossformer | 0.00007474 | 0.04827047 | 0.001548 | 0.00123381 | 0.29456369 | 0.004189 |
| TimeMixer | 0.00000000 | 0.17520702 | 0.000000 | 0.00000000 | 0.99999964 | 0.000000 |
