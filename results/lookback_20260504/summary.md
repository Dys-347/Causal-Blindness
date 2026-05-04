# Look-back Window Robustness 2026-05-04

DUET-Mix baseline is trained with different history lengths on the same controlled synthetic benchmark. The prediction length is fixed to 96 and the expected H1 response for `delta=+5` remains about 5.000.

| Seq len | Target MSE | Target MAE | Pred. H1 | H1 IRE | H1 Slope | Target-zero | Curve slope | Curve IRE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.3329 | 0.4645 | 0.0883 | 4.9760 | 0.0049 | 0.7988 | 0.0048 | 2.5557 |
| 96 | 0.0248 | 0.1278 | 0.0568 | 4.9811 | 0.0038 | 0.7776 | 0.0007 | 2.5822 |
| 192 | 0.0157 | 0.1019 | 0.0226 | 5.0021 | -0.0003 | 0.8024 | -0.0012 | 2.5912 |
| 336 | 0.0205 | 0.1155 | 0.0244 | 4.9979 | 0.0005 | 0.7812 | -0.0013 | 2.5891 |

## Interpretation

The key diagnostic is whether `H1 Slope` remains near zero while `Target-zero` stays large. That pattern means the model still relies on target-history shortcuts rather than the last-step driver, even when the look-back window changes.
