# ETTh2 Augmented Diagnostic Summary

Run ID: `ett_augmented_20260504`

Model: iTransformer

Dataset: ETTh2 augmented with `Target_RIR[t] = 2 * HUFL[t-1] + seasonal(t) + noise`

Prediction length: 96

Seeds: `20260503`, `20260504`, `20260505`

## Mean +- Std

| Variant | All MSE | Raw OT MSE | Raw OT MAE | Semi MSE | Curve slope | Curve IRE | Curve ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.3217 +- 0.0021 | 0.1475 +- 0.0014 | 0.2959 +- 0.0014 | 0.4788 +- 0.0050 | 0.0083 +- 0.0028 | 2.5959 +- 0.0077 | 0.0100 +- 0.0032 |
| + RIR | 0.3245 +- 0.0061 | 0.1607 +- 0.0049 | 0.3095 +- 0.0055 | 0.4710 +- 0.0069 | 1.0033 +- 0.0215 | 0.5288 +- 0.0151 | 0.8987 +- 0.0103 |

## Per-Seed Results

| Seed | Variant | All MSE | Raw OT MSE | Raw OT MAE | Semi MSE | Curve slope | Curve IRE | Curve ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20260503 | Baseline | 0.3229 | 0.1485 | 0.2968 | 0.4806 | 0.0098 | 2.5920 | 0.0115 |
| 20260504 | Baseline | 0.3229 | 0.1481 | 0.2966 | 0.4826 | 0.0051 | 2.6048 | 0.0063 |
| 20260505 | Baseline | 0.3192 | 0.1459 | 0.2944 | 0.4731 | 0.0100 | 2.5910 | 0.0121 |
| 20260503 | + RIR | 0.3316 | 0.1594 | 0.3065 | 0.4782 | 1.0279 | 0.5424 | 0.9098 |
| 20260504 | + RIR | 0.3215 | 0.1662 | 0.3159 | 0.4644 | 0.9881 | 0.5314 | 0.8896 |
| 20260505 | + RIR | 0.3206 | 0.1567 | 0.3062 | 0.4703 | 0.9939 | 0.5126 | 0.8965 |

## Interpretation

ETTh2 confirms that the augmented-real diagnostic is not specific to ETTh1. The
baseline iTransformer again has a near-flat response curve for the appended causal
target, while RIR recovers an almost calibrated response curve.

Unlike the ETTh1 side-effect table, raw `OT` metrics are modestly worse under RIR
on ETTh2. This should be reported transparently as a side-effect diagnostic: the
response repair is strong, aggregate all-channel MSE changes only slightly, but
local collateral effects on an unrelated raw target can vary by dataset.
