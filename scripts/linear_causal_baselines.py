import argparse
import csv
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd


def split_and_standardize(df, seq_len):
    value_cols = [col for col in df.columns if col != "date"]
    cause_col = "Cause_Var"
    target_col = "Target_Var"
    distractor_cols = [col for col in value_cols if col not in [cause_col, target_col]]
    ordered_cols = [cause_col] + distractor_cols + [target_col]

    values = df[ordered_cols].to_numpy(dtype=np.float64)
    n = len(values)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test

    train_values = values[:n_train]
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    values_std = (values - mean) / scale

    borders = {
        "train": (0, n_train),
        "val": (n_train - seq_len, n_train + n_val),
        "test": (n - n_test - seq_len, n),
    }
    return values_std, ordered_cols, mean, scale, borders


def make_windows(values, border, seq_len, pred_len):
    start, end = border
    x_list = []
    y_h1 = []
    last_valid = end - seq_len - pred_len + 1
    for idx in range(start, last_valid):
        x = values[idx : idx + seq_len]
        y = values[idx + seq_len, -1]
        x_list.append(x)
        y_h1.append(y)
    return np.stack(x_list, axis=0), np.asarray(y_h1, dtype=np.float64)


def design_matrix(windows, kind):
    if kind == "AR_Y_last":
        feats = windows[:, -1, [-1]]
    elif kind == "ARX_C_last":
        feats = windows[:, -1, [0]]
    elif kind == "ARX_CY_last":
        feats = windows[:, -1, [0, -1]]
    elif kind == "VARX_all_last":
        feats = windows[:, -1, :]
    elif kind == "VARX_full_history":
        feats = windows.reshape(windows.shape[0], -1)
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")
    return np.concatenate([feats, np.ones((feats.shape[0], 1), dtype=feats.dtype)], axis=1)


def fit_ridge(x, y, ridge):
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def predict(windows, kind, weights):
    return design_matrix(windows, kind) @ weights


def eval_obs(y_true, y_pred):
    err = y_pred - y_true
    return {
        "h1_target_mse": float(np.mean(err * err)),
        "h1_target_mae": float(np.mean(np.abs(err))),
    }


def eval_delta_curve(windows, kind, weights, deltas, causal_gain):
    base_pred = predict(windows, kind, weights)
    rows = []
    for delta in deltas:
        variant = windows.copy()
        variant[:, -1, 0] += float(delta)
        pred_delta = predict(variant, kind, weights) - base_pred
        true_delta = np.full_like(pred_delta, causal_gain * float(delta))
        err = pred_delta - true_delta
        denom = float(np.sum(true_delta * true_delta))
        slope = float(np.sum(pred_delta * true_delta) / denom) if denom > 1e-12 else float("nan")
        if pred_delta.size > 1 and np.std(pred_delta) > 1e-12 and np.std(true_delta) > 1e-12:
            corr = float(np.corrcoef(pred_delta, true_delta)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "delta": float(delta),
                "expected_mean": float(np.mean(true_delta)),
                "pred_mean": float(np.mean(pred_delta)),
                "pred_abs_mean": float(np.mean(np.abs(pred_delta))),
                "ire_mae": float(np.mean(np.abs(err))),
                "ire_rmse": float(math.sqrt(np.mean(err * err))),
                "response_ratio": float(np.mean(pred_delta) / np.mean(true_delta))
                if abs(np.mean(true_delta)) > 1e-12
                else float("nan"),
                "response_slope": slope,
                "response_corr": corr,
                "sign_accuracy_on_nonzero_true": float(np.mean(np.sign(pred_delta) == np.sign(true_delta))),
            }
        )
    return rows


def summarize_curve(rows):
    pred = np.asarray([row["pred_mean"] for row in rows if abs(row["delta"]) > 1e-12], dtype=np.float64)
    true = np.asarray([row["expected_mean"] for row in rows if abs(row["delta"]) > 1e-12], dtype=np.float64)
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    corr = float(np.corrcoef(pred, true)[0, 1]) if pred.size > 1 and np.std(pred) > 1e-12 else float("nan")
    return {
        "curve_slope_from_means": slope,
        "curve_corr_from_means": corr,
        "curve_ire_mae_mean": float(np.mean([row["ire_mae"] for row in rows if abs(row["delta"]) > 1e-12])),
        "curve_response_ratio_mean": float(
            np.mean([row["response_ratio"] for row in rows if abs(row["delta"]) > 1e-12])
        ),
        "num_delta_points": int(sum(1 for row in rows if abs(row["delta"]) > 1e-12)),
    }


def write_outputs(results, output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "linear_causal_baselines_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    model_rows = []
    delta_rows = []
    for result in results:
        model_row = {
            "model": result["model"],
            "ridge": result["ridge"],
            "h1_target_mse": result["obs"]["h1_target_mse"],
            "h1_target_mae": result["obs"]["h1_target_mae"],
            "coef_cause_last": result["coef_cause_last"],
        }
        model_row.update({f"full_{k}": v for k, v in result["full_summary"].items()})
        model_row.update({f"small_{k}": v for k, v in result["small_summary"].items()})
        model_rows.append(model_row)
        for scope in ["full", "small"]:
            for row in result[f"{scope}_rows"]:
                out = {"model": result["model"], "scope": scope}
                out.update(row)
                delta_rows.append(out)

    with open(os.path.join(output_dir, "linear_causal_baselines_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = list(model_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_rows)

    with open(os.path.join(output_dir, "linear_causal_baselines_per_delta.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = list(delta_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(delta_rows)

    with open(os.path.join(output_dir, "linear_causal_baselines_report.md"), "w", encoding="utf-8") as f:
        f.write("# Linear Causal Baseline Sanity Check\n\n")
        f.write(f"Generated at: `{metadata['timestamp']}`\n\n")
        f.write("The CSV is reordered to `Cause, Distractors, Target`, then globally standardized with training-set statistics.\n\n")
        f.write(f"Scaled causal gain: `{metadata['causal_gain_scaled']:.9f}`\n\n")
        f.write("| Model | H1 MSE | H1 MAE | Cause Coef. | Full Slope | Full IRE | Small Slope | Small IRE |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in model_rows:
            f.write(
                "| {model} | {mse:.6f} | {mae:.6f} | {coef:.6f} | {fs:.6f} | {fi:.6f} | {ss:.6f} | {si:.6f} |\n".format(
                    model=row["model"],
                    mse=row["h1_target_mse"],
                    mae=row["h1_target_mae"],
                    coef=row["coef_cause_last"],
                    fs=row["full_curve_slope_from_means"],
                    fi=row["full_curve_ire_mae_mean"],
                    ss=row["small_curve_slope_from_means"],
                    si=row["small_curve_ire_mae_mean"],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset/synthetic_multivariate.csv")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=["AR_Y_last", "ARX_C_last", "ARX_CY_last", "VARX_all_last"])
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--full-deltas", nargs="+", type=float, default=[-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5])
    parser.add_argument("--small-deltas", nargs="+", type=float, default=[-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1])
    parser.add_argument("--output-dir", default="causal_r1_linear_baselines_20260504")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    values, ordered_cols, mean, scale, borders = split_and_standardize(df, seq_len=args.seq_len)
    x_train, y_train = make_windows(values, borders["train"], args.seq_len, args.pred_len)
    x_test, y_test = make_windows(values, borders["test"], args.seq_len, args.pred_len)
    causal_gain = float(2.0 * scale[0] / scale[-1])

    results = []
    for model in args.models:
        x_design = design_matrix(x_train, model)
        weights = fit_ridge(x_design, y_train, ridge=args.ridge)
        y_pred = predict(x_test, model, weights)
        full_rows = eval_delta_curve(x_test, model, weights, args.full_deltas, causal_gain)
        small_rows = eval_delta_curve(x_test, model, weights, args.small_deltas, causal_gain)
        cause_coef = 0.0
        if model in ["ARX_C_last", "ARX_CY_last", "VARX_all_last"]:
            cause_coef = float(weights[0])
        elif model == "VARX_full_history":
            cause_coef = float(weights[(args.seq_len - 1) * values.shape[1] + 0])
        results.append(
            {
                "model": model,
                "ridge": args.ridge,
                "weights_shape": list(weights.shape),
                "coef_cause_last": cause_coef,
                "obs": eval_obs(y_test, y_pred),
                "full_rows": full_rows,
                "full_summary": summarize_curve(full_rows),
                "small_rows": small_rows,
                "small_summary": summarize_curve(small_rows),
            }
        )

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "data_path": args.data_path,
        "ordered_cols": ordered_cols,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "train_border": borders["train"],
        "test_border": borders["test"],
        "num_train_windows": int(x_train.shape[0]),
        "num_test_windows": int(x_test.shape[0]),
        "causal_gain_scaled": causal_gain,
        "full_deltas": args.full_deltas,
        "small_deltas": args.small_deltas,
    }
    print(json.dumps(metadata, indent=2))
    for result in results:
        print(
            "{model}: H1_MSE={mse:.6f}, cause_coef={coef:.6f}, small_slope={ss:.6f}, full_slope={fs:.6f}".format(
                model=result["model"],
                mse=result["obs"]["h1_target_mse"],
                coef=result["coef_cause_last"],
                ss=result["small_summary"]["curve_slope_from_means"],
                fs=result["full_summary"]["curve_slope_from_means"],
            )
        )
    write_outputs(results, args.output_dir, metadata)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
