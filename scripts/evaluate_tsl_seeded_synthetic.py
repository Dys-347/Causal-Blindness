import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

from train_tsl_seeded_synthetic import build_args, MODEL_OVERRIDES


def load_model(model_name, checkpoint_path, batch_size):
    use_gpu = torch.cuda.is_available()
    args = build_args(
        model_name=model_name,
        seed=20260505,
        batch_size=batch_size,
        learning_rate=0.0001,
        epochs=1,
        patience=1,
        data_path=os.environ.get("CAUSAL_R1_SYNTHETIC_CSV", "synthetic_multivariate.csv"),
    )
    args.use_gpu = use_gpu
    exp = Exp_Long_Term_Forecast(args)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=exp.device)
    cleaned = {}
    for key, value in state_dict.items():
        if "total_ops" in key or "total_params" in key:
            continue
        cleaned[key] = value
    exp.model.load_state_dict(cleaned)
    exp.model.eval()
    return exp, args


def make_decoder_inputs(args, batch_y, device):
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
    return dec_inp.float().to(device)


def forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark):
    dec_inp = make_decoder_inputs(args, batch_y, exp.device)
    with torch.no_grad():
        out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return out[:, -args.pred_len :, :]


def evaluate(exp, args, delta, max_batches=None):
    test_data, test_loader = data_provider(args, flag="test")
    scaler = test_data.scaler
    cause_scale = float(scaler.scale_[0])
    target_scale = float(scaler.scale_[-1])
    causal_gain = 2.0 * cause_scale / target_scale

    target_sq_err = 0.0
    target_abs_err = 0.0
    target_n = 0
    h1_pred_abs_sum = 0.0
    h1_true_abs_sum = 0.0
    h1_err_abs_sum = 0.0
    h1_pred_sq_sum = 0.0
    h1_true_sq_sum = 0.0
    h1_err_sq_sum = 0.0
    h1_pred_dot_true = 0.0
    h1_true_sq_denom = 0.0
    target_zero_mpd_sum = 0.0
    target_zero_n = 0
    num_batches = 0
    window_rows = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            true = batch_y[:, -args.pred_len :, :]

            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            target_pred = pred[:, :, -1]
            target_true = true[:, :, -1]
            diff = target_pred - target_true
            target_sq_err += torch.square(diff).sum().item()
            target_abs_err += torch.abs(diff).sum().item()
            target_n += diff.numel()

            x_cf = batch_x.clone()
            x_cf[:, -1, 0] = x_cf[:, -1, 0] + delta
            pred_cf = forward_model(exp, args, x_cf, batch_y, batch_x_mark, batch_y_mark)
            pred_change = pred_cf[:, 0, -1] - pred[:, 0, -1]
            expected = torch.full_like(pred_change, causal_gain * delta)
            err = pred_change - expected
            pred_np = pred_change.detach().cpu().numpy()
            expected_np = expected.detach().cpu().numpy()
            error_np = err.detach().cpu().numpy()
            for pred_value, expected_value, error_value in zip(pred_np, expected_np, error_np):
                sign_correct = int(np.sign(pred_value) == np.sign(expected_value)) if abs(expected_value) > 1e-12 else 1
                window_rows.append(
                    {
                        "window_index": len(window_rows),
                        "pred_change": float(pred_value),
                        "expected_change": float(expected_value),
                        "error": float(error_value),
                        "abs_error": float(abs(error_value)),
                        "sign_correct": sign_correct,
                    }
                )

            h1_pred_abs_sum += torch.abs(pred_change).sum().item()
            h1_true_abs_sum += torch.abs(expected).sum().item()
            h1_err_abs_sum += torch.abs(err).sum().item()
            h1_pred_sq_sum += torch.square(pred_change).sum().item()
            h1_true_sq_sum += torch.square(expected).sum().item()
            h1_err_sq_sum += torch.square(err).sum().item()
            h1_pred_dot_true += torch.sum(pred_change * expected).item()
            h1_true_sq_denom += torch.sum(expected * expected).item()

            x_zero = batch_x.clone()
            x_zero[:, :, -1] = 0.0
            pred_zero = forward_model(exp, args, x_zero, batch_y, batch_x_mark, batch_y_mark)
            target_zero_mpd_sum += torch.abs(pred_zero[:, 0, -1] - pred[:, 0, -1]).sum().item()
            target_zero_n += pred.shape[0]
            num_batches += 1

    target_mse = target_sq_err / max(target_n, 1)
    target_mae = target_abs_err / max(target_n, 1)
    pred_h1_abs_mean = h1_pred_abs_sum / max(target_zero_n, 1)
    true_h1_abs_mean = h1_true_abs_sum / max(target_zero_n, 1)
    h1_ire = h1_err_abs_sum / max(target_zero_n, 1)
    h1_slope = h1_pred_dot_true / max(h1_true_sq_denom, 1e-12)
    target_zero_mpd = target_zero_mpd_sum / max(target_zero_n, 1)
    pred_changes = np.asarray([row["pred_change"] for row in window_rows], dtype=np.float64)
    expected_changes = np.asarray([row["expected_change"] for row in window_rows], dtype=np.float64)
    if len(window_rows) > 0:
        sign_accuracy = float(np.mean(np.sign(pred_changes) == np.sign(expected_changes)))
        abs_expected = np.maximum(np.abs(expected_changes), 1e-12)
        response_ge_20pct = float(np.mean(np.abs(pred_changes) >= 0.2 * abs_expected))
        response_ge_50pct = float(np.mean(np.abs(pred_changes) >= 0.5 * abs_expected))
        pred_signed_mean = float(np.mean(pred_changes))
        pred_signed_std = float(np.std(pred_changes, ddof=1)) if len(pred_changes) > 1 else 0.0
        pred_q05, pred_q50, pred_q95 = [float(x) for x in np.quantile(pred_changes, [0.05, 0.5, 0.95])]
    else:
        sign_accuracy = float("nan")
        response_ge_20pct = float("nan")
        response_ge_50pct = float("nan")
        pred_signed_mean = float("nan")
        pred_signed_std = float("nan")
        pred_q05 = float("nan")
        pred_q50 = float("nan")
        pred_q95 = float("nan")

    metrics = {
        "target_mse": target_mse,
        "target_mae": target_mae,
        "pred_h1_abs_mean": pred_h1_abs_mean,
        "pred_h1_signed_mean": pred_signed_mean,
        "pred_h1_signed_std": pred_signed_std,
        "pred_h1_q05": pred_q05,
        "pred_h1_q50": pred_q50,
        "pred_h1_q95": pred_q95,
        "true_h1_abs_mean": true_h1_abs_mean,
        "h1_ire": h1_ire,
        "h1_slope": h1_slope,
        "h1_sign_accuracy": sign_accuracy,
        "h1_response_ge_20pct": response_ge_20pct,
        "h1_response_ge_50pct": response_ge_50pct,
        "target_zero_mpd": target_zero_mpd,
        "num_batches": num_batches,
        "num_test_windows": len(test_data),
        "causal_gain_scaled": causal_gain,
        "pred_h1_rms": math.sqrt(h1_pred_sq_sum / max(target_zero_n, 1)),
        "true_h1_rms": math.sqrt(h1_true_sq_sum / max(target_zero_n, 1)),
        "h1_ire_rmse": math.sqrt(h1_err_sq_sum / max(target_zero_n, 1)),
    }
    return metrics, window_rows


def write_outputs(output_dir, record, window_rows):
    os.makedirs(output_dir, exist_ok=True)
    json_path = Path(output_dir) / "seeded_synthetic_eval.json"
    csv_path = Path(output_dir) / "seeded_synthetic_eval.csv"
    md_path = Path(output_dir) / "seeded_synthetic_eval.md"
    window_csv_path = Path(output_dir) / "window_h1_response.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        writer.writeheader()
        writer.writerow(record)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Seeded TSL Synthetic Evaluation\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        for key, value in record.items():
            if isinstance(value, float):
                f.write(f"| {key} | {value:.6f} |\n")
            else:
                f.write(f"| {key} | {value} |\n")
    with open(window_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["window_index", "pred_change", "expected_change", "error", "abs_error", "sign_correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_OVERRIDES.keys()))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--max-batches", type=int, default=None)
    args_cli = parser.parse_args()

    exp, args = load_model(args_cli.model, args_cli.checkpoint_path, args_cli.batch_size)
    metrics, window_rows = evaluate(exp, args, delta=args_cli.delta, max_batches=args_cli.max_batches)
    record = {
        "model": args_cli.model,
        "variant": args_cli.variant,
        "checkpoint_path": args_cli.checkpoint_path,
        **metrics,
    }
    write_outputs(args_cli.output_dir, record, window_rows)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
