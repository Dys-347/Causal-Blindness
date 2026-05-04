import argparse
import csv
import json
import math
import os
import random
from collections import OrderedDict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_ett_augmented(
    source_path,
    output_path,
    beta,
    seasonal_scale,
    noise_scale,
    seed,
    cause_column,
    raw_target_column,
    semi_target_column,
):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(source_path)
    required = {"date", cause_column, raw_target_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source_path}: {missing}")

    date = pd.to_datetime(df["date"])
    hour = date.dt.hour.to_numpy(dtype=np.float32)
    dayofyear = date.dt.dayofyear.to_numpy(dtype=np.float32)
    seasonal = (
        np.sin(2.0 * np.pi * hour / 24.0)
        + 0.5 * np.cos(2.0 * np.pi * hour / 24.0)
        + 0.5 * np.sin(2.0 * np.pi * dayofyear / 365.25)
    )
    cause = df[cause_column].astype(np.float32).to_numpy()
    cause_lag = np.roll(cause, 1)
    cause_lag[0] = cause_lag[1]
    noise = rng.normal(loc=0.0, scale=noise_scale, size=len(df)).astype(np.float32)
    target = beta * cause_lag + seasonal_scale * seasonal + noise

    out = df.copy()
    out[semi_target_column] = target.astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    return {
        "source_path": source_path,
        "output_path": output_path,
        "beta": beta,
        "seasonal_scale": seasonal_scale,
        "noise_scale": noise_scale,
        "num_rows": int(len(out)),
        "columns": list(out.columns),
        "cause_column": cause_column,
        "raw_target_column": raw_target_column,
        "semi_target_column": semi_target_column,
        "structural_equation": f"{semi_target_column}[t] = beta * {cause_column}[t-1] + seasonal(date[t]) + noise",
    }


def infer_feature_columns(csv_path, target_column):
    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    if "date" not in columns:
        raise ValueError(f"Expected a date column in {csv_path}")
    if target_column not in columns:
        raise ValueError(f"Expected target column {target_column} in {csv_path}")
    feature_columns = [col for col in columns if col not in {"date", target_column}]
    return feature_columns + [target_column]


def build_args(cli, n_vars):
    return argparse.Namespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"{cli.dataset_name}_Augmented_iTransformer_{cli.variant}_{cli.seq_len}_{cli.pred_len}",
        model="iTransformer",
        des=f"{cli.dataset_name}_RIR_{cli.variant}",
        data=cli.data,
        root_path=os.path.dirname(cli.aug_data_path) + "/",
        data_path=os.path.basename(cli.aug_data_path),
        features="M",
        target=cli.semi_target_column,
        freq=cli.freq,
        checkpoints="./checkpoints/",
        seasonal_patterns="Monthly",
        seq_len=cli.seq_len,
        label_len=cli.label_len,
        pred_len=cli.pred_len,
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,
        d_model=cli.d_model,
        n_heads=cli.n_heads,
        e_layers=cli.e_layers,
        d_layers=1,
        d_ff=cli.d_ff,
        factor=3,
        dropout=cli.dropout,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        moving_avg=25,
        distil=True,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        inverse=False,
        num_workers=cli.num_workers,
        itr=1,
        train_epochs=cli.epochs,
        batch_size=cli.batch_size,
        patience=cli.patience,
        learning_rate=cli.lr,
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=torch.cuda.is_available(),
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0",
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        patch_len=16,
        stride=8,
    )


def make_decoder_inputs(args, batch_y, device):
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
    return dec_inp.float().to(device)


def forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark):
    dec_inp = make_decoder_inputs(args, batch_y, exp.device)
    out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return out[:, -args.pred_len :, :]


def sample_delta(batch_size, max_abs_delta, device, min_abs_ratio):
    signs = torch.where(torch.rand(batch_size, device=device) > 0.5, 1.0, -1.0)
    low = float(max_abs_delta) * float(min_abs_ratio)
    mags = low + (float(max_abs_delta) - low) * torch.rand(batch_size, device=device)
    return signs * mags


def eval_prediction(exp, args, loader, raw_ot_idx, semi_idx):
    exp.model.eval()
    stats = {
        "all_sq": 0.0,
        "all_abs": 0.0,
        "ot_sq": 0.0,
        "ot_abs": 0.0,
        "semi_sq": 0.0,
        "semi_abs": 0.0,
        "all_n": 0,
        "channel_n": 0,
    }
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            true = batch_y[:, -args.pred_len :, :]
            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            err = pred - true
            ot_err = err[:, :, raw_ot_idx]
            semi_err = err[:, :, semi_idx]
            stats["all_sq"] += torch.square(err).sum().item()
            stats["all_abs"] += torch.abs(err).sum().item()
            stats["ot_sq"] += torch.square(ot_err).sum().item()
            stats["ot_abs"] += torch.abs(ot_err).sum().item()
            stats["semi_sq"] += torch.square(semi_err).sum().item()
            stats["semi_abs"] += torch.abs(semi_err).sum().item()
            stats["all_n"] += err.numel()
            stats["channel_n"] += ot_err.numel()
    n_all = max(stats["all_n"], 1)
    n_channel = max(stats["channel_n"], 1)
    return {
        "all_mse": stats["all_sq"] / n_all,
        "all_mae": stats["all_abs"] / n_all,
        "raw_ot_mse": stats["ot_sq"] / n_channel,
        "raw_ot_mae": stats["ot_abs"] / n_channel,
        "semi_target_mse": stats["semi_sq"] / n_channel,
        "semi_target_mae": stats["semi_abs"] / n_channel,
        "num_all_points": int(stats["all_n"]),
        "num_channel_points": int(stats["channel_n"]),
    }


def curve_stats_init(delta, expected):
    return {"delta": float(delta), "expected_mean": float(expected), "pred": [], "true": []}


def safe_corr(a, b):
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def finalize_curve_row(stats):
    pred = np.concatenate(stats["pred"]) if stats["pred"] else np.array([], dtype=np.float32)
    true = np.concatenate(stats["true"]) if stats["true"] else np.array([], dtype=np.float32)
    err = pred - true
    expected = stats["expected_mean"]
    pred_mean = float(np.mean(pred)) if pred.size else float("nan")
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    return {
        "delta": stats["delta"],
        "expected_mean": expected,
        "pred_mean": pred_mean,
        "pred_abs_mean": float(np.mean(np.abs(pred))) if pred.size else float("nan"),
        "ire_mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "ire_rmse": float(np.sqrt(np.mean(err * err))) if err.size else float("nan"),
        "response_ratio": pred_mean / expected if abs(expected) > 1e-12 else float("nan"),
        "response_slope": slope,
        "response_corr": safe_corr(pred, true),
        "num_windows": int(pred.size),
    }


def summarize_curve(rows):
    valid = [row for row in rows if abs(row["delta"]) > 1e-12]
    pred = np.asarray([row["pred_mean"] for row in valid], dtype=np.float64)
    true = np.asarray([row["expected_mean"] for row in valid], dtype=np.float64)
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    return {
        "curve_slope_from_means": slope,
        "curve_corr_from_means": safe_corr(pred, true),
        "curve_ire_mae_mean": float(np.mean([row["ire_mae"] for row in valid])),
        "curve_response_ratio_mean": float(np.mean([row["response_ratio"] for row in valid])),
        "num_delta_points": len(valid),
    }


def evaluate_delta_curve(exp, args, loader, cause_idx, semi_idx, causal_gain, deltas):
    exp.model.eval()
    stats = {float(delta): curve_stats_init(delta, causal_gain * float(delta)) for delta in deltas}
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred_h1 = pred[:, 0, semi_idx]
            for delta in deltas:
                delta = float(delta)
                x_cf = batch_x.clone()
                x_cf[:, -1, cause_idx] += delta
                pred_cf = forward_model(exp, args, x_cf, batch_y, batch_x_mark, batch_y_mark)
                pred_change = pred_cf[:, 0, semi_idx] - pred_h1
                true_change = torch.full_like(pred_change, causal_gain * delta)
                stats[delta]["pred"].append(pred_change.detach().cpu().numpy().reshape(-1))
                stats[delta]["true"].append(true_change.detach().cpu().numpy().reshape(-1))
    rows = [finalize_curve_row(stats[float(delta)]) for delta in deltas]
    return {"rows": rows, "summary": summarize_curve(rows)}


def save_curve_plot(curve, output_dir, label, metadata):
    rows = sorted(curve["rows"], key=lambda row: row["delta"])
    x = [row["delta"] for row in rows]
    expected = [row["expected_mean"] for row in rows]
    pred = [row["pred_mean"] for row in rows]
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(x, expected, color="black", linestyle="--", marker="o", label="Expected")
    plt.plot(x, pred, marker="o", label=label)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel(f"{metadata['cause_column']} intervention delta on last historical step")
    plt.ylabel(f"Predicted horizon-1 {metadata['semi_target_column']} change")
    plt.title(f"{metadata['dataset_name']} Semi-Synthetic Delta-Response ({label})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metadata['dataset_name'].lower()}_delta_response_curve.png"), dpi=180)
    plt.savefig(os.path.join(output_dir, "etth1_delta_response_curve.png"), dpi=180)
    plt.close()


def write_outputs(output_dir, metadata, train_log, prediction_metrics, curve_result):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "etth1_rir_side_effect_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "train_log": train_log,
                "prediction_metrics": prediction_metrics,
                "delta_curve": curve_result,
            },
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "prediction_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_metrics.keys()))
        writer.writeheader()
        writer.writerow(prediction_metrics)

    with open(os.path.join(output_dir, "delta_response_curve.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "delta",
            "expected_mean",
            "pred_mean",
            "pred_abs_mean",
            "response_ratio",
            "ire_mae",
            "ire_rmse",
            "response_slope",
            "response_corr",
            "num_windows",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_result["rows"])

    save_curve_plot(curve_result, output_dir, metadata["variant"], metadata)
    curve = curve_result["summary"]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {metadata['dataset_name']} RIR Side-Effect Evaluation\n\n")
        f.write("## Prediction Metrics\n\n")
        f.write("| Variant | All MSE | All MAE | Raw OT MSE | Raw OT MAE | Semi Target MSE | Semi Target MAE |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| {variant} | {all_mse:.6f} | {all_mae:.6f} | {ot_mse:.6f} | {ot_mae:.6f} | {semi_mse:.6f} | {semi_mae:.6f} |\n\n".format(
                variant=metadata["variant"],
                all_mse=prediction_metrics["all_mse"],
                all_mae=prediction_metrics["all_mae"],
                ot_mse=prediction_metrics["raw_ot_mse"],
                ot_mae=prediction_metrics["raw_ot_mae"],
                semi_mse=prediction_metrics["semi_target_mse"],
                semi_mae=prediction_metrics["semi_target_mae"],
            )
        )
        f.write("## Response Curve Summary\n\n")
        f.write("| Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |\n")
        f.write("|---:|---:|---:|---:|\n")
        f.write(
            "| {slope:.6f} | {corr:.6f} | {ire:.6f} | {ratio:.6f} |\n".format(
                slope=curve["curve_slope_from_means"],
                corr=curve["curve_corr_from_means"],
                ire=curve["curve_ire_mae_mean"],
                ratio=curve["curve_response_ratio_mean"],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--data", default=None)
    parser.add_argument("--freq", default=None)
    parser.add_argument("--source-data-path", default="dataset/ETT-small/ETTh1.csv")
    parser.add_argument("--aug-data-path", default="dataset/ETT-small/ETTh1_RIR_augmented.csv")
    parser.add_argument("--make-data", action="store_true")
    parser.add_argument("--variant", choices=["baseline", "rir"], default="baseline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--cause-column", default="HUFL")
    parser.add_argument("--raw-target-column", default="OT")
    parser.add_argument("--semi-target-column", default="Target_RIR")
    parser.add_argument("--seasonal-scale", type=float, default=2.0)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--lambda-resp", type=float, default=0.05)
    parser.add_argument("--lambda-dist", type=float, default=0.005)
    parser.add_argument("--selection-pred-weight", type=float, default=0.1)
    parser.add_argument("--selection-dist-weight", type=float, default=0.05)
    parser.add_argument("--min-abs-delta-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--num-workers", type=int, default=0)
    cli = parser.parse_args()

    if cli.data is None:
        cli.data = cli.dataset_name
    if cli.freq is None:
        cli.freq = "t" if cli.dataset_name.startswith("ETTm") else "h"

    set_seed(cli.seed)
    data_info = None
    if cli.make_data or not os.path.exists(cli.aug_data_path):
        data_info = make_ett_augmented(
            source_path=cli.source_data_path,
            output_path=cli.aug_data_path,
            beta=cli.beta,
            seasonal_scale=cli.seasonal_scale,
            noise_scale=cli.noise_scale,
            seed=cli.seed,
            cause_column=cli.cause_column,
            raw_target_column=cli.raw_target_column,
            semi_target_column=cli.semi_target_column,
        )

    feature_columns = infer_feature_columns(cli.aug_data_path, cli.semi_target_column)
    n_vars = len(feature_columns)
    args = build_args(cli, n_vars)
    exp = Exp_Long_Term_Forecast(args)
    train_data, train_loader = data_provider(args, flag="train")
    val_data, val_loader = data_provider(args, flag="val")
    test_data, test_loader = data_provider(args, flag="test")

    cause_idx = feature_columns.index(cli.cause_column)
    raw_ot_idx = feature_columns.index(cli.raw_target_column)
    semi_idx = feature_columns.index(cli.semi_target_column)
    cause_scale = float(train_data.scaler.scale_[cause_idx])
    semi_scale = float(train_data.scaler.scale_[semi_idx])
    causal_gain = cli.beta * cause_scale / semi_scale

    criterion = nn.MSELoss()
    optimizer = optim.Adam(exp.model.parameters(), lr=cli.lr)
    best_score = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    train_log = []

    for epoch in range(1, cli.epochs + 1):
        exp.model.train()
        running = {"loss": [], "pred_loss": [], "resp_loss": [], "dist_loss": [], "slope": []}
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            true = batch_y[:, -args.pred_len :, :]

            optimizer.zero_grad()
            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred_loss = criterion(pred, true)
            resp_loss = torch.zeros((), device=exp.device)
            dist_loss = torch.zeros((), device=exp.device)
            slope = float("nan")

            if cli.variant == "rir":
                delta = sample_delta(batch_x.shape[0], cli.delta, exp.device, cli.min_abs_delta_ratio)
                x_cf = batch_x.clone()
                x_cf[:, -1, cause_idx] += delta
                pred_cf = forward_model(exp, args, x_cf, batch_y, batch_x_mark, batch_y_mark)
                pred_change = pred_cf[:, 0, semi_idx] - pred[:, 0, semi_idx]
                expected = causal_gain * delta
                resp_loss = criterion(pred_change, expected)

                dist_delta = sample_delta(batch_x.shape[0], cli.delta, exp.device, cli.min_abs_delta_ratio)
                x_dist = batch_x.clone()
                x_dist[:, -1, 1:semi_idx] += dist_delta.view(-1, 1)
                pred_dist = forward_model(exp, args, x_dist, batch_y, batch_x_mark, batch_y_mark)
                dist_change = pred_dist[:, 0, semi_idx] - pred[:, 0, semi_idx]
                dist_loss = criterion(dist_change, torch.zeros_like(dist_change))
                denom = torch.sum(expected.detach() * expected.detach()).item()
                slope = (
                    torch.sum(pred_change.detach() * expected.detach()).item() / denom
                    if denom > 1e-12
                    else float("nan")
                )

            loss = pred_loss + cli.lambda_resp * resp_loss + cli.lambda_dist * dist_loss
            loss.backward()
            optimizer.step()

            running["loss"].append(loss.item())
            running["pred_loss"].append(pred_loss.item())
            running["resp_loss"].append(resp_loss.item())
            running["dist_loss"].append(dist_loss.item())
            running["slope"].append(slope)

        pred_metrics_val = eval_prediction(exp, args, val_loader, raw_ot_idx, semi_idx)
        curve_val = evaluate_delta_curve(exp, args, val_loader, cause_idx, semi_idx, causal_gain, [-5, -3, -1, 1, 3, 5])
        if cli.variant == "rir":
            score = (
                curve_val["summary"]["curve_ire_mae_mean"]
                + cli.selection_pred_weight * pred_metrics_val["all_mse"]
            )
        else:
            score = pred_metrics_val["all_mse"]

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running["loss"])),
            "train_pred_loss": float(np.mean(running["pred_loss"])),
            "train_resp_loss": float(np.mean(running["resp_loss"])),
            "train_dist_loss": float(np.mean(running["dist_loss"])),
            "train_response_slope": float(np.nanmean(running["slope"])),
            "val_prediction": pred_metrics_val,
            "val_curve_summary": curve_val["summary"],
            "selection_score": float(score),
        }
        train_log.append(record)
        print(json.dumps(record, indent=2), flush=True)

        if score < best_score:
            best_score = float(score)
            best_epoch = epoch
            bad_epochs = 0
            best_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in exp.model.state_dict().items())
            print(f"Saved in-memory best epoch {epoch} with score={score:.6f}", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= cli.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        exp.model.load_state_dict(best_state)

    prediction_metrics = eval_prediction(exp, args, test_loader, raw_ot_idx, semi_idx)
    deltas = [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]
    curve_result = evaluate_delta_curve(exp, args, test_loader, cause_idx, semi_idx, causal_gain, deltas)

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset_name": cli.dataset_name,
        "variant": cli.variant,
        "data_info": data_info,
        "source_data_path": cli.source_data_path,
        "aug_data_path": cli.aug_data_path,
        "feature_columns": feature_columns,
        "cause_column": cli.cause_column,
        "raw_target_column": cli.raw_target_column,
        "semi_target_column": cli.semi_target_column,
        "config": vars(cli),
        "device": str(exp.device),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "cause_idx": cause_idx,
        "raw_ot_idx": raw_ot_idx,
        "semi_target_idx": semi_idx,
        "cause_scale": cause_scale,
        "semi_target_scale": semi_scale,
        "causal_gain_scaled": causal_gain,
        "note": "Raw target metrics evaluate side effects on the original benchmark target. Response metrics evaluate the appended semi-synthetic channel.",
    }
    write_outputs(cli.output_dir, metadata, train_log, prediction_metrics, curve_result)
    curve = curve_result["summary"]
    print(
        "[{}-{}] all_mse={:.6f}, raw_target_mse={:.6f}, raw_target_mae={:.6f}, "
        "semi_mse={:.6f}, curve_slope={:.6f}, curve_corr={:.6f}, curve_ire={:.6f}".format(
            cli.dataset_name,
            cli.variant,
            prediction_metrics["all_mse"],
            prediction_metrics["raw_ot_mse"],
            prediction_metrics["raw_ot_mae"],
            prediction_metrics["semi_target_mse"],
            curve["curve_slope_from_means"],
            curve["curve_corr_from_means"],
            curve["curve_ire_mae_mean"],
        ),
        flush=True,
    )
    print(f"Saved {cli.dataset_name} RIR side-effect results to: {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
