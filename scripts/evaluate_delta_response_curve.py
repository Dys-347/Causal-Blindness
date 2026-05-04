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
import torch

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


MODEL_CONFIGS = {
    "DLinear": {
        "checkpoint": (
            "checkpoints/"
            "long_term_forecast_Synthetic_Multi_DLinear_96_96_DLinear_SyntheticMulti_"
            "ftM_sl96_ll48_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_"
            "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
        ),
    },
    "PatchTST": {
        "checkpoint": (
            "checkpoints/"
            "long_term_forecast_Synthetic_Multi_PatchTST_96_96_PatchTST_SyntheticMulti_"
            "ftM_sl96_ll48_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_"
            "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
        ),
    },
    "iTransformer": {
        "checkpoint": (
            "checkpoints/"
            "long_term_forecast_Synthetic_Multi_iTransformer_96_96_iTransformer_SyntheticMulti_"
            "ftM_sl96_ll48_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_"
            "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
        ),
    },
    "iTransformer_CRR": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1/checkpoint.pth",
    },
    "iTransformer_CRR_FT01": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1_ft_lam01/checkpoint.pth",
    },
    "iTransformer_CRR_FT03": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1_ft_lam03/checkpoint.pth",
    },
    "iTransformer_FT_PREDONLY": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_ft_predonly/checkpoint.pth",
    },
    "iTransformer_CRR_FT01_RESPONLY": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1_ft_lam01_responly/checkpoint.pth",
    },
    "Crossformer": {
        "checkpoint": (
            "checkpoints/"
            "long_term_forecast_Synthetic_Multi_Crossformer_96_96_Crossformer_SyntheticMulti_"
            "ftM_sl96_ll48_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_"
            "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
        ),
    },
    "TimeMixer": {
        "checkpoint": (
            "checkpoints/"
            "long_term_forecast_Synthetic_Multi_TimeMixer_96_96_TimeMixer_SyntheticMulti_"
            "ftM_sl96_ll0_pl96_dm16_nh4_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_"
            "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
        ),
        "overrides": {
            "label_len": 0,
            "d_model": 16,
            "d_ff": 32,
            "down_sampling_layers": 3,
            "down_sampling_method": "avg",
            "down_sampling_window": 2,
            "learning_rate": 0.01,
        },
    },
}


def build_args(model_name, batch_size, use_gpu, overrides=None):
    args = argparse.Namespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id=f"Synthetic_Multi_{model_name}_DeltaResponseCurve",
        model=model_name,
        des="Delta_Response_Curve",
        data="SyntheticMulti",
        root_path="./dataset/",
        data_path="synthetic_multivariate.csv",
        features="M",
        target="Target_Var",
        freq="h",
        checkpoints="./checkpoints/",
        seasonal_patterns="Monthly",
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=21,
        dec_in=21,
        c_out=21,
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=3,
        dropout=0.05,
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
        num_workers=0,
        itr=1,
        train_epochs=10,
        batch_size=batch_size,
        patience=3,
        learning_rate=0.0001,
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=use_gpu,
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
    for key, value in (overrides or {}).items():
        setattr(args, key, value)
    return args


def load_model(model_name, batch_size):
    use_gpu = torch.cuda.is_available()
    config = MODEL_CONFIGS[model_name]
    arch_name = config.get("arch", model_name)
    args = build_args(arch_name, batch_size=batch_size, use_gpu=use_gpu, overrides=config.get("overrides"))
    exp = Exp_Long_Term_Forecast(args)
    ckpt_path = config["checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=exp.device)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if "total_ops" in key or "total_params" in key:
            continue
        cleaned[key] = value
    exp.model.load_state_dict(cleaned)
    exp.model.eval()
    return exp, args, ckpt_path


def make_decoder_inputs(args, batch_y, device):
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
    return dec_inp.float().to(device)


def predict(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)
    dec_inp = make_decoder_inputs(args, batch_y, exp.device)
    with torch.no_grad():
        out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return out[:, -args.pred_len :, :]


def empty_stats(delta, expected_scalar):
    return {
        "delta": float(delta),
        "expected_scalar": float(expected_scalar),
        "n": 0,
        "pred_sum": 0.0,
        "pred_abs_sum": 0.0,
        "pred_sq_sum": 0.0,
        "err_abs_sum": 0.0,
        "err_sq_sum": 0.0,
        "same_sign": 0.0,
        "nonzero_true": 0,
        "pred_values": [],
        "true_values": [],
    }


def add_stats(stats, pred_change, true_change):
    err = pred_change - true_change
    stats["n"] += pred_change.numel()
    stats["pred_sum"] += pred_change.sum().item()
    stats["pred_abs_sum"] += torch.abs(pred_change).sum().item()
    stats["pred_sq_sum"] += torch.square(pred_change).sum().item()
    stats["err_abs_sum"] += torch.abs(err).sum().item()
    stats["err_sq_sum"] += torch.square(err).sum().item()
    mask = torch.abs(true_change) > 1e-8
    if mask.any():
        signs_match = torch.sign(pred_change[mask]) == torch.sign(true_change[mask])
        stats["same_sign"] += signs_match.float().sum().item()
        stats["nonzero_true"] += mask.sum().item()
    stats["pred_values"].append(pred_change.detach().cpu().numpy())
    stats["true_values"].append(true_change.detach().cpu().numpy())


def safe_corr(a, b):
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def finalize_stats(stats):
    n = max(stats["n"], 1)
    pred = np.concatenate(stats["pred_values"], axis=0) if stats["pred_values"] else np.array([])
    true = np.concatenate(stats["true_values"], axis=0) if stats["true_values"] else np.array([])
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    expected = stats["expected_scalar"]
    pred_mean = stats["pred_sum"] / n
    return {
        "delta": stats["delta"],
        "expected_mean": expected,
        "pred_mean": pred_mean,
        "pred_abs_mean": stats["pred_abs_sum"] / n,
        "ire_mae": stats["err_abs_sum"] / n,
        "ire_rmse": math.sqrt(stats["err_sq_sum"] / n),
        "response_ratio": pred_mean / expected if abs(expected) > 1e-12 else float("nan"),
        "response_slope": slope,
        "response_corr": safe_corr(pred, true),
        "sign_accuracy_on_nonzero_true": (
            stats["same_sign"] / stats["nonzero_true"] if stats["nonzero_true"] > 0 else float("nan")
        ),
        "num_windows": stats["n"],
    }


def summarize_curve(rows):
    pred_all = []
    true_all = []
    for row in rows:
        delta = row["delta"]
        expected = row["expected_mean"]
        pred = row["pred_mean"]
        if abs(delta) < 1e-12:
            continue
        pred_all.append(pred)
        true_all.append(expected)
    pred_arr = np.asarray(pred_all, dtype=np.float64)
    true_arr = np.asarray(true_all, dtype=np.float64)
    denom = float(np.sum(true_arr * true_arr))
    slope = float(np.sum(pred_arr * true_arr) / denom) if denom > 1e-12 else float("nan")
    corr = safe_corr(pred_arr, true_arr)
    ire_mean = float(np.mean([row["ire_mae"] for row in rows if abs(row["delta"]) > 1e-12]))
    ratio_mean = float(np.mean([row["response_ratio"] for row in rows if abs(row["delta"]) > 1e-12]))
    return {
        "curve_slope_from_means": slope,
        "curve_corr_from_means": corr,
        "curve_ire_mae_mean": ire_mean,
        "curve_response_ratio_mean": ratio_mean,
        "num_delta_points": len([row for row in rows if abs(row["delta"]) > 1e-12]),
    }


def evaluate_model(model_name, batch_size, deltas, max_batches):
    exp, args, ckpt_path = load_model(model_name, batch_size=batch_size)
    test_data, test_loader = data_provider(args, flag="test")
    scaler = test_data.scaler
    cause_scale = float(scaler.scale_[0])
    target_scale = float(scaler.scale_[-1])
    causal_gain = 2.0 * cause_scale / target_scale
    stats = {float(delta): empty_stats(delta, causal_gain * float(delta)) for delta in deltas}
    num_batches = 0

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_x_device = batch_x.float().to(exp.device)
        pred_orig = predict(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred_orig_h1 = pred_orig[:, 0, -1]

        for delta in deltas:
            delta = float(delta)
            x_variant = batch_x_device.clone()
            x_variant[:, -1, 0] = x_variant[:, -1, 0] + delta
            pred_variant = predict(exp, args, x_variant, batch_y, batch_x_mark, batch_y_mark)
            pred_change = pred_variant[:, 0, -1] - pred_orig_h1
            true_change = torch.full_like(pred_change, causal_gain * delta)
            add_stats(stats[delta], pred_change, true_change)
        num_batches += 1

    rows = [finalize_stats(stats[float(delta)]) for delta in deltas]
    summary = summarize_curve(rows)
    return {
        "model": model_name,
        "checkpoint": ckpt_path,
        "num_batches": num_batches,
        "num_test_windows": len(test_data),
        "cause_scale": cause_scale,
        "target_scale": target_scale,
        "causal_gain_scaled": causal_gain,
        "rows": rows,
        "summary": summary,
    }


def save_curve_plot(results, output_dir):
    plt.figure(figsize=(8, 6))
    expected_x = None
    for item in results:
        rows = sorted(item["rows"], key=lambda row: row["delta"])
        x = [row["delta"] for row in rows]
        expected = [row["expected_mean"] for row in rows]
        pred = [row["pred_mean"] for row in rows]
        if expected_x is None:
            expected_x = (x, expected)
        plt.plot(x, pred, marker="o", label=item["model"])
    if expected_x is not None:
        plt.plot(expected_x[0], expected_x[1], color="black", linestyle="--", label="Expected response")
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("Cause intervention delta on last historical step")
    plt.ylabel("Predicted horizon-1 target change")
    plt.title("Delta-Response Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_response_curve.png"), dpi=180)
    plt.close()


def write_outputs(results, output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "delta_response_curve_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    rows = []
    for item in results:
        for row in item["rows"]:
            out = {
                "model": item["model"],
                "causal_gain_scaled": item["causal_gain_scaled"],
                "num_batches": item["num_batches"],
                "num_test_windows": item["num_test_windows"],
            }
            out.update(row)
            rows.append(out)

    csv_path = os.path.join(output_dir, "delta_response_curve_summary.csv")
    preferred = [
        "model",
        "delta",
        "expected_mean",
        "pred_mean",
        "response_ratio",
        "ire_mae",
        "ire_rmse",
        "response_slope",
        "response_corr",
        "sign_accuracy_on_nonzero_true",
        "causal_gain_scaled",
        "num_windows",
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for item in results:
        summary = {"model": item["model"], "causal_gain_scaled": item["causal_gain_scaled"]}
        summary.update(item["summary"])
        summary_rows.append(summary)
    with open(os.path.join(output_dir, "delta_response_curve_model_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "curve_slope_from_means",
            "curve_corr_from_means",
            "curve_ire_mae_mean",
            "curve_response_ratio_mean",
            "num_delta_points",
            "causal_gain_scaled",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    save_curve_plot(results, output_dir)

    md_path = os.path.join(output_dir, "delta_response_curve_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Delta-Response Curve Evaluation\n\n")
        f.write(f"Generated at: `{metadata['timestamp']}`\n\n")
        f.write("The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. ")
        f.write("We intervene on the last historical cause value and measure the horizon-1 target response.\n\n")
        f.write("## Model-Level Summary\n\n")
        f.write("| Model | Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for item in results:
            s = item["summary"]
            f.write(
                "| {model} | {slope:.6f} | {corr:.6f} | {ire:.6f} | {ratio:.6f} |\n".format(
                    model=item["model"],
                    slope=s["curve_slope_from_means"],
                    corr=s["curve_corr_from_means"],
                    ire=s["curve_ire_mae_mean"],
                    ratio=s["curve_response_ratio_mean"],
                )
            )
        f.write("\n## Per-Delta Response\n\n")
        f.write("| Model | Delta | Expected | Predicted | Ratio | IRE |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for item in results:
            for row in sorted(item["rows"], key=lambda item_row: item_row["delta"]):
                f.write(
                    "| {model} | {delta:.3f} | {expected:.6f} | {pred:.6f} | {ratio:.6f} | {ire:.6f} |\n".format(
                        model=item["model"],
                        delta=row["delta"],
                        expected=row["expected_mean"],
                        pred=row["pred_mean"],
                        ratio=row["response_ratio"],
                        ire=row["ire_mae"],
                    )
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["iTransformer", "iTransformer_CRR_FT01"])
    parser.add_argument("--deltas", nargs="+", type=float, default=[-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("causal_r1_delta_response", timestamp)
    metadata = {
        "timestamp": timestamp,
        "seed": args.seed,
        "deltas": args.deltas,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "models": args.models,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "counterfactual_scope": "horizon_1_only",
        "structural_equation": "Target_t = 2 * Cause_{t-1} + noise",
        "note": "The expected standardized H1 response is causal_gain * delta, where causal_gain = 2 * sigma_cause / sigma_target.",
    }

    print(json.dumps(metadata, indent=2))
    results = []
    for model_name in args.models:
        print(f"\n===== Evaluating {model_name} =====")
        result = evaluate_model(
            model_name=model_name,
            batch_size=args.batch_size,
            deltas=args.deltas,
            max_batches=args.max_batches,
        )
        results.append(result)
        s = result["summary"]
        print(
            "[{}] curve_slope={:.6f}, curve_corr={:.6f}, mean_IRE={:.6f}, mean_ratio={:.6f}".format(
                model_name,
                s["curve_slope_from_means"],
                s["curve_corr_from_means"],
                s["curve_ire_mae_mean"],
                s["curve_response_ratio_mean"],
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(results, output_dir, metadata)
    print(f"\nSaved delta-response curve results to: {output_dir}")


if __name__ == "__main__":
    main()
