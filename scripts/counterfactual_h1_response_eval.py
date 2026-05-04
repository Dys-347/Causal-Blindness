import argparse
import csv
import json
import math
import os
import random
from collections import OrderedDict
from copy import deepcopy
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
        model_id=f"Synthetic_Multi_{model_name}_H1CounterfactualEval",
        model=model_name,
        des="H1_Counterfactual_Response",
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


def apply_intervention(batch_x, name, delta):
    x = batch_x.clone()
    if name == "cause_last_shift_plus_delta":
        x[:, -1, 0] = x[:, -1, 0] + delta
    elif name == "cause_shift_plus_delta":
        x[:, :, 0] = x[:, :, 0] + delta
    elif name == "cause_zero":
        x[:, :, 0] = 0.0
    elif name == "cause_flip":
        x[:, :, 0] = -x[:, :, 0]
    elif name == "cause_flip_x5":
        x[:, :, 0] = -5.0 * x[:, :, 0]
    elif name == "distractors_last_shift_plus_delta":
        x[:, -1, 1:20] = x[:, -1, 1:20] + delta
    elif name == "distractors_shift_plus_delta":
        x[:, :, 1:20] = x[:, :, 1:20] + delta
    elif name == "noncause_1to7_shift_plus_delta":
        x[:, :, 1:8] = x[:, :, 1:8] + delta
    elif name == "target_zero":
        x[:, :, -1] = 0.0
    elif name == "target_shift_plus_delta":
        x[:, :, -1] = x[:, :, -1] + delta
    else:
        raise ValueError(f"Unknown intervention: {name}")
    return x


def expected_h1_change(x_orig, x_variant, name, causal_gain):
    batch_size = x_orig.shape[0]
    if name.startswith("cause_"):
        delta_cause_scaled = x_variant[:, -1, 0] - x_orig[:, -1, 0]
        return causal_gain * delta_cause_scaled
    return torch.zeros(batch_size, device=x_orig.device)


def new_stats():
    return {
        "n": 0,
        "pred_abs_sum": 0.0,
        "pred_sq_sum": 0.0,
        "true_abs_sum": 0.0,
        "true_sq_sum": 0.0,
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
    stats["pred_abs_sum"] += torch.abs(pred_change).sum().item()
    stats["pred_sq_sum"] += torch.square(pred_change).sum().item()
    stats["true_abs_sum"] += torch.abs(true_change).sum().item()
    stats["true_sq_sum"] += torch.square(true_change).sum().item()
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
    corr = safe_corr(pred, true)
    return {
        "pred_change_abs_mean": stats["pred_abs_sum"] / n,
        "pred_change_rms": math.sqrt(stats["pred_sq_sum"] / n),
        "true_change_abs_mean": stats["true_abs_sum"] / n,
        "true_change_rms": math.sqrt(stats["true_sq_sum"] / n),
        "ire_mae": stats["err_abs_sum"] / n,
        "ire_rmse": math.sqrt(stats["err_sq_sum"] / n),
        "response_slope": slope,
        "response_corr": corr,
        "sign_accuracy_on_nonzero_true": (
            stats["same_sign"] / stats["nonzero_true"] if stats["nonzero_true"] > 0 else float("nan")
        ),
        "num_windows": stats["n"],
        "num_nonzero_true": stats["nonzero_true"],
    }


def save_scatter_plot(rows, output_dir):
    cause_rows = [row for row in rows if row["intervention"].startswith("cause_")]
    if not cause_rows:
        return
    labels = [row["model"] + "\n" + row["intervention"].replace("cause_", "") for row in cause_rows]
    x = np.arange(len(cause_rows))
    true_abs = [row["true_change_abs_mean"] for row in cause_rows]
    pred_abs = [row["pred_change_abs_mean"] for row in cause_rows]
    width = 0.38
    plt.figure(figsize=(max(12, len(rows) * 0.55), 6))
    plt.bar(x - width / 2, true_abs, width, label="Expected |Delta target|")
    plt.bar(x + width / 2, pred_abs, width, label="Predicted |Delta target|")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Horizon-1 absolute target change")
    plt.title("Counterfactual H1 Response: Expected vs Predicted")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "h1_expected_vs_predicted_response.png"), dpi=160)
    plt.close()


def evaluate_model(model_name, batch_size, interventions, delta, max_batches):
    exp, args, ckpt_path = load_model(model_name, batch_size=batch_size)
    test_data, test_loader = data_provider(args, flag="test")
    scaler = test_data.scaler
    cause_scale = float(scaler.scale_[0])
    target_scale = float(scaler.scale_[-1])
    causal_gain = 2.0 * cause_scale / target_scale
    stats = {name: new_stats() for name in interventions}
    num_batches = 0

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_x_device = batch_x.float().to(exp.device)
        pred_orig = predict(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred_orig_h1 = pred_orig[:, 0, -1]

        for name in interventions:
            x_variant = apply_intervention(batch_x_device, name, delta)
            pred_variant = predict(exp, args, x_variant, batch_y, batch_x_mark, batch_y_mark)
            pred_change = pred_variant[:, 0, -1] - pred_orig_h1
            true_change = expected_h1_change(batch_x_device, x_variant, name, causal_gain)
            add_stats(stats[name], pred_change, true_change)
        num_batches += 1

    result = {
        "model": model_name,
        "checkpoint": ckpt_path,
        "num_batches": num_batches,
        "num_test_windows": len(test_data),
        "cause_scale": cause_scale,
        "target_scale": target_scale,
        "causal_gain_scaled": causal_gain,
        "interventions": {name: finalize_stats(item) for name, item in stats.items()},
    }
    return result


def write_outputs(results, output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "counterfactual_h1_response_results.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    rows = []
    for item in results:
        for name, stats in item["interventions"].items():
            row = {
                "model": item["model"],
                "intervention": name,
                "causal_gain_scaled": item["causal_gain_scaled"],
                "num_batches": item["num_batches"],
                "num_test_windows": item["num_test_windows"],
            }
            row.update(stats)
            rows.append(row)

    csv_path = os.path.join(output_dir, "counterfactual_h1_response_summary.csv")
    preferred = [
        "model",
        "intervention",
        "true_change_abs_mean",
        "pred_change_abs_mean",
        "ire_mae",
        "ire_rmse",
        "response_slope",
        "response_corr",
        "sign_accuracy_on_nonzero_true",
        "true_change_rms",
        "pred_change_rms",
        "causal_gain_scaled",
        "num_windows",
        "num_nonzero_true",
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    save_scatter_plot(rows, output_dir)

    md_path = os.path.join(output_dir, "counterfactual_h1_response_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Counterfactual Horizon-1 Response Evaluation\n\n")
        f.write(f"Generated at: `{metadata['timestamp']}`\n\n")
        f.write("The synthetic data uses `Target_t = 2 * Cause_{t-1} + noise`. ")
        f.write("Therefore, an intervention on the last historical cause value has a known counterfactual effect on horizon 1.\n\n")
        f.write("| Model | Intervention | Expected | Predicted | IRE MAE | Slope | Sign Acc. |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if row["intervention"] not in [
                "cause_last_shift_plus_delta",
                "cause_shift_plus_delta",
                "cause_zero",
                "cause_flip",
                "cause_flip_x5",
                "target_zero",
            ]:
                continue
            f.write(
                "| {model} | {intervention} | {true:.6f} | {pred:.6f} | {ire:.6f} | {slope:.6f} | {sign:.6f} |\n".format(
                    model=row["model"],
                    intervention=row["intervention"],
                    true=row["true_change_abs_mean"],
                    pred=row["pred_change_abs_mean"],
                    ire=row["ire_mae"],
                    slope=row["response_slope"] if not math.isnan(row["response_slope"]) else float("nan"),
                    sign=(
                        row["sign_accuracy_on_nonzero_true"]
                        if not math.isnan(row["sign_accuracy_on_nonzero_true"])
                        else float("nan")
                    ),
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["DLinear", "PatchTST", "iTransformer", "Crossformer", "TimeMixer"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    interventions = [
        "cause_last_shift_plus_delta",
        "cause_shift_plus_delta",
        "cause_zero",
        "cause_flip",
        "cause_flip_x5",
        "distractors_last_shift_plus_delta",
        "distractors_shift_plus_delta",
        "noncause_1to7_shift_plus_delta",
        "target_zero",
        "target_shift_plus_delta",
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("causal_r1_counterfactual_h1", timestamp)
    metadata = {
        "timestamp": timestamp,
        "seed": args.seed,
        "delta": args.delta,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "models": args.models,
        "interventions": interventions,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "counterfactual_scope": "horizon_1_only",
        "structural_equation": "Target_t = 2 * Cause_{t-1} + noise",
        "note": "For historical-input cause interventions, only horizon 1 has a directly known counterfactual label without assuming future cause trajectories.",
    }

    print(json.dumps(metadata, indent=2))
    results = []
    for model_name in args.models:
        print(f"\n===== Evaluating {model_name} =====")
        result = evaluate_model(
            model_name=model_name,
            batch_size=args.batch_size,
            interventions=interventions,
            delta=args.delta,
            max_batches=args.max_batches,
        )
        results.append(result)
        cause = result["interventions"]["cause_last_shift_plus_delta"]
        target_zero = result["interventions"]["target_zero"]
        print(
            "[{}] cause_last_shift expected={:.6f}, predicted={:.6f}, IRE={:.6f}, slope={:.6f}; "
            "target_zero pred_change={:.6f}, IRE={:.6f}".format(
                model_name,
                cause["true_change_abs_mean"],
                cause["pred_change_abs_mean"],
                cause["ire_mae"],
                cause["response_slope"],
                target_zero["pred_change_abs_mean"],
                target_zero["ire_mae"],
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(results, output_dir, metadata)
    print(f"\nSaved counterfactual H1 response results to: {output_dir}")


if __name__ == "__main__":
    main()
