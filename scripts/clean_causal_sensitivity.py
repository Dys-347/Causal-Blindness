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


CHANNEL_MAP = {
    "cause": 0,
    "distractors": [1, 20],
    "target": -1,
    "num_channels": 21,
}


def build_args(model_name, batch_size, use_gpu, overrides=None):
    args = argparse.Namespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id=f"Synthetic_Multi_{model_name}_CleanCausalEval",
        model=model_name,
        des="Clean_Causal_Sensitivity",
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


def new_stats():
    return {
        "n_target": 0,
        "abs_target_sum": 0.0,
        "sq_target_sum": 0.0,
        "abs_target_sumsq": 0.0,
        "n_all": 0,
        "abs_all_sum": 0.0,
        "sq_all_sum": 0.0,
        "input_abs_sum": 0.0,
        "input_n": 0,
    }


def add_prediction_error(stats, pred, true):
    diff_t = pred[:, :, -1] - true[:, :, -1]
    abs_t = torch.abs(diff_t)
    stats["n_target"] += diff_t.numel()
    stats["abs_target_sum"] += abs_t.sum().item()
    stats["sq_target_sum"] += torch.square(diff_t).sum().item()

    diff_all = pred - true
    stats["n_all"] += diff_all.numel()
    stats["abs_all_sum"] += torch.abs(diff_all).sum().item()
    stats["sq_all_sum"] += torch.square(diff_all).sum().item()


def add_intervention_diff(stats, pred_variant, pred_orig, x_variant, x_orig):
    diff_t = pred_variant[:, :, -1] - pred_orig[:, :, -1]
    abs_t = torch.abs(diff_t)
    stats["n_target"] += diff_t.numel()
    stats["abs_target_sum"] += abs_t.sum().item()
    stats["sq_target_sum"] += torch.square(diff_t).sum().item()
    stats["abs_target_sumsq"] += torch.square(abs_t).sum().item()

    diff_all = pred_variant - pred_orig
    stats["n_all"] += diff_all.numel()
    stats["abs_all_sum"] += torch.abs(diff_all).sum().item()
    stats["sq_all_sum"] += torch.square(diff_all).sum().item()

    input_abs = torch.abs(x_variant - x_orig)
    stats["input_abs_sum"] += input_abs.sum().item()
    stats["input_n"] += input_abs.numel()


def finalize_prediction_error(stats):
    n_t = max(stats["n_target"], 1)
    n_all = max(stats["n_all"], 1)
    return {
        "target_mae": stats["abs_target_sum"] / n_t,
        "target_mse": stats["sq_target_sum"] / n_t,
        "all_mae": stats["abs_all_sum"] / n_all,
        "all_mse": stats["sq_all_sum"] / n_all,
        "num_target_points": stats["n_target"],
    }


def finalize_intervention_stats(stats):
    n_t = max(stats["n_target"], 1)
    n_all = max(stats["n_all"], 1)
    mean_abs_t = stats["abs_target_sum"] / n_t
    second_abs_t = stats["abs_target_sumsq"] / n_t
    std_abs_t = math.sqrt(max(second_abs_t - mean_abs_t * mean_abs_t, 0.0))
    input_mean = stats["input_abs_sum"] / max(stats["input_n"], 1)
    return {
        "mpd_target_mean": mean_abs_t,
        "mpd_target_std": std_abs_t,
        "rmsd_target": math.sqrt(stats["sq_target_sum"] / n_t),
        "mpd_all_mean": stats["abs_all_sum"] / n_all,
        "rmsd_all": math.sqrt(stats["sq_all_sum"] / n_all),
        "input_mpd_mean": input_mean,
        "normalized_target_sensitivity": mean_abs_t / (input_mean + 1e-12),
        "num_target_points": stats["n_target"],
    }


def save_sample_plot(sample_cache, model_name, output_dir):
    if not sample_cache:
        return
    x_hist = np.arange(sample_cache["history_target"].shape[0])
    x_future = np.arange(
        sample_cache["history_target"].shape[0],
        sample_cache["history_target"].shape[0] + sample_cache["orig"].shape[0],
    )
    plt.figure(figsize=(13, 7))
    plt.plot(x_hist, sample_cache["history_target"], color="black", alpha=0.45, label="Input Target History")
    plt.plot(x_future, sample_cache["true"], color="gray", alpha=0.65, label="True Future Target")
    plt.plot(x_future, sample_cache["orig"], color="blue", linewidth=2.0, label="Original")
    styles = {
        "cause_shift_plus_delta": ("red", "--"),
        "cause_zero": ("orange", "--"),
        "cause_flip": ("purple", "--"),
        "distractors_last_shift_plus_delta": ("teal", ":"),
        "distractors_shift_plus_delta": ("green", ":"),
        "target_zero": ("brown", "-."),
    }
    for name, values in sample_cache["variants"].items():
        if name not in styles:
            continue
        color, linestyle = styles[name]
        plt.plot(x_future, values, color=color, linestyle=linestyle, linewidth=1.6, label=name)
    plt.title(f"Clean Causal Sensitivity Sample: {model_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Target Value")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{model_name}_sample_response.png")
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate_model(model_name, batch_size, interventions, delta, max_batches, output_dir):
    exp, args, ckpt_path = load_model(model_name, batch_size=batch_size)
    test_data, test_loader = data_provider(args, flag="test")

    pred_stats = new_stats()
    int_stats = {name: new_stats() for name in interventions}
    sample_cache = None
    num_batches = 0

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_x_device = batch_x.float().to(exp.device)
        batch_y_device = batch_y.float().to(exp.device)
        true_future = batch_y_device[:, -args.pred_len :, :]

        pred_orig = predict(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
        add_prediction_error(pred_stats, pred_orig, true_future)

        if sample_cache is None:
            sample_cache = {
                "history_target": batch_x[0, :, -1].detach().cpu().numpy(),
                "true": true_future[0, :, -1].detach().cpu().numpy(),
                "orig": pred_orig[0, :, -1].detach().cpu().numpy(),
                "variants": {},
            }

        for name in interventions:
            x_variant = apply_intervention(batch_x_device, name, delta)
            pred_variant = predict(exp, args, x_variant, batch_y, batch_x_mark, batch_y_mark)
            add_intervention_diff(int_stats[name], pred_variant, pred_orig, x_variant, batch_x_device)
            if batch_idx == 0 and sample_cache is not None:
                sample_cache["variants"][name] = pred_variant[0, :, -1].detach().cpu().numpy()

        num_batches += 1

    obs = finalize_prediction_error(pred_stats)
    interventions_out = {name: finalize_intervention_stats(stats) for name, stats in int_stats.items()}

    cause_ref = interventions_out.get("cause_shift_plus_delta", {}).get("mpd_target_mean", float("nan"))
    distractor_ref = interventions_out.get("distractors_shift_plus_delta", {}).get("mpd_target_mean", float("nan"))
    target_ref = interventions_out.get("target_zero", {}).get("mpd_target_mean", float("nan"))
    derived = {
        "csr_cause_shift_over_distractor_shift": cause_ref / (distractor_ref + 1e-12),
        "target_zero_over_cause_shift": target_ref / (cause_ref + 1e-12),
    }

    save_sample_plot(sample_cache, model_name, output_dir)

    result = {
        "model": model_name,
        "checkpoint": ckpt_path,
        "num_batches": num_batches,
        "num_test_windows": len(test_data),
        "observational": obs,
        "interventions": interventions_out,
        "derived": derived,
    }
    return result


def write_outputs(results, output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "clean_causal_sensitivity_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    rows = []
    for item in results:
        base = {
            "model": item["model"],
            "num_batches": item["num_batches"],
            "num_test_windows": item["num_test_windows"],
            **item["observational"],
            **item["derived"],
        }
        for name, stats in item["interventions"].items():
            row = deepcopy(base)
            row["intervention"] = name
            row.update(stats)
            rows.append(row)

    csv_path = os.path.join(output_dir, "clean_causal_sensitivity_summary.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model",
        "intervention",
        "target_mae",
        "target_mse",
        "all_mae",
        "all_mse",
        "mpd_target_mean",
        "mpd_target_std",
        "rmsd_target",
        "mpd_all_mean",
        "rmsd_all",
        "input_mpd_mean",
        "normalized_target_sensitivity",
        "csr_cause_shift_over_distractor_shift",
        "target_zero_over_cause_shift",
        "num_batches",
        "num_test_windows",
        "num_target_points",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(output_dir, "clean_causal_sensitivity_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Clean Causal Sensitivity Evaluation\n\n")
        f.write(f"Generated at: `{metadata['timestamp']}`\n\n")
        f.write("## Channel Map\n\n")
        f.write("- channel 0: `Cause_Var`\n")
        f.write("- channels 1-19: distractors\n")
        f.write("- channel 20 / -1: `Target_Var`\n\n")
        f.write("## Summary\n\n")
        f.write("| Model | Target MSE | Target MAE | Cause shift MPD | Distractor shift MPD | CSR | Target-zero / Cause-shift |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for item in results:
            obs = item["observational"]
            ints = item["interventions"]
            derived = item["derived"]
            f.write(
                "| {model} | {mse:.6f} | {mae:.6f} | {cause:.6f} | {dist:.6f} | {csr:.6f} | {tr:.6f} |\n".format(
                    model=item["model"],
                    mse=obs["target_mse"],
                    mae=obs["target_mae"],
                    cause=ints["cause_shift_plus_delta"]["mpd_target_mean"],
                    dist=ints["distractors_shift_plus_delta"]["mpd_target_mean"],
                    csr=derived["csr_cause_shift_over_distractor_shift"],
                    tr=derived["target_zero_over_cause_shift"],
                )
            )
        f.write("\n")
        f.write("Detailed rows are in `clean_causal_sensitivity_summary.csv`.\n")


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
    output_dir = args.output_dir or os.path.join("causal_r1_clean_eval", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "seed": args.seed,
        "delta": args.delta,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "models": args.models,
        "interventions": interventions,
        "channel_map": CHANNEL_MAP,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "note": "All interventions are applied after DataLoader standardization, matching the historical result.py style.",
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
            output_dir=output_dir,
        )
        results.append(result)
        obs = result["observational"]
        ints = result["interventions"]
        print(
            "[{}] target_mse={:.6f}, target_mae={:.6f}, cause_shift_mpd={:.6f}, "
            "distractor_shift_mpd={:.6f}, target_zero_mpd={:.6f}".format(
                model_name,
                obs["target_mse"],
                obs["target_mae"],
                ints["cause_shift_plus_delta"]["mpd_target_mean"],
                ints["distractors_shift_plus_delta"]["mpd_target_mean"],
                ints["target_zero"]["mpd_target_mean"],
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(results, output_dir, metadata)
    print(f"\nSaved clean causal sensitivity results to: {output_dir}")


if __name__ == "__main__":
    main()
