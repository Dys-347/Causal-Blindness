import argparse
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

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
    "iTransformer_RIR_FT01": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1_ft_lam01/checkpoint.pth",
    },
    "iTransformer_RIR_FT03": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_crr_h1_ft_lam03/checkpoint.pth",
    },
    "iTransformer_MSE_FT": {
        "arch": "iTransformer",
        "checkpoint": "checkpoints/causal_r1_iTransformer_ft_predonly/checkpoint.pth",
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
        model_id=f"Synthetic_Multi_{model_name}_GradientSensitivity",
        model=model_name,
        des="Gradient_Input_Sensitivity",
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
    config = MODEL_CONFIGS[model_name]
    arch = config.get("arch", model_name)
    args = build_args(arch, batch_size=batch_size, use_gpu=torch.cuda.is_available(), overrides=config.get("overrides"))
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


def forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark):
    dec_inp = make_decoder_inputs(args, batch_y, exp.device)
    out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return out[:, -args.pred_len :, :]


def safe_ratio(num, den):
    return float(num / den) if abs(float(den)) > 1e-12 else float("nan")


def central_diff(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, mutate_fn, eps):
    with torch.no_grad():
        x_plus = batch_x.clone()
        x_minus = batch_x.clone()
        mutate_fn(x_plus, eps)
        mutate_fn(x_minus, -eps)
        pred_plus = forward_model(exp, args, x_plus, batch_y, batch_x_mark, batch_y_mark)[:, 0, -1]
        pred_minus = forward_model(exp, args, x_minus, batch_y, batch_x_mark, batch_y_mark)[:, 0, -1]
        deriv = (pred_plus - pred_minus) / (2.0 * eps)
    return float(deriv.abs().mean().detach().cpu())


def evaluate_model(model_name, batch_size, max_batches, eps):
    exp, args, ckpt_path = load_model(model_name, batch_size)
    _, test_loader = data_provider(args, flag="test")

    rows = []
    for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        if idx >= max_batches:
            break
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)
        rows.append(
            {
                "batch": idx,
                "cause_last": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, -1, 0].add_(d), eps
                ),
                "target_last": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, -1, -1].add_(d), eps
                ),
                "distractor_last_mean": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, -1, 1:-1].add_(d), eps
                ),
                "cause_history": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, :, 0].add_(d), eps
                ),
                "target_history": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, :, -1].add_(d), eps
                ),
                "distractor_history": central_diff(
                    exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark, lambda x, d: x[:, :, 1:-1].add_(d), eps
                ),
            }
        )

    if not rows:
        raise RuntimeError(f"No batches evaluated for {model_name}")
    keys = [k for k in rows[0].keys() if k != "batch"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(
        {
            "model": model_name,
            "checkpoint": ckpt_path,
            "num_batches": len(rows),
            "cause_last_over_target_last": safe_ratio(summary["cause_last"], summary["target_last"]),
            "cause_history_over_target_history": safe_ratio(summary["cause_history"], summary["target_history"]),
            "cause_last_over_distractor_last": safe_ratio(summary["cause_last"], summary["distractor_last_mean"]),
        }
    )
    return summary, rows


def write_outputs(output_dir, summaries, batch_rows):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "gradient_sensitivity_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summaries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    with open(os.path.join(output_dir, "gradient_sensitivity_batches.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = list(batch_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_rows)

    with open(os.path.join(output_dir, "gradient_sensitivity_results.json"), "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "batches": batch_rows}, f, indent=2)

    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Functional Input Sensitivity\n\n")
        f.write("All values are central-difference absolute sensitivities of the horizon-1 target prediction with respect to small standardized input perturbations.\n\n")
        f.write("| Model | Cause last | Target last | Cause/Target last | Cause history | Target history | Cause/Target history |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                "| {model} | {cause_last:.8f} | {target_last:.8f} | {ratio_last:.6f} | {cause_history:.8f} | {target_history:.8f} | {ratio_history:.6f} |\n".format(
                    model=row["model"],
                    cause_last=row["cause_last"],
                    target_last=row["target_last"],
                    ratio_last=row["cause_last_over_target_last"],
                    cause_history=row["cause_history"],
                    target_history=row["target_history"],
                    ratio_history=row["cause_history_over_target_history"],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["iTransformer", "iTransformer_MSE_FT", "iTransformer_RIR_FT01", "iTransformer_RIR_FT03"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"causal_r1_gradient_sensitivity_{timestamp}"
    summaries = []
    batch_rows = []
    for model in args.models:
        print(f"[RUN] {model}", flush=True)
        summary, rows = evaluate_model(model, args.batch_size, args.max_batches, args.eps)
        summaries.append(summary)
        for row in rows:
            row = dict(row)
            row["model"] = model
            batch_rows.append(row)
        print(
            "[{}] cause_last={:.8f}, target_last={:.8f}, ratio={:.6f}, cause_hist={:.8f}, target_hist={:.8f}".format(
                model,
                summary["cause_last"],
                summary["target_last"],
                summary["cause_last_over_target_last"],
                summary["cause_history"],
                summary["target_history"],
            ),
            flush=True,
        )
    write_outputs(output_dir, summaries, batch_rows)
    print(f"Saved gradient sensitivity results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
