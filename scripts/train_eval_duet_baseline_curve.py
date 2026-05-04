import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_eval_duet_crr_synthetic import evaluate_delta_curve, save_curve_plot
from train_eval_duet_synthetic_causal import (
    DUETModel,
    EarlyStopper,
    build_config,
    evaluate_loss,
    formal_evaluate,
    load_synthetic_data,
    make_criterion,
    make_splits,
    set_seed,
    train_one_epoch,
    write_outputs,
)


def write_curve_outputs(output_dir, metadata, curve_result):
    with open(os.path.join(output_dir, "duet_baseline_delta_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "delta_curve": curve_result}, f, indent=2)

    with open(os.path.join(output_dir, "duet_baseline_delta_curve.csv"), "w", newline="", encoding="utf-8") as f:
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

    label = f"DUET baseline CI={metadata['ci']}"
    save_curve_plot(curve_result, output_dir, label)

    summary = curve_result["summary"]
    with open(os.path.join(output_dir, "duet_baseline_delta_curve_report.md"), "w", encoding="utf-8") as f:
        f.write("# DUET Baseline Delta-Response Curve\n\n")
        f.write("## Curve Summary\n\n")
        f.write("| Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |\n")
        f.write("|---:|---:|---:|---:|\n")
        f.write(
            "| {slope:.6f} | {corr:.6f} | {ire:.6f} | {ratio:.6f} |\n\n".format(
                slope=summary["curve_slope_from_means"],
                corr=summary["curve_corr_from_means"],
                ire=summary["curve_ire_mae_mean"],
                ratio=summary["curve_response_ratio_mean"],
            )
        )
        f.write("## Per-Delta Response\n\n")
        f.write("| Delta | Expected | Predicted | Ratio | IRE |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in sorted(curve_result["rows"], key=lambda item: item["delta"]):
            f.write(
                "| {delta:.3f} | {expected:.6f} | {pred:.6f} | {ratio:.6f} | {ire:.6f} |\n".format(
                    delta=row["delta"],
                    expected=row["expected_mean"],
                    pred=row["pred_mean"],
                    ratio=row["response_ratio"],
                    ire=row["ire_mae"],
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=os.environ.get("CAUSAL_R1_SYNTHETIC_CSV", "dataset/synthetic_multivariate.csv"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--ci", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss", default="MSE")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc-dropout", type=float, default=0.1)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    values, columns = load_synthetic_data(args.data_path)
    _, scaler, borders, datasets = make_splits(values, args.seq_len, args.pred_len)
    n_vars = values.shape[1]
    config = build_config(args, n_vars)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ),
    }

    model = DUETModel(config).to(device)
    criterion = make_criterion(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(args.patience)
    train_log = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss = evaluate_loss(model, loaders["val"], criterion, device)
        test_loss = evaluate_loss(model, loaders["test"], criterion, device)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }
        train_log.append(record)
        print(json.dumps(record, indent=2), flush=True)
        stopper.step(val_loss, model, epoch)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    cause_scale = float(scaler.scale_[0])
    target_scale = float(scaler.scale_[-1])
    causal_gain = 2.0 * cause_scale / target_scale
    eval_result = formal_evaluate(model, loaders["test"], device, causal_gain, args.delta)
    deltas = [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]
    curve_result = evaluate_delta_curve(model, loaders["test"], device, causal_gain, deltas)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "causal_r1_duet",
        f"baseline_curve_ci{int(args.ci)}_{timestamp}",
    )
    metadata = {
        "timestamp": timestamp,
        "data_path": args.data_path,
        "columns": columns,
        "ci": int(args.ci),
        "config": vars(args),
        "borders": borders,
        "num_windows": {name: len(ds) for name, ds in datasets.items()},
        "device": str(device),
        "causal_gain_scaled": causal_gain,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best,
        "note": "DUETModel.forward_ is used to avoid the repository's modified debug forward().",
    }
    write_outputs(output_dir, metadata, train_log, eval_result)
    write_curve_outputs(output_dir, metadata, curve_result)

    obs = eval_result["observational"]
    cause = eval_result["h1"]["cause_last_shift_plus_delta"]
    target_zero = eval_result["h1"]["target_zero"]
    curve = curve_result["summary"]
    print(
        "[DUET-Baseline-CI{}] target_mse={:.6f}, target_mae={:.6f}, "
        "cause_last expected={:.6f}, predicted={:.6f}, IRE={:.6f}, slope={:.6f}, "
        "target_zero={:.6f}; curve_slope={:.6f}, curve_corr={:.6f}, curve_ire={:.6f}".format(
            int(args.ci),
            obs["target_mse"],
            obs["target_mae"],
            cause["true_change_abs_mean"],
            cause["pred_change_abs_mean"],
            cause["ire_mae"],
            cause["response_slope"],
            target_zero["pred_change_abs_mean"],
            curve["curve_slope_from_means"],
            curve["curve_corr_from_means"],
            curve["curve_ire_mae_mean"],
        ),
        flush=True,
    )
    print(f"Saved DUET baseline curve results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
