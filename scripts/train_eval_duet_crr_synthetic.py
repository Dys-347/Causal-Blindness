import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_eval_duet_synthetic_causal import (
    DUETModel,
    EarlyStopper,
    build_config,
    clean_forward,
    formal_evaluate,
    load_synthetic_data,
    make_criterion,
    make_splits,
    set_seed,
)
from synthetic_mechanism_utils import (
    build_response_context,
    expected_h1_change,
    load_mechanism_metadata,
)


def sample_delta(batch_size, max_abs_delta, device, min_abs_ratio=0.1):
    signs = torch.where(torch.rand(batch_size, device=device) > 0.5, 1.0, -1.0)
    low = float(max_abs_delta) * float(min_abs_ratio)
    mags = low + (float(max_abs_delta) - low) * torch.rand(batch_size, device=device)
    return signs * mags


def train_one_epoch_crr(model, loader, optimizer, pred_criterion, resp_criterion, device, response_context, args):
    model.train()
    totals = []
    pred_losses = []
    imp_losses = []
    resp_losses = []
    dist_losses = []
    slopes = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        pred, imp_loss = clean_forward(model, x)
        pred = pred[:, -y.shape[1] :, :]
        pred_loss = pred_criterion(pred, y)

        delta = sample_delta(x.shape[0], args.delta, device, args.min_abs_delta_ratio)
        x_cf = x.clone()
        x_cf[:, -1, 0] += delta
        pred_cf, imp_cf = clean_forward(model, x_cf)
        pred_cf = pred_cf[:, -y.shape[1] :, :]
        pred_change = pred_cf[:, 0, -1] - pred[:, 0, -1]
        expected = expected_h1_change(x, x_cf, "cause_last_shift_plus_delta", response_context)
        resp_loss = resp_criterion(pred_change, expected)

        dist_delta = sample_delta(x.shape[0], args.delta, device, args.min_abs_delta_ratio)
        x_dist = x.clone()
        x_dist[:, -1, 1:-1] += dist_delta.view(-1, 1)
        pred_dist, imp_dist = clean_forward(model, x_dist)
        pred_dist = pred_dist[:, -y.shape[1] :, :]
        dist_change = pred_dist[:, 0, -1] - pred[:, 0, -1]
        dist_loss = resp_criterion(dist_change, torch.zeros_like(dist_change))

        imp_total = imp_loss + args.importance_cf_weight * (imp_cf + imp_dist)
        total = pred_loss + imp_total + args.lambda_resp * resp_loss + args.lambda_dist * dist_loss
        total.backward()
        optimizer.step()

        denom = torch.sum(expected.detach() * expected.detach()).item()
        slope = (
            torch.sum(pred_change.detach() * expected.detach()).item() / denom
            if denom > 1e-12
            else float("nan")
        )
        totals.append(float(total.detach().cpu()))
        pred_losses.append(float(pred_loss.detach().cpu()))
        imp_losses.append(float(imp_total.detach().cpu()))
        resp_losses.append(float(resp_loss.detach().cpu()))
        dist_losses.append(float(dist_loss.detach().cpu()))
        slopes.append(slope)

    return {
        "loss": float(np.mean(totals)),
        "pred_loss": float(np.mean(pred_losses)),
        "importance_loss": float(np.mean(imp_losses)),
        "resp_loss": float(np.mean(resp_losses)),
        "dist_loss": float(np.mean(dist_losses)),
        "response_slope": float(np.mean(slopes)),
    }


@torch.no_grad()
def evaluate_crr_loss(model, loader, pred_criterion, resp_criterion, device, response_context, args):
    model.eval()
    pred_losses = []
    resp_losses = []
    dist_losses = []
    slopes = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, _ = clean_forward(model, x)
        pred = pred[:, -y.shape[1] :, :]
        pred_losses.append(float(pred_criterion(pred, y).detach().cpu()))

        delta = sample_delta(x.shape[0], args.delta, device, args.min_abs_delta_ratio)
        x_cf = x.clone()
        x_cf[:, -1, 0] += delta
        pred_cf, _ = clean_forward(model, x_cf)
        pred_cf = pred_cf[:, -y.shape[1] :, :]
        pred_change = pred_cf[:, 0, -1] - pred[:, 0, -1]
        expected = expected_h1_change(x, x_cf, "cause_last_shift_plus_delta", response_context)
        resp_losses.append(float(resp_criterion(pred_change, expected).detach().cpu()))

        dist_delta = sample_delta(x.shape[0], args.delta, device, args.min_abs_delta_ratio)
        x_dist = x.clone()
        x_dist[:, -1, 1:-1] += dist_delta.view(-1, 1)
        pred_dist, _ = clean_forward(model, x_dist)
        pred_dist = pred_dist[:, -y.shape[1] :, :]
        dist_change = pred_dist[:, 0, -1] - pred[:, 0, -1]
        dist_losses.append(float(resp_criterion(dist_change, torch.zeros_like(dist_change)).detach().cpu()))

        denom = torch.sum(expected.detach() * expected.detach()).item()
        slope = (
            torch.sum(pred_change.detach() * expected.detach()).item() / denom
            if denom > 1e-12
            else float("nan")
        )
        slopes.append(slope)

    return {
        "pred_loss": float(np.mean(pred_losses)),
        "resp_loss": float(np.mean(resp_losses)),
        "dist_loss": float(np.mean(dist_losses)),
        "response_slope": float(np.mean(slopes)),
    }


def curve_stats_init(delta):
    return {
        "delta": float(delta),
        "pred": [],
        "true": [],
    }


def safe_corr(a, b):
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def finalize_curve_row(stats):
    pred = np.concatenate(stats["pred"]) if stats["pred"] else np.array([], dtype=np.float32)
    true = np.concatenate(stats["true"]) if stats["true"] else np.array([], dtype=np.float32)
    err = pred - true
    expected = float(np.mean(true)) if true.size else float("nan")
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
    pred = np.asarray([row["pred_mean"] for row in rows if abs(row["delta"]) > 1e-12], dtype=np.float64)
    true = np.asarray([row["expected_mean"] for row in rows if abs(row["delta"]) > 1e-12], dtype=np.float64)
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    return {
        "curve_slope_from_means": slope,
        "curve_corr_from_means": safe_corr(pred, true),
        "curve_ire_mae_mean": float(np.mean([row["ire_mae"] for row in rows if abs(row["delta"]) > 1e-12])),
        "curve_response_ratio_mean": float(
            np.mean([row["response_ratio"] for row in rows if abs(row["delta"]) > 1e-12])
        ),
        "num_delta_points": len([row for row in rows if abs(row["delta"]) > 1e-12]),
    }


@torch.no_grad()
def evaluate_delta_curve(model, loader, device, response_context, deltas):
    model.eval()
    stats = {float(delta): curve_stats_init(delta) for delta in deltas}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred, _ = clean_forward(model, x)
        pred = pred[:, -y.shape[1] :, :]
        pred_h1 = pred[:, 0, -1]
        for delta in deltas:
            delta = float(delta)
            x_cf = x.clone()
            x_cf[:, -1, 0] += delta
            pred_cf, _ = clean_forward(model, x_cf)
            pred_cf = pred_cf[:, -y.shape[1] :, :]
            pred_change = pred_cf[:, 0, -1] - pred_h1
            true_change = expected_h1_change(x, x_cf, "cause_last_shift_plus_delta", response_context)
            stats[delta]["pred"].append(pred_change.detach().cpu().numpy().reshape(-1))
            stats[delta]["true"].append(true_change.detach().cpu().numpy().reshape(-1))
    rows = [finalize_curve_row(stats[float(delta)]) for delta in deltas]
    return {"rows": rows, "summary": summarize_curve(rows)}


def save_curve_plot(curve, output_dir, label):
    rows = sorted(curve["rows"], key=lambda row: row["delta"])
    x = [row["delta"] for row in rows]
    expected = [row["expected_mean"] for row in rows]
    pred = [row["pred_mean"] for row in rows]
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(x, expected, color="black", linestyle="--", marker="o", label="Expected")
    plt.plot(x, pred, marker="o", label=label)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("Cause intervention delta on last historical step")
    plt.ylabel("Predicted horizon-1 target change")
    plt.title(f"DUET Delta-Response Curve ({label})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "duet_delta_response_curve.png"), dpi=180)
    plt.close()


def write_crr_outputs(output_dir, metadata, train_log, eval_result, curve_result):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "duet_crr_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "train_log": train_log,
                "evaluation": eval_result,
                "delta_curve": curve_result,
            },
            f,
            indent=2,
        )

    with open(os.path.join(output_dir, "duet_crr_h1_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = ["intervention"] + sorted({k for v in eval_result["h1"].values() for k in v.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in eval_result["h1"].items():
            row = {"intervention": name}
            row.update(stats)
            writer.writerow(row)

    with open(os.path.join(output_dir, "duet_crr_delta_curve.csv"), "w", newline="", encoding="utf-8") as f:
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

    obs = eval_result["observational"]
    cause = eval_result["h1"]["cause_last_shift_plus_delta"]
    dist = eval_result["h1"]["distractors_last_shift_plus_delta"]
    target_zero = eval_result["h1"]["target_zero"]
    curve = curve_result["summary"]
    label = f"CI={metadata['ci']}, lr={metadata['config']['lr']}"
    save_curve_plot(curve_result, output_dir, label)

    with open(os.path.join(output_dir, "duet_crr_report.md"), "w", encoding="utf-8") as f:
        f.write("# DUET + CRR/RIR Synthetic Causal Evaluation\n\n")
        f.write("## Core H1 Result\n\n")
        f.write("| Model | CI | Target MSE | Target MAE | Cause Expected | Cause Predicted | H1 IRE | Slope | Last-Dist False | Target-zero |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| DUET + CRR/RIR | {ci} | {mse:.6f} | {mae:.6f} | {expected:.6f} | {pred:.6f} | {ire:.6f} | {slope:.6f} | {dist:.6f} | {tz:.6f} |\n\n".format(
                ci=metadata["ci"],
                mse=obs["target_mse"],
                mae=obs["target_mae"],
                expected=cause["true_change_abs_mean"],
                pred=cause["pred_change_abs_mean"],
                ire=cause["ire_mae"],
                slope=cause["response_slope"],
                dist=dist["pred_change_abs_mean"],
                tz=target_zero["pred_change_abs_mean"],
            )
        )
        f.write("## Delta-Response Curve Summary\n\n")
        f.write("| Curve Slope | Curve Corr. | Mean IRE | Mean Ratio |\n")
        f.write("|---:|---:|---:|---:|\n")
        f.write(
            "| {slope:.6f} | {corr:.6f} | {ire:.6f} | {ratio:.6f} |\n\n".format(
                slope=curve["curve_slope_from_means"],
                corr=curve["curve_corr_from_means"],
                ire=curve["curve_ire_mae_mean"],
                ratio=curve["curve_response_ratio_mean"],
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
    parser.add_argument("--lambda-resp", type=float, default=0.1)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--selection-pred-weight", type=float, default=0.1)
    parser.add_argument("--selection-dist-weight", type=float, default=0.05)
    parser.add_argument("--importance-cf-weight", type=float, default=0.0)
    parser.add_argument("--min-abs-delta-ratio", type=float, default=0.1)
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
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
    }

    mechanism_metadata = load_mechanism_metadata(args.data_path)
    response_context = build_response_context(scaler, mechanism_metadata)

    model = DUETModel(config).to(device)
    pred_criterion = make_criterion(args.loss)
    resp_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(args.patience)
    train_log = []

    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_crr(
            model,
            loaders["train"],
            optimizer,
            pred_criterion,
            resp_criterion,
            device,
            response_context,
            args,
        )
        val_metrics = evaluate_crr_loss(model, loaders["val"], pred_criterion, resp_criterion, device, response_context, args)
        test_metrics = evaluate_crr_loss(model, loaders["test"], pred_criterion, resp_criterion, device, response_context, args)
        score = (
            val_metrics["resp_loss"]
            + args.selection_pred_weight * val_metrics["pred_loss"]
            + args.selection_dist_weight * val_metrics["dist_loss"]
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "selection_score": float(score),
        }
        train_log.append(record)
        print(json.dumps(record, indent=2), flush=True)
        if stopper.step(score, model, epoch):
            best_score = float(score)
            print(f"Saved in-memory best epoch {epoch} with score={score:.6f}", flush=True)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    eval_result = formal_evaluate(model, loaders["test"], device, response_context, args.delta)
    deltas = [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]
    curve_result = evaluate_delta_curve(model, loaders["test"], device, response_context, deltas)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("causal_r1_duet", f"crr_ci{int(args.ci)}_{timestamp}")
    metadata = {
        "timestamp": timestamp,
        "data_path": args.data_path,
        "columns": columns,
        "ci": int(args.ci),
        "config": vars(args),
        "borders": borders,
        "num_windows": {name: len(ds) for name, ds in datasets.items()},
        "device": str(device),
        "mechanism_metadata": mechanism_metadata,
        "response_context": {key: value for key, value in response_context.items() if key != "metadata"},
        "causal_gain_scaled": response_context.get("causal_gain_scaled"),
        "best_epoch": stopper.best_epoch,
        "best_score": best_score,
        "note": "DUETModel.forward_ is used to avoid the repository's modified debug forward().",
    }
    write_crr_outputs(output_dir, metadata, train_log, eval_result, curve_result)

    obs = eval_result["observational"]
    cause = eval_result["h1"]["cause_last_shift_plus_delta"]
    dist = eval_result["h1"]["distractors_last_shift_plus_delta"]
    target_zero = eval_result["h1"]["target_zero"]
    curve = curve_result["summary"]
    print(
        "[DUET-CRR-CI{}] target_mse={:.6f}, target_mae={:.6f}, "
        "cause_last expected={:.6f}, predicted={:.6f}, IRE={:.6f}, slope={:.6f}, "
        "last_distractor_false={:.6f}, target_zero={:.6f}; "
        "curve_slope={:.6f}, curve_corr={:.6f}, curve_ire={:.6f}".format(
            int(args.ci),
            obs["target_mse"],
            obs["target_mae"],
            cause["true_change_abs_mean"],
            cause["pred_change_abs_mean"],
            cause["ire_mae"],
            cause["response_slope"],
            dist["pred_change_abs_mean"],
            target_zero["pred_change_abs_mean"],
            curve["curve_slope_from_means"],
            curve["curve_corr_from_means"],
            curve["curve_ire_mae_mean"],
        ),
        flush=True,
    )
    print(f"Saved DUET + CRR/RIR synthetic causal results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
