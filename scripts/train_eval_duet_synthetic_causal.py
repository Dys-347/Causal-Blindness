import argparse
import csv
import json
import os
import random
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ts_benchmark.baselines.duet.models.duet_model import DUETModel


class WindowDataset(Dataset):
    def __init__(self, data, border1, border2, seq_len, pred_len):
        self.data = data.astype(np.float32)
        self.border1 = int(border1)
        self.border2 = int(border2)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.length = self.border2 - self.border1 - self.seq_len - self.pred_len + 1
        if self.length <= 0:
            raise ValueError(
                f"Invalid split: border1={border1}, border2={border2}, "
                f"seq_len={seq_len}, pred_len={pred_len}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = self.border1 + idx
        x_start = start
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len
        return self.data[x_start:x_end], self.data[y_start:y_end]


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state = None
        self.best_epoch = -1

    def step(self, value, model, epoch):
        if value < self.best:
            self.best = float(value)
            self.bad_epochs = 0
            self.best_epoch = int(epoch)
            self.best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args, n_vars):
    return SimpleNamespace(
        CI=bool(args.ci),
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        horizon=args.pred_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        hidden_size=args.hidden_size,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        factor=args.factor,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        output_attention=False,
        activation="gelu",
        moving_avg=args.moving_avg,
        num_experts=args.num_experts,
        noisy_gating=True,
        k=args.k,
    )


def load_synthetic_data(path):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    cause_col = "Cause_Var" if "Cause_Var" in df.columns else df.columns[0]
    target_col = "Target_Var" if "Target_Var" in df.columns else df.columns[1]
    other_cols = [col for col in df.columns if col not in {cause_col, target_col}]
    ordered_cols = [cause_col] + other_cols + [target_col]
    values = df[ordered_cols].to_numpy(dtype=np.float32)
    return values, ordered_cols


def make_splits(values, seq_len, pred_len):
    n = len(values)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test

    scaler = StandardScaler()
    scaler.fit(values[:num_train])
    scaled = scaler.transform(values).astype(np.float32)

    borders = {
        "train": (0, num_train),
        "val": (num_train - seq_len, num_train + num_val),
        "test": (n - num_test - seq_len, n),
    }
    datasets = {
        name: WindowDataset(scaled, b1, b2, seq_len, pred_len)
        for name, (b1, b2) in borders.items()
    }
    return scaled, scaler, borders, datasets


def clean_forward(model, x):
    # DUETModel.forward() in the inspected server repository contains manual
    # intervention-debug code. The original clean implementation is kept as
    # forward_(), so all formal experiments call it explicitly.
    return model.forward_(x)


def make_criterion(name):
    name = name.upper()
    if name == "MSE":
        return nn.MSELoss()
    if name == "MAE":
        return nn.L1Loss()
    if name == "HUBER":
        return nn.HuberLoss(delta=0.5)
    raise ValueError(f"Unknown loss: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    pred_losses = []
    imp_losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out, imp_loss = clean_forward(model, x)
        out = out[:, -y.shape[1] :, :]
        pred_loss = criterion(out, y)
        total_loss = pred_loss + imp_loss
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().cpu()))
        pred_losses.append(float(pred_loss.detach().cpu()))
        imp_losses.append(float(imp_loss.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "pred_loss": float(np.mean(pred_losses)),
        "importance_loss": float(np.mean(imp_losses)),
    }


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out, _ = clean_forward(model, x)
        out = out[:, -y.shape[1] :, :]
        losses.append(float(criterion(out, y).detach().cpu()))
    return float(np.mean(losses))


def apply_intervention(x, name, delta):
    out = x.clone()
    if name == "cause_last_shift_plus_delta":
        out[:, -1, 0] += delta
    elif name == "cause_shift_plus_delta":
        out[:, :, 0] += delta
    elif name == "cause_zero":
        out[:, :, 0] = 0.0
    elif name == "cause_flip":
        out[:, :, 0] = -out[:, :, 0]
    elif name == "cause_flip_x5":
        out[:, :, 0] = -5.0 * out[:, :, 0]
    elif name == "distractors_last_shift_plus_delta":
        out[:, -1, 1:-1] += delta
    elif name == "distractors_shift_plus_delta":
        out[:, :, 1:-1] += delta
    elif name == "target_zero":
        out[:, :, -1] = 0.0
    elif name == "target_shift_plus_delta":
        out[:, :, -1] += delta
    else:
        raise ValueError(f"Unknown intervention: {name}")
    return out


def expected_h1_change(x_orig, x_variant, name, causal_gain):
    if name.startswith("cause_"):
        return causal_gain * (x_variant[:, -1, 0] - x_orig[:, -1, 0])
    return torch.zeros(x_orig.shape[0], device=x_orig.device)


def init_stats():
    return defaultdict(list)


def add_array(stats, key, value):
    arr = value.detach().cpu().numpy().reshape(-1)
    stats[key].append(arr)


def cat_stat(stats, key):
    values = stats.get(key, [])
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values)


def finalize_h1(stats):
    pred = cat_stat(stats, "pred_change")
    true = cat_stat(stats, "true_change")
    err = pred - true
    true_abs = np.abs(true)
    pred_abs = np.abs(pred)
    denom = float(np.sum(true * true))
    slope = float(np.sum(pred * true) / denom) if denom > 1e-12 else float("nan")
    corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(true) > 1e-12 and np.std(pred) > 1e-12 else float("nan")
    nonzero = true_abs > 1e-9
    sign_acc = (
        float(np.mean(np.sign(pred[nonzero]) == np.sign(true[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "true_change_abs_mean": float(np.mean(true_abs)) if true.size else float("nan"),
        "pred_change_abs_mean": float(np.mean(pred_abs)) if pred.size else float("nan"),
        "ire_mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "ire_rmse": float(np.sqrt(np.mean(err * err))) if err.size else float("nan"),
        "response_slope": slope,
        "response_corr": corr,
        "sign_accuracy_on_nonzero_true": sign_acc,
        "num_windows": int(pred.size),
        "num_nonzero_true": int(np.sum(nonzero)),
    }


def finalize_obs(stats):
    pred = cat_stat(stats, "pred_target")
    true = cat_stat(stats, "true_target")
    err = pred - true
    return {
        "target_mse": float(np.mean(err * err)),
        "target_mae": float(np.mean(np.abs(err))),
        "num_target_points": int(pred.size),
    }


def finalize_sensitivity(stats):
    diff_target = cat_stat(stats, "target_diff")
    diff_all = cat_stat(stats, "all_diff")
    input_diff = cat_stat(stats, "input_diff")
    return {
        "mpd_target_mean": float(np.mean(np.abs(diff_target))) if diff_target.size else float("nan"),
        "rmsd_target": float(np.sqrt(np.mean(diff_target * diff_target))) if diff_target.size else float("nan"),
        "mpd_all_mean": float(np.mean(np.abs(diff_all))) if diff_all.size else float("nan"),
        "rmsd_all": float(np.sqrt(np.mean(diff_all * diff_all))) if diff_all.size else float("nan"),
        "input_mpd_mean": float(np.mean(np.abs(input_diff))) if input_diff.size else float("nan"),
    }


@torch.no_grad()
def formal_evaluate(model, loader, device, causal_gain, delta):
    model.eval()
    interventions = [
        "cause_last_shift_plus_delta",
        "cause_shift_plus_delta",
        "cause_zero",
        "cause_flip",
        "cause_flip_x5",
        "distractors_last_shift_plus_delta",
        "distractors_shift_plus_delta",
        "target_zero",
        "target_shift_plus_delta",
    ]
    obs_stats = init_stats()
    h1_stats = {name: init_stats() for name in interventions}
    sens_stats = {name: init_stats() for name in interventions}
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred_orig, _ = clean_forward(model, x)
        pred_orig = pred_orig[:, -y.shape[1] :, :]

        add_array(obs_stats, "pred_target", pred_orig[:, :, -1])
        add_array(obs_stats, "true_target", y[:, :, -1])

        for name in interventions:
            x_variant = apply_intervention(x, name, delta)
            pred_variant, _ = clean_forward(model, x_variant)
            pred_variant = pred_variant[:, -y.shape[1] :, :]

            pred_change_h1 = pred_variant[:, 0, -1] - pred_orig[:, 0, -1]
            true_change_h1 = expected_h1_change(x, x_variant, name, causal_gain)
            add_array(h1_stats[name], "pred_change", pred_change_h1)
            add_array(h1_stats[name], "true_change", true_change_h1)

            add_array(sens_stats[name], "target_diff", pred_variant[:, :, -1] - pred_orig[:, :, -1])
            add_array(sens_stats[name], "all_diff", pred_variant - pred_orig)
            add_array(sens_stats[name], "input_diff", x_variant - x)
        num_batches += 1

    return {
        "observational": finalize_obs(obs_stats),
        "h1": {name: finalize_h1(stats) for name, stats in h1_stats.items()},
        "sensitivity": {name: finalize_sensitivity(stats) for name, stats in sens_stats.items()},
        "num_batches": num_batches,
    }


def write_outputs(output_dir, metadata, train_log, eval_result):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "duet_synthetic_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "train_log": train_log,
                "evaluation": eval_result,
            },
            f,
            indent=2,
        )

    h1_csv = os.path.join(output_dir, "duet_h1_summary.csv")
    with open(h1_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["intervention"] + sorted({k for v in eval_result["h1"].values() for k in v.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in eval_result["h1"].items():
            row = {"intervention": name}
            row.update(stats)
            writer.writerow(row)

    sens_csv = os.path.join(output_dir, "duet_sensitivity_summary.csv")
    with open(sens_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["intervention"] + sorted({k for v in eval_result["sensitivity"].values() for k in v.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in eval_result["sensitivity"].items():
            row = {"intervention": name}
            row.update(stats)
            writer.writerow(row)

    obs = eval_result["observational"]
    cause = eval_result["h1"]["cause_last_shift_plus_delta"]
    dist = eval_result["h1"]["distractors_last_shift_plus_delta"]
    target_zero = eval_result["h1"]["target_zero"]
    with open(os.path.join(output_dir, "duet_report.md"), "w", encoding="utf-8") as f:
        f.write("# DUET Synthetic Causal Evaluation\n\n")
        f.write("## Core Results\n\n")
        f.write("| Model | CI | Target MSE | Target MAE | Cause H1 Expected | Cause H1 Predicted | H1 IRE | Slope | Last-Distractor False Response | Target-zero Response |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            "| DUET | {ci} | {mse:.6f} | {mae:.6f} | {expected:.6f} | {pred:.6f} | {ire:.6f} | {slope:.6f} | {dist:.6f} | {tz:.6f} |\n".format(
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
        f.write("\nDetailed JSON/CSV files are in this directory.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=os.environ.get("CAUSAL_R1_SYNTHETIC_CSV", "dataset/synthetic_multivariate.csv"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--ci", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss", default="MSE")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
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
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "causal_r1_duet",
        f"synthetic_ci{int(args.ci)}_{timestamp}",
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

    obs = eval_result["observational"]
    cause = eval_result["h1"]["cause_last_shift_plus_delta"]
    dist = eval_result["h1"]["distractors_last_shift_plus_delta"]
    target_zero = eval_result["h1"]["target_zero"]
    print(
        "[DUET-CI{}] target_mse={:.6f}, target_mae={:.6f}, "
        "cause_last expected={:.6f}, predicted={:.6f}, IRE={:.6f}, slope={:.6f}, "
        "last_distractor_false={:.6f}, target_zero={:.6f}".format(
            int(args.ci),
            obs["target_mse"],
            obs["target_mae"],
            cause["true_change_abs_mean"],
            cause["pred_change_abs_mean"],
            cause["ire_mae"],
            cause["response_slope"],
            dist["pred_change_abs_mean"],
            target_zero["pred_change_abs_mean"],
        ),
        flush=True,
    )
    print(f"Saved DUET synthetic causal results to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
