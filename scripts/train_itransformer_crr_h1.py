import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim

sys.path.insert(0, os.getcwd())

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

BASELINE_ITRANSFORMER_CKPT = (
    "checkpoints/"
    "long_term_forecast_Synthetic_Multi_iTransformer_96_96_iTransformer_SyntheticMulti_"
    "ftM_sl96_ll48_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_"
    "dtTrue_Exp_Causal_Check_0/checkpoint.pth"
)


def build_args(batch_size, learning_rate):
    return argparse.Namespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id="Synthetic_Multi_iTransformer_CRR_H1_96_96",
        model="iTransformer",
        des="Exp_Causal_CRR_H1",
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
        learning_rate=learning_rate,
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


def sample_delta(batch_size, max_abs_delta, device):
    # Symmetric random intervention. Avoid tiny deltas so the response signal is not
    # dominated by numerical noise.
    signs = torch.where(torch.rand(batch_size, device=device) > 0.5, 1.0, -1.0)
    mags = 0.5 * max_abs_delta + 0.5 * max_abs_delta * torch.rand(batch_size, device=device)
    return signs * mags


def evaluate(exp, args, loader, criterion, causal_gain, max_batches=None):
    exp.model.eval()
    pred_losses = []
    resp_losses = []
    slopes = []
    with torch.no_grad():
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            true = batch_y[:, -args.pred_len :, :]

            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred_loss = criterion(pred, true)

            delta = sample_delta(batch_x.shape[0], 5.0, exp.device)
            x_cf = batch_x.clone()
            x_cf[:, -1, 0] = x_cf[:, -1, 0] + delta
            pred_cf = forward_model(exp, args, x_cf, batch_y, batch_x_mark, batch_y_mark)
            pred_change = pred_cf[:, 0, -1] - pred[:, 0, -1]
            expected = causal_gain * delta
            resp_loss = criterion(pred_change, expected)
            denom = torch.sum(expected * expected).item()
            slope = torch.sum(pred_change * expected).item() / denom if denom > 1e-12 else float("nan")

            pred_losses.append(pred_loss.item())
            resp_losses.append(resp_loss.item())
            slopes.append(slope)

    exp.model.train()
    return {
        "pred_loss": float(np.mean(pred_losses)) if pred_losses else float("nan"),
        "resp_loss": float(np.mean(resp_losses)) if resp_losses else float("nan"),
        "response_slope": float(np.mean(slopes)) if slopes else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/causal_r1_iTransformer_crr_h1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-resp", type=float, default=1.0)
    parser.add_argument("--lambda-dist", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--selection-pred-weight", type=float, default=0.1)
    args_cli = parser.parse_args()

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    args = build_args(batch_size=args_cli.batch_size, learning_rate=args_cli.learning_rate)
    args.train_epochs = args_cli.epochs
    args.patience = args_cli.patience
    exp = Exp_Long_Term_Forecast(args)

    if args_cli.init_checkpoint:
        ckpt_path = BASELINE_ITRANSFORMER_CKPT if args_cli.init_checkpoint == "baseline" else args_cli.init_checkpoint
        state_dict = torch.load(ckpt_path, map_location=exp.device)
        cleaned = {}
        for key, value in state_dict.items():
            if "total_ops" in key or "total_params" in key:
                continue
            cleaned[key] = value
        exp.model.load_state_dict(cleaned)
        print(f"Loaded initialization checkpoint: {ckpt_path}")

    train_data, train_loader = data_provider(args, flag="train")
    val_data, val_loader = data_provider(args, flag="val")
    test_data, test_loader = data_provider(args, flag="test")

    cause_scale = float(train_data.scaler.scale_[0])
    target_scale = float(train_data.scaler.scale_[-1])
    causal_gain = 2.0 * cause_scale / target_scale

    criterion = nn.MSELoss()
    optimizer = optim.Adam(exp.model.parameters(), lr=args_cli.learning_rate)

    os.makedirs(args_cli.output_dir, exist_ok=True)
    log = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seed": args_cli.seed,
        "output_dir": args_cli.output_dir,
        "lambda_resp": args_cli.lambda_resp,
        "lambda_dist": args_cli.lambda_dist,
        "init_checkpoint": args_cli.init_checkpoint,
        "selection_pred_weight": args_cli.selection_pred_weight,
        "delta": args_cli.delta,
        "causal_gain_scaled": causal_gain,
        "epochs": [],
    }

    best_score = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args_cli.output_dir, "checkpoint.pth")

    for epoch in range(1, args_cli.epochs + 1):
        exp.model.train()
        running = []
        running_pred = []
        running_resp = []
        running_dist = []
        running_slope = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            true = batch_y[:, -args.pred_len :, :]

            optimizer.zero_grad()

            pred = forward_model(exp, args, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss_pred = criterion(pred, true)

            delta = sample_delta(batch_x.shape[0], args_cli.delta, exp.device)
            x_cf = batch_x.clone()
            x_cf[:, -1, 0] = x_cf[:, -1, 0] + delta
            pred_cf = forward_model(exp, args, x_cf, batch_y, batch_x_mark, batch_y_mark)
            pred_change = pred_cf[:, 0, -1] - pred[:, 0, -1]
            expected = causal_gain * delta
            loss_resp = criterion(pred_change, expected)

            x_dist = batch_x.clone()
            dist_delta = sample_delta(batch_x.shape[0], args_cli.delta, exp.device)
            x_dist[:, -1, 1:20] = x_dist[:, -1, 1:20] + dist_delta.view(-1, 1)
            pred_dist = forward_model(exp, args, x_dist, batch_y, batch_x_mark, batch_y_mark)
            dist_change = pred_dist[:, 0, -1] - pred[:, 0, -1]
            loss_dist = criterion(dist_change, torch.zeros_like(dist_change))

            loss = loss_pred + args_cli.lambda_resp * loss_resp + args_cli.lambda_dist * loss_dist
            loss.backward()
            optimizer.step()

            denom = torch.sum(expected.detach() * expected.detach()).item()
            slope = (
                torch.sum(pred_change.detach() * expected.detach()).item() / denom
                if denom > 1e-12
                else float("nan")
            )
            running.append(loss.item())
            running_pred.append(loss_pred.item())
            running_resp.append(loss_resp.item())
            running_dist.append(loss_dist.item())
            running_slope.append(slope)

        val_metrics = evaluate(exp, args, val_loader, criterion, causal_gain)
        test_metrics = evaluate(exp, args, test_loader, criterion, causal_gain)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "train_pred_loss": float(np.mean(running_pred)),
            "train_resp_loss": float(np.mean(running_resp)),
            "train_dist_loss": float(np.mean(running_dist)),
            "train_response_slope": float(np.mean(running_slope)),
            "val": val_metrics,
            "test": test_metrics,
        }
        log["epochs"].append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        # Model selection prioritizes response correctness while keeping prediction
        # loss visible. This is intentionally simple for the first CRR experiment.
        score = val_metrics["resp_loss"] + args_cli.selection_pred_weight * val_metrics["pred_loss"]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(exp.model.state_dict(), best_path)
            print(f"Saved best checkpoint to {best_path} (score={score:.6f})")
        else:
            bad_epochs += 1
            print(f"Early-stop counter: {bad_epochs}/{args_cli.patience}")
            if bad_epochs >= args_cli.patience:
                break

    log["best_epoch"] = best_epoch
    log["best_score"] = best_score
    with open(os.path.join(args_cli.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Finished CRR training. Best epoch={best_epoch}, best score={best_score:.6f}")


if __name__ == "__main__":
    main()
