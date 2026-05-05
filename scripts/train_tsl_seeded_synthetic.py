import argparse
import os
import random
import json
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


MODEL_OVERRIDES = {
    "DLinear": {},
    "PatchTST": {},
    "iTransformer": {},
    "Crossformer": {},
    "TimeMixer": {
        "label_len": 0,
        "d_model": 16,
        "d_ff": 32,
        "down_sampling_layers": 3,
        "down_sampling_method": "avg",
        "down_sampling_window": 2,
        "learning_rate": 0.01,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(model_name, seed, batch_size, learning_rate, epochs, patience, data_path):
    overrides = MODEL_OVERRIDES[model_name]
    args = argparse.Namespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"Synthetic_Multi_{model_name}_seed{seed}",
        model=model_name,
        des=f"Exp_Causal_Check_seed{seed}",
        data="SyntheticMulti",
        root_path="./dataset/",
        data_path=data_path,
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
        train_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
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
        seed=seed,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_setting(args, seed):
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_"
        f"eb{args.embed}_dt{args.distil}_{args.des}_0"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_OVERRIDES.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--data-path",
        default=os.environ.get("CAUSAL_R1_SYNTHETIC_CSV", "synthetic_multivariate.csv"),
    )
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)
    args = build_args(
        model_name=args_cli.model,
        seed=args_cli.seed,
        batch_size=args_cli.batch_size,
        learning_rate=args_cli.learning_rate,
        epochs=args_cli.epochs,
        patience=args_cli.patience,
        data_path=args_cli.data_path,
    )
    exp = Exp_Long_Term_Forecast(args)
    setting = build_setting(args, args_cli.seed)

    print(f"[INFO] model={args_cli.model} seed={args_cli.seed}")
    print(f"[INFO] setting={setting}")
    print(f"[INFO] checkpoints={os.path.join(args.checkpoints, setting)}")

    exp.train(setting)
    exp.test(setting)

    checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    if args_cli.run_dir:
        os.makedirs(args_cli.run_dir, exist_ok=True)
        with open(os.path.join(args_cli.run_dir, "train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args_cli.model,
                    "seed": args_cli.seed,
                    "setting": setting,
                    "checkpoint_path": checkpoint_path,
                    "data_path": args.data_path,
                    "batch_size": args_cli.batch_size,
                    "learning_rate": args_cli.learning_rate,
                    "epochs": args_cli.epochs,
                    "patience": args_cli.patience,
                },
                f,
                indent=2,
            )
    print(f"[DONE] checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
