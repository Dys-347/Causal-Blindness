import argparse
import json
import os

import numpy as np
import pandas as pd

from synthetic_mechanism_utils import metadata_path_for


def make_cause(n_steps, rng):
    t = np.arange(n_steps, dtype=np.float64)
    seasonal = (
        0.75 * np.sin(2.0 * np.pi * t / 48.0)
        + 0.35 * np.sin(2.0 * np.pi * t / 137.0 + 0.4)
        + 0.25 * np.cos(2.0 * np.pi * t / 288.0)
    )
    ar = np.zeros(n_steps, dtype=np.float64)
    noise = rng.normal(0.0, 0.25, size=n_steps)
    for idx in range(1, n_steps):
        ar[idx] = 0.82 * ar[idx - 1] + noise[idx]
    return seasonal + 0.45 * ar


def make_distractors(n_steps, rng, n_distractors):
    t = np.arange(n_steps, dtype=np.float64)
    columns = {}
    for idx in range(1, n_distractors + 1):
        period = rng.uniform(24.0, 360.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.25, 1.1)
        series = amp * np.sin(2.0 * np.pi * t / period + phase)
        if idx % 3 == 0:
            walk = np.cumsum(rng.normal(0.0, 0.02, size=n_steps))
            series = series + walk
        if idx % 4 == 0:
            series = series + 0.25 * np.cos(2.0 * np.pi * t / rng.uniform(80.0, 500.0))
        series = series + rng.normal(0.0, 0.12, size=n_steps)
        columns[f"Dist_{idx:02d}"] = series
    return columns


def make_target(cause, mechanism, rng, noise_std):
    target = np.zeros_like(cause)
    if mechanism == "linear_one_lag":
        target[1:] = 2.0 * cause[:-1]
        response = {"type": "linear_last_cause", "raw_gain": 2.0}
        equation = "Y_t = 2.0 * C_{t-1} + epsilon_t"
    elif mechanism == "linear_multi_lag":
        target[1:] += 1.5 * cause[:-1]
        target[3:] += 0.8 * cause[:-3]
        response = {"type": "linear_last_cause", "raw_gain": 1.5}
        equation = "Y_t = 1.5 * C_{t-1} + 0.8 * C_{t-3} + epsilon_t"
    elif mechanism == "nonlinear_sin":
        amplitude = 2.0
        target[1:] = amplitude * np.sin(cause[:-1])
        response = {"type": "sin_last_cause", "amplitude": amplitude}
        equation = "Y_t = 2.0 * sin(C_{t-1}) + epsilon_t"
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    target += rng.normal(0.0, noise_std, size=target.shape[0])
    return target, response, equation


def write_dataset(output_dir, mechanism, n_steps, n_distractors, seed, noise_std):
    rng = np.random.default_rng(seed)
    cause = make_cause(n_steps, rng)
    distractors = make_distractors(n_steps, rng, n_distractors)
    target, response, equation = make_target(cause, mechanism, rng, noise_std)

    dates = pd.date_range("2020-01-01", periods=n_steps, freq="h")
    data = {"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "Cause_Var": cause}
    data.update(distractors)
    data["Target_Var"] = target

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"synthetic_{mechanism}.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)

    metadata = {
        "mechanism": mechanism,
        "equation": equation,
        "n_steps": int(n_steps),
        "n_distractors": int(n_distractors),
        "seed": int(seed),
        "noise_std": float(noise_std),
        "cause_col": "Cause_Var",
        "target_col": "Target_Var",
        "distractor_cols": list(distractors.keys()),
        "h1_response": response,
        "channel_order_after_loading": ["Cause_Var"] + list(distractors.keys()) + ["Target_Var"],
        "splits": {"train": 0.7, "val": 0.1, "test": 0.2},
        "note": "Horizon-1 labels are for last-step interventions on standardized Cause_Var inputs. Nonlinear labels are sample-dependent and should be computed from this metadata.",
    }
    with open(metadata_path_for(csv_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return csv_path, metadata_path_for(csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset/causal_r1_mechanism_family")
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=["linear_one_lag", "linear_multi_lag", "nonlinear_sin"],
    )
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--n-distractors", type=int, default=19)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260504)
    args = parser.parse_args()

    outputs = []
    for offset, mechanism in enumerate(args.mechanisms):
        csv_path, meta_path = write_dataset(
            output_dir=args.output_dir,
            mechanism=mechanism,
            n_steps=args.n_steps,
            n_distractors=args.n_distractors,
            seed=args.seed + offset,
            noise_std=args.noise_std,
        )
        outputs.append({"mechanism": mechanism, "csv": csv_path, "metadata": meta_path})
        print(f"[OK] {mechanism}: {csv_path} ({meta_path})")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"outputs": outputs}, f, indent=2)
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

