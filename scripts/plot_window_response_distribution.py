import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    "dark": "#2F3437",
    "grid": "#D7DDE2",
    "gray": "#8A8F94",
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )


def polish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6F767D")
    ax.spines["bottom"].set_color("#6F767D")
    ax.tick_params(color="#6F767D", labelcolor=PALETTE["dark"], width=0.8)
    ax.grid(True, axis="y", color=PALETTE["grid"], linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def display_label(model, variant):
    if variant == "baseline":
        return model
    pretty = {"ft01": "RIR FT01", "ft03": "RIR FT03"}.get(variant, variant)
    return f"{model} + {pretty}"


def read_eval_json(path):
    json_path = path.parent / "seeded_synthetic_eval.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def collect_records(input_roots):
    frames = []
    for input_root in input_roots:
        root = Path(input_root)
        for csv_path in root.rglob("window_h1_response.csv"):
            meta = read_eval_json(csv_path)
            if meta is None:
                continue
            df = pd.read_csv(csv_path)
            model = meta.get("model", "unknown")
            variant = meta.get("variant", "baseline")
            df["model"] = model
            df["variant"] = variant
            df["label"] = display_label(model, variant)
            df["source"] = str(csv_path.relative_to(root))
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    rows = []
    for label, sub in df.groupby("label", sort=True):
        rows.append(
            {
                "label": label,
                "n": int(len(sub)),
                "pred_mean": float(sub["pred_change"].mean()),
                "pred_std": float(sub["pred_change"].std(ddof=1)),
                "pred_q05": float(sub["pred_change"].quantile(0.05)),
                "pred_q50": float(sub["pred_change"].quantile(0.50)),
                "pred_q95": float(sub["pred_change"].quantile(0.95)),
                "expected_mean": float(sub["expected_change"].mean()),
                "abs_error_mean": float(sub["abs_error"].mean()),
                "sign_accuracy": float(sub["sign_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_histogram(df, labels, output_dir, name):
    if labels:
        df = df[df["label"].isin(labels)]
    if df.empty:
        raise ValueError("No window-response records matched the requested labels.")

    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [PALETTE["gray"], PALETTE["orange"], PALETTE["green"], PALETTE["blue"], PALETTE["red"]]
    bins = np.linspace(
        min(-0.5, float(df["pred_change"].quantile(0.01)) - 0.25),
        max(5.5, float(df["pred_change"].quantile(0.99)) + 0.25),
        42,
    )

    for idx, (label, sub) in enumerate(df.groupby("label", sort=False)):
        ax.hist(
            sub["pred_change"].to_numpy(),
            bins=bins,
            density=True,
            alpha=0.28,
            color=colors[idx % len(colors)],
            edgecolor=colors[idx % len(colors)],
            linewidth=0.8,
            label=label,
        )
        ax.axvline(sub["pred_change"].mean(), color=colors[idx % len(colors)], lw=1.8)

    expected = float(df["expected_change"].mean())
    ax.axvline(expected, color=PALETTE["dark"], lw=1.9, linestyle=(0, (4, 3)), label="Expected response")
    ax.set_xlabel("Per-window predicted H1 target response")
    ax.set_ylabel("Density")
    ax.set_title("Window-level response distribution", loc="left", weight="bold")
    polish_axes(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02), borderaxespad=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        plt.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", dpi=260)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Root directory containing seeded_synthetic_eval.json and window_h1_response.csv files. Repeat for baseline/repair roots.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "figures"))
    parser.add_argument("--name", default="window_response_distribution")
    parser.add_argument("--label", action="append", default=None, help="Optional display label to include. Repeat to plot selected groups.")
    args = parser.parse_args()

    df = collect_records(args.input_root)
    if df.empty:
        raise FileNotFoundError("No window_h1_response.csv files found under the provided roots.")
    output_dir = Path(args.output_dir)
    summary = summarize(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{args.name}_summary.csv", index=False)
    plot_histogram(df, args.label, output_dir, args.name)
    print(summary.to_string(index=False))
    print(f"saved={output_dir / (args.name + '.pdf')}")


if __name__ == "__main__":
    main()
