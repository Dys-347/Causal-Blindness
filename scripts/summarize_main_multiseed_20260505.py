import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mean_std(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return mean, std


def discover_records(input_root):
    root = Path(input_root)
    records = []
    for path in root.rglob("seeded_synthetic_eval.json"):
        rec = read_json(path)
        rec["record_path"] = str(path.relative_to(root))
        records.append(rec)
    return records


def aggregate(records):
    grouped = {}
    for rec in records:
        variant = rec.get("variant", "baseline")
        grouped.setdefault((rec["model"], variant), []).append(rec)

    rows = []
    for model, variant in sorted(grouped):
        vals = grouped[(model, variant)]
        label = model if variant == "baseline" else f"{model} + {variant}"
        row = {"model": model, "variant": variant, "label": label, "n": len(vals)}
        for key in [
            "target_mse",
            "target_mae",
            "pred_h1_abs_mean",
            "pred_h1_signed_mean",
            "h1_ire",
            "h1_slope",
            "h1_sign_accuracy",
            "h1_response_ge_20pct",
            "target_zero_mpd",
        ]:
            mean, std = fmt_mean_std([v.get(key) for v in vals])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = []
    lines.append("# Main Multi-Seed Summary")
    lines.append("")
    lines.append("| Model | n | Target MSE | Target MAE | Pred H1 | Signed H1 | H1 IRE | H1 slope | Sign acc. | >=20% resp. | Target-zero MPD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {label} | {n} | {mse:.4f} +- {mse_std:.4f} | {mae:.4f} +- {mae_std:.4f} | "
            "{pred:.4f} +- {pred_std:.4f} | {signed:.4f} +- {signed_std:.4f} | "
            "{ire:.4f} +- {ire_std:.4f} | {slope:.4f} +- {slope_std:.4f} | "
            "{sign:.3f} +- {sign_std:.3f} | {ge20:.3f} +- {ge20_std:.3f} | "
            "{zero:.4f} +- {zero_std:.4f} |".format(
                label=row["label"],
                n=row["n"],
                mse=row["target_mse_mean"],
                mse_std=row["target_mse_std"],
                mae=row["target_mae_mean"],
                mae_std=row["target_mae_std"],
                pred=row["pred_h1_abs_mean_mean"],
                pred_std=row["pred_h1_abs_mean_std"],
                signed=row["pred_h1_signed_mean_mean"],
                signed_std=row["pred_h1_signed_mean_std"],
                ire=row["h1_ire_mean"],
                ire_std=row["h1_ire_std"],
                slope=row["h1_slope_mean"],
                slope_std=row["h1_slope_std"],
                sign=row["h1_sign_accuracy_mean"],
                sign_std=row["h1_sign_accuracy_std"],
                ge20=row["h1_response_ge_20pct_mean"],
                ge20_std=row["h1_response_ge_20pct_std"],
                zero=row["target_zero_mpd_mean"],
                zero_std=row["target_zero_mpd_std"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    records = discover_records(args.input_root)
    rows = aggregate(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "main_multiseed_summary.csv", rows)
    write_markdown(out_dir / "summary.md", rows)
    write_csv(out_dir / "main_multiseed_records.csv", records)
    print(f"records={len(records)}")
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
