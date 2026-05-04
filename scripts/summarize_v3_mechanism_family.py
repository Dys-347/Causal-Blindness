import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_std(values):
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def h1_row(path, intervention):
    for row in read_csv_rows(path):
        if row.get("intervention") == intervention:
            return row
    raise KeyError(f"{intervention} not found in {path}")


def curve_summary_from_csv(path):
    if not path.exists():
        return {
            "curve_slope": float("nan"),
            "curve_ire": float("nan"),
            "curve_ratio": float("nan"),
        }
    rows = read_csv_rows(path)
    pred = []
    true = []
    ires = []
    ratios = []
    for row in rows:
        delta = fnum(row.get("delta"))
        if abs(delta) <= 1e-12:
            continue
        pred.append(fnum(row.get("pred_mean")))
        true.append(fnum(row.get("expected_mean")))
        ires.append(fnum(row.get("ire_mae")))
        ratios.append(fnum(row.get("response_ratio")))
    denom = sum(x * x for x in true)
    slope = sum(p * t for p, t in zip(pred, true)) / denom if denom > 1e-12 else float("nan")
    return {
        "curve_slope": slope,
        "curve_ire": statistics.mean(ires) if ires else float("nan"),
        "curve_ratio": statistics.mean(ratios) if ratios else float("nan"),
    }


def parse_result_dir(path):
    baseline_json = path / "duet_synthetic_results.json"
    rir_json = path / "duet_crr_results.json"
    if baseline_json.exists():
        js = read_json(baseline_json)
        variant = "baseline"
        h1_path = path / "duet_h1_summary.csv"
        curve_path = path / "duet_baseline_delta_curve.csv"
    elif rir_json.exists():
        js = read_json(rir_json)
        variant = "rir"
        h1_path = path / "duet_crr_h1_summary.csv"
        curve_path = path / "duet_crr_delta_curve.csv"
    else:
        return None

    meta = js["metadata"]
    cfg = meta.get("config", {})
    mechanism = meta.get("mechanism_metadata", {}).get("mechanism", "unknown")
    obs = js["evaluation"]["observational"]
    cause = h1_row(h1_path, "cause_last_shift_plus_delta")
    dist = h1_row(h1_path, "distractors_last_shift_plus_delta")
    target_zero = h1_row(h1_path, "target_zero")
    return {
        "mechanism": mechanism,
        "variant": variant,
        "seed": cfg.get("seed"),
        "target_mse": fnum(obs.get("target_mse")),
        "target_mae": fnum(obs.get("target_mae")),
        "expected_h1": fnum(cause.get("true_change_abs_mean")),
        "pred_h1": fnum(cause.get("pred_change_abs_mean")),
        "h1_ire": fnum(cause.get("ire_mae")),
        "h1_slope": fnum(cause.get("response_slope")),
        "last_dist_false": fnum(dist.get("pred_change_abs_mean")),
        "target_zero": fnum(target_zero.get("pred_change_abs_mean")),
        **curve_summary_from_csv(curve_path),
        "path": str(path),
    }


def collect_records(input_root):
    records = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_dir():
            continue
        rec = parse_result_dir(path)
        if rec is not None:
            records.append(rec)
    return records


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    metrics = [
        "target_mse",
        "target_mae",
        "expected_h1",
        "pred_h1",
        "h1_ire",
        "h1_slope",
        "last_dist_false",
        "target_zero",
        "curve_slope",
        "curve_ire",
        "curve_ratio",
    ]
    groups = {}
    for row in records:
        key = (row["mechanism"], row["variant"])
        groups.setdefault(key, []).append(row)
    out = []
    for (mechanism, variant), rows in sorted(groups.items()):
        rec = {"mechanism": mechanism, "variant": variant, "n": len(rows)}
        for metric in metrics:
            mean, std = mean_std([row[metric] for row in rows])
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
        out.append(rec)
    return out


def fmt_pm(mean, std):
    if math.isnan(mean):
        return "nan"
    return f"{mean:.4f} +- {std:.4f}"


def write_markdown(path, summary_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# V3 Synthetic Mechanism-Family Summary\n\n")
        f.write("This summary is for reviewer-defense experiments beyond the original one-lag linear mechanism.\n\n")
        f.write("| Mechanism | Variant | n | Target MSE | Pred H1 | H1 IRE | H1 slope | Curve slope | Curve IRE |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {mechanism} | {variant} | {n} | {target_mse} | {pred_h1} | {h1_ire} | {h1_slope} | {curve_slope} | {curve_ire} |\n".format(
                    mechanism=row["mechanism"],
                    variant=row["variant"],
                    n=row["n"],
                    target_mse=fmt_pm(row["target_mse_mean"], row["target_mse_std"]),
                    pred_h1=fmt_pm(row["pred_h1_mean"], row["pred_h1_std"]),
                    h1_ire=fmt_pm(row["h1_ire_mean"], row["h1_ire_std"]),
                    h1_slope=fmt_pm(row["h1_slope_mean"], row["h1_slope_std"]),
                    curve_slope=fmt_pm(row["curve_slope_mean"], row["curve_slope_std"]),
                    curve_ire=fmt_pm(row["curve_ire_mean"], row["curve_ire_std"]),
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default="results/v3_mechanism_family_20260504")
    args = parser.parse_args()

    records = collect_records(Path(args.input_root))
    summary_rows = summarize(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "mechanism_family_records.csv", records)
    write_csv(out_dir / "mechanism_family_summary.csv", summary_rows)
    write_markdown(out_dir / "summary.md", summary_rows)
    print(f"records={len(records)}")
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()

