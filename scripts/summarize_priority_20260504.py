import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "priority_20260504"
DUET_OUT = ROOT / "results" / "duet_priority_20260504"
ETTH_OUT = ROOT / "results" / "etth1_priority_20260504"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.6f}"


def curve_summary_from_csv(path):
    if not path.exists():
        return {
            "curve_slope": float("nan"),
            "curve_corr": float("nan"),
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
    if len(pred) >= 2 and statistics.pstdev(pred) > 1e-12 and statistics.pstdev(true) > 1e-12:
        mp = statistics.mean(pred)
        mt = statistics.mean(true)
        corr = sum((p - mp) * (t - mt) for p, t in zip(pred, true))
        corr /= math.sqrt(sum((p - mp) ** 2 for p in pred) * sum((t - mt) ** 2 for t in true))
    else:
        corr = float("nan")
    return {
        "curve_slope": slope,
        "curve_corr": corr,
        "curve_ire": statistics.mean(ires) if ires else float("nan"),
        "curve_ratio": statistics.mean(ratios) if ratios else float("nan"),
    }


def h1_row(path, intervention):
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("intervention") == intervention:
            return row
    raise KeyError(f"{intervention} not found in {path}")


def duet_record(label, variant, base_dir, h1_file, json_file, curve_file=None):
    h1_path = base_dir / h1_file
    js = read_json(base_dir / json_file)
    meta = js["metadata"]
    obs = js["evaluation"]["observational"]
    cause = h1_row(h1_path, "cause_last_shift_plus_delta")
    dist = h1_row(h1_path, "distractors_last_shift_plus_delta")
    target_zero = h1_row(h1_path, "target_zero")
    curve = curve_summary_from_csv(base_dir / curve_file) if curve_file else curve_summary_from_csv(Path("__missing__"))
    cfg = meta.get("config", {})
    return {
        "label": label,
        "variant": variant,
        "ci": meta.get("ci"),
        "seed": cfg.get("seed"),
        "target_mse": obs["target_mse"],
        "target_mae": obs["target_mae"],
        "expected_h1": fnum(cause["true_change_abs_mean"]),
        "pred_h1": fnum(cause["pred_change_abs_mean"]),
        "h1_ire": fnum(cause["ire_mae"]),
        "h1_slope": fnum(cause["response_slope"]),
        "last_dist_false": fnum(dist["pred_change_abs_mean"]),
        "target_zero": fnum(target_zero["pred_change_abs_mean"]),
        **curve,
        "path": str(base_dir.relative_to(ROOT)),
    }


def collect_duet():
    records = []
    records.append(
        duet_record(
            "DUET-Mix baseline",
            "baseline",
            ROOT / "results" / "duet_crr_20260503" / "baseline_curve_ci0_20260503",
            "duet_h1_summary.csv",
            "duet_synthetic_results.json",
            "duet_baseline_delta_curve.csv",
        )
    )
    records.append(
        duet_record(
            "DUET-Mix + RIR",
            "rir",
            ROOT / "results" / "duet_crr_20260503" / "crr_ci0_lam005_20260503",
            "duet_crr_h1_summary.csv",
            "duet_crr_results.json",
            "duet_crr_delta_curve.csv",
        )
    )
    for seed in [20260504, 20260505]:
        records.append(
            duet_record(
                "DUET-Mix baseline",
                "baseline",
                DUET_OUT / f"mix_baseline_seed{seed}_priority_20260504",
                "duet_h1_summary.csv",
                "duet_synthetic_results.json",
                None,
            )
        )
        records.append(
            duet_record(
                "DUET-Mix + RIR",
                "rir",
                DUET_OUT / f"mix_rir_seed{seed}_priority_20260504",
                "duet_crr_h1_summary.csv",
                "duet_crr_results.json",
                "duet_crr_delta_curve.csv",
            )
        )
    records.append(
        duet_record(
            "DUET-CI + RIR",
            "ci_rir",
            DUET_OUT / "ci1_rir_seed20260503_priority_20260504",
            "duet_crr_h1_summary.csv",
            "duet_crr_results.json",
            "duet_crr_delta_curve.csv",
        )
    )
    return records


def collect_etth():
    records = []
    dirs = [
        ROOT / "results" / "etth1_rir_side_effect_20260503" / "baseline_20260503",
        ROOT / "results" / "etth1_rir_side_effect_20260503" / "rir_20260503",
    ]
    dirs += [p for p in ETTH_OUT.iterdir() if p.is_dir()]
    for d in sorted(dirs):
        js = read_json(d / "etth1_rir_side_effect_results.json")
        meta = js["metadata"]
        cfg = meta["config"]
        pred = js["prediction_metrics"]
        curve = js["delta_curve"]["summary"]
        records.append(
            {
                "variant": meta["variant"],
                "seed": cfg["seed"],
                "pred_len": cfg["pred_len"],
                "all_mse": pred["all_mse"],
                "all_mae": pred["all_mae"],
                "raw_ot_mse": pred["raw_ot_mse"],
                "raw_ot_mae": pred["raw_ot_mae"],
                "semi_target_mse": pred["semi_target_mse"],
                "semi_target_mae": pred["semi_target_mae"],
                "curve_slope": curve["curve_slope_from_means"],
                "curve_corr": curve["curve_corr_from_means"],
                "curve_ire": curve["curve_ire_mae_mean"],
                "curve_ratio": curve["curve_response_ratio_mean"],
                "path": str(d.relative_to(ROOT)),
            }
        )
    return records


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, group_keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda item: str(item[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(vals)
        for m in metrics:
            xs = [float(v[m]) for v in vals if not math.isnan(float(v[m]))]
            rec[f"{m}_mean"] = statistics.mean(xs) if xs else float("nan")
            rec[f"{m}_std"] = statistics.stdev(xs) if len(xs) >= 2 else 0.0
        out.append(rec)
    return out


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    duet = collect_duet()
    etth = collect_etth()
    write_csv(OUT_DIR / "duet_multiseed_records.csv", duet)
    write_csv(OUT_DIR / "etth1_augmented_records.csv", etth)

    duet_summary = summarize(
        duet,
        ["label"],
        ["target_mse", "target_mae", "pred_h1", "h1_ire", "h1_slope", "last_dist_false", "target_zero", "curve_slope", "curve_ire"],
    )
    etth_summary = summarize(
        etth,
        ["variant", "pred_len"],
        ["all_mse", "all_mae", "raw_ot_mse", "raw_ot_mae", "semi_target_mse", "semi_target_mae", "curve_slope", "curve_ire"],
    )
    write_csv(OUT_DIR / "duet_multiseed_summary.csv", duet_summary)
    write_csv(OUT_DIR / "etth1_augmented_summary.csv", etth_summary)

    md = []
    md.append("# Priority Experiments Summary 2026-05-04\n")
    md.append("## DUET Records\n")
    md.append(markdown_table(duet, ["label", "seed", "ci", "target_mse", "target_mae", "pred_h1", "h1_ire", "h1_slope", "last_dist_false", "target_zero", "curve_slope", "curve_ire"]))
    md.append("\n## DUET Mean/Std\n")
    md.append(markdown_table(duet_summary, list(duet_summary[0].keys())))
    md.append("\n## ETTh1 Records\n")
    md.append(markdown_table(etth, ["variant", "seed", "pred_len", "all_mse", "raw_ot_mse", "raw_ot_mae", "semi_target_mse", "curve_slope", "curve_ire"]))
    md.append("\n## ETTh1 Mean/Std\n")
    md.append(markdown_table(etth_summary, list(etth_summary[0].keys())))
    with open(OUT_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))

    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
