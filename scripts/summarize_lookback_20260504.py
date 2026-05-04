import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "duet_lookback_20260504"
OUT_DIR = ROOT / "results" / "lookback_20260504"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(mapping, *keys, default=float("nan")):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def collect_records():
    records = []
    if not RESULT_ROOT.exists():
        return records

    for result_json in sorted(RESULT_ROOT.glob("lookback_seq*_seed*_lookback_20260504/duet_synthetic_results.json")):
        payload = read_json(result_json)
        meta = payload["metadata"]
        config = meta["config"]
        eval_result = payload["evaluation"]
        curve_json = result_json.parent / "duet_baseline_delta_curve.json"
        curve = read_json(curve_json)["delta_curve"]["summary"] if curve_json.exists() else {}

        cause = eval_result["h1"]["cause_last_shift_plus_delta"]
        dist = eval_result["h1"]["distractors_last_shift_plus_delta"]
        target_zero = eval_result["h1"]["target_zero"]
        obs = eval_result["observational"]
        records.append(
            {
                "model": "DUET-Mix baseline",
                "seq_len": int(config["seq_len"]),
                "pred_len": int(config["pred_len"]),
                "seed": int(config["seed"]),
                "target_mse": obs["target_mse"],
                "target_mae": obs["target_mae"],
                "expected_h1": cause["true_change_abs_mean"],
                "pred_h1": cause["pred_change_abs_mean"],
                "h1_ire": cause["ire_mae"],
                "h1_slope": cause["response_slope"],
                "last_dist_false": dist["pred_change_abs_mean"],
                "target_zero": target_zero["pred_change_abs_mean"],
                "curve_slope": safe_get(curve, "curve_slope_from_means"),
                "curve_ire": safe_get(curve, "curve_ire_mae_mean"),
                "path": str(result_json.parent.relative_to(ROOT)),
            }
        )
    return sorted(records, key=lambda row: (row["seq_len"], row["seed"]))


def write_csv(path, records):
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def fmt(value, digits=4):
    if value is None:
        return "--"
    try:
        if np.isnan(value):
            return "--"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def write_markdown(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Look-back Window Robustness 2026-05-04\n\n")
        f.write(
            "DUET-Mix baseline is trained with different history lengths on the same "
            "controlled synthetic benchmark. The prediction length is fixed to 96 and "
            "the expected H1 response for `delta=+5` remains about 5.000.\n\n"
        )
        if not records:
            f.write("No records found yet.\n")
            return
        f.write("| Seq len | Target MSE | Target MAE | Pred. H1 | H1 IRE | H1 Slope | Target-zero | Curve slope | Curve IRE |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in records:
            f.write(
                "| {seq} | {mse} | {mae} | {pred} | {ire} | {slope} | {tz} | {cslope} | {cire} |\n".format(
                    seq=row["seq_len"],
                    mse=fmt(row["target_mse"]),
                    mae=fmt(row["target_mae"]),
                    pred=fmt(row["pred_h1"]),
                    ire=fmt(row["h1_ire"]),
                    slope=fmt(row["h1_slope"]),
                    tz=fmt(row["target_zero"]),
                    cslope=fmt(row["curve_slope"]),
                    cire=fmt(row["curve_ire"]),
                )
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "The key diagnostic is whether `H1 Slope` remains near zero while "
            "`Target-zero` stays large. That pattern means the model still relies on "
            "target-history shortcuts rather than the last-step driver, even when the "
            "look-back window changes.\n"
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    write_csv(OUT_DIR / "duet_lookback_records.csv", records)
    write_markdown(OUT_DIR / "summary.md", records)
    print(f"records={len(records)}")
    print(f"summary={OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
