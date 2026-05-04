from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#B07AA1",
    "teal": "#76B7B2",
    "gray": "#8A8F94",
    "dark": "#2F3437",
    "grid": "#D7DDE2",
    "paper": "#FFFFFF",
    "blue_fill": "#EAF1F8",
    "orange_fill": "#FDEAD8",
    "green_fill": "#EAF4EA",
    "gray_fill": "#F0F2F4",
}


def set_style() -> None:
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
            "lines.solid_capstyle": "round",
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
        }
    )


def polish_axes(ax, grid_axis="y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6F767D")
    ax.spines["bottom"].set_color("#6F767D")
    ax.tick_params(color="#6F767D", labelcolor=PALETTE["dark"], width=0.8)
    ax.grid(
        True,
        axis=grid_axis,
        color=PALETTE["grid"],
        linestyle="--",
        linewidth=0.7,
        alpha=0.6,
    )
    ax.set_axisbelow(True)


def savefig(name: str) -> None:
    for ext in ["pdf", "png"]:
        plt.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=260)
    plt.close()


def draw_box(
    ax,
    xy,
    w,
    h,
    text,
    fc="#FFFFFF",
    ec="#333333",
    lw=1.1,
    fontsize=9.2,
    color=None,
    rounding=0.028,
    zorder=2,
):
    color = PALETTE["dark"] if color is None else color
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        linespacing=1.12,
        zorder=zorder + 1,
    )
    return box


def draw_arrow(
    ax,
    p1,
    p2,
    color="#333333",
    lw=1.4,
    mutation_scale=12,
    style="-|>",
    alpha=1.0,
    rad=0.0,
    linestyle="-",
    zorder=4,
):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_channel_strip(ax, x, y, w, h, active, active_color) -> None:
    n = 7
    gap = w * 0.018
    cell_w = (w - gap * (n - 1)) / n
    for i in range(n):
        fc = active_color if i in active else "#D8DEE3"
        alpha = 0.98 if i in active else 0.78
        rect = Rectangle(
            (x + i * (cell_w + gap), y),
            cell_w,
            h,
            facecolor=fc,
            edgecolor="white",
            linewidth=0.6,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)


def draw_input_card(ax, xy, w, h, symbol, title, color, fill, active) -> None:
    card = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.026",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.35,
        zorder=2,
    )
    ax.add_patch(card)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h * 0.76,
        symbol,
        ha="center",
        va="center",
        fontsize=10.0,
        color=PALETTE["dark"],
        zorder=4,
    )
    ax.text(
        xy[0] + w / 2,
        xy[1] + h * 0.43,
        title,
        ha="center",
        va="center",
        fontsize=7.8,
        color=PALETTE["dark"],
        zorder=4,
    )
    draw_channel_strip(ax, xy[0] + w * 0.15, xy[1] + h * 0.15, w * 0.70, h * 0.11, active, color)


def conceptual_diagram() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25))
    fig.subplots_adjust(wspace=0.13)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    x = np.linspace(0.08, 0.92, 240)
    y = 0.51 + 0.17 * np.sin(2.0 * np.pi * (x - 0.12)) + 0.035 * np.sin(5.0 * np.pi * x)

    ax = axes[0]
    ax.text(0.05, 0.93, "A. Observational shortcut", fontsize=12, weight="bold", color=PALETTE["dark"])
    ax.fill_between(x, y - 0.105, y + 0.105, color=PALETTE["blue_fill"], alpha=0.52, zorder=1)
    ax.fill_between(x, y - 0.055, y + 0.055, color="#D7E4F2", alpha=0.82, zorder=2)
    ax.plot(x, y, color=PALETTE["blue"], lw=2.5, zorder=3)

    pts_x = np.linspace(0.15, 0.86, 12)
    pts_y = np.interp(pts_x, x, y)
    offsets = np.array([0.012, -0.016, 0.007, -0.01, 0.018, -0.006, 0.011, -0.014, 0.006, -0.008, 0.013, -0.011])
    ax.scatter(
        pts_x,
        pts_y + offsets,
        s=26,
        color="#FFFFFF",
        edgecolor=PALETTE["blue"],
        linewidth=1.1,
        zorder=5,
    )
    ax.plot(x, y + 0.02, color=PALETTE["orange"], lw=2.0, linestyle=(0, (4, 3)), zorder=4)

    draw_box(
        ax,
        (0.13, 0.73),
        0.27,
        0.12,
        "target-history\nshortcut",
        fc=PALETTE["orange_fill"],
        ec=PALETTE["orange"],
        fontsize=9.2,
    )
    draw_arrow(ax, (0.40, 0.76), (0.54, 0.62), color=PALETTE["orange"], lw=1.4, rad=-0.08)
    ax.text(0.55, 0.17, "Low MSE only constrains\npredictions on the observed manifold", ha="center", fontsize=9.6, color=PALETTE["dark"])
    ax.text(0.72, 0.62, "training windows", ha="center", fontsize=8.5, color=PALETTE["blue"])

    ax = axes[1]
    ax.text(0.05, 0.93, "B. Driver intervention", fontsize=12, weight="bold", color=PALETTE["dark"])
    ax.fill_between(x, y - 0.1, y + 0.1, color=PALETTE["gray_fill"], alpha=0.8, zorder=1)
    ax.fill_between(x, y - 0.05, y + 0.05, color="#DDE2E6", alpha=0.9, zorder=2)
    ax.plot(x, y, color="#A1A9B0", lw=2.1, zorder=3)

    x0 = 0.43
    y0 = float(np.interp(x0, x, y))
    x1, y1 = 0.74, y0 + 0.26
    ax.scatter([x0], [y0], s=46, color=PALETTE["blue"], edgecolor="white", linewidth=1.1, zorder=6)
    ax.scatter([x1], [y1], s=50, color=PALETTE["orange"], edgecolor="white", linewidth=1.1, zorder=7)
    draw_arrow(ax, (x0 + 0.02, y0 + 0.02), (x1 - 0.015, y1 - 0.015), color=PALETTE["orange"], lw=2.2, mutation_scale=15)

    flat_y = y0 + 0.015
    correct_y = y1
    ax.plot([0.55, 0.9], [flat_y, flat_y], color=PALETTE["gray"], lw=2.0, linestyle=(0, (4, 3)), zorder=4)
    ax.plot([0.55, 0.9], [correct_y, correct_y], color=PALETTE["green"], lw=2.4, zorder=4)
    ax.fill_between([0.55, 0.9], [flat_y, flat_y], [correct_y, correct_y], color=PALETTE["green_fill"], alpha=0.72, zorder=2)

    ax.text(0.73, flat_y - 0.095, "shortcut:\nflat response", ha="center", fontsize=8.8, color=PALETTE["gray"])
    ax.text(0.73, correct_y + 0.055, "RIR / ideal:\ncausal response", ha="center", fontsize=8.8, color=PALETTE["green"])
    ax.text(0.33, 0.13, "Causal blindness = low error\nbut wrong intervention response", ha="center", fontsize=9.8, weight="bold", color=PALETTE["dark"])

    fig.text(
        0.5,
        0.01,
        "Observed correlations can be enough for MSE, while intervention tests reveal the missing driver response.",
        ha="center",
        fontsize=9.4,
        color="#555B60",
    )
    savefig("concept_causal_blindness")


def sensitivity_plot() -> None:
    models = ["iTransformer", "iTransformer\n+ RIR", "PatchTST", "Crossformer", "TimeMixer"]
    cause = np.array([0.0318, 1.5075, 0.0000, 0.0001, 0.0000])
    target = np.array([0.2385, 1.3417, 0.5362, 0.0483, 0.1752])
    x = np.arange(len(models))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    bars_c = ax.bar(
        x - w / 2,
        cause,
        w,
        label="Cause-last sensitivity",
        color=PALETTE["blue"],
        edgecolor="#FFFFFF",
        linewidth=0.7,
        zorder=3,
    )
    bars_t = ax.bar(
        x + w / 2,
        target,
        w,
        label="Target-last sensitivity",
        color=PALETTE["red"],
        edgecolor="#FFFFFF",
        linewidth=0.7,
        alpha=0.88,
        zorder=3,
    )

    ax.set_ylabel("Central-difference sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.72)
    polish_axes(ax, grid_axis="y")
    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.08), handlelength=1.2)
    ax.set_title("Functional sensitivity exposes target-history shortcuts", loc="left", pad=18, weight="bold")

    for bars in (bars_c, bars_t):
        for bar in bars:
            h = bar.get_height()
            if h > 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.035,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color=PALETTE["dark"],
                )

    ax.annotate(
        "response geometry\nmoves to the driver",
        xy=(x[1] - w / 2, cause[1]),
        xytext=(2.25, 1.52),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=PALETTE["dark"]),
        fontsize=8.7,
        ha="center",
        va="center",
        color=PALETTE["dark"],
    )
    savefig("functional_sensitivity_bars")


def rir_workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.93, "Randomized Intervention-Response Regularization", fontsize=12.5, weight="bold", color=PALETTE["dark"])
    ax.text(0.035, 0.875, "Three aligned forward passes; one shared forecaster.", fontsize=9.4, color="#5B6268")

    lane_y = {
        "causal": 0.68,
        "orig": 0.43,
        "dist": 0.18,
    }
    lane_meta = {
        "causal": ("causal shifted", "$X^{cf}$", PALETTE["orange"], PALETTE["orange_fill"], [0]),
        "orig": ("original", "$X$", PALETTE["blue"], PALETTE["blue_fill"], [0, 6]),
        "dist": ("distractor shifted", "$X^{dist}$", PALETTE["green"], PALETTE["green_fill"], [2, 3, 4]),
    }

    for lane, y in lane_y.items():
        title, symbol, color, fill, active = lane_meta[lane]
        draw_input_card(ax, (0.055, y), 0.155, 0.12, symbol, title, color, fill, active)

        draw_box(ax, (0.31, y), 0.155, 0.12, "shared\n$f_\\theta$", fc="#FFFFFF", ec="#707780", fontsize=9.2)
        draw_arrow(ax, (0.21, y + 0.06), (0.31, y + 0.06), color=color, lw=1.55, mutation_scale=12)

        out_label = "$\\hat{Y}^{cf}$" if lane == "causal" else "$\\hat{Y}^{dist}$" if lane == "dist" else "$\\hat{Y}$"
        draw_box(ax, (0.555, y), 0.13, 0.12, out_label, fc=fill if lane != "orig" else "#FFFFFF", ec=color if lane != "orig" else PALETTE["blue"], fontsize=10)
        draw_arrow(ax, (0.465, y + 0.06), (0.555, y + 0.06), color=color, lw=1.55, mutation_scale=12)

    draw_box(
        ax,
        (0.785, 0.725),
        0.165,
        0.12,
        "$\\mathcal{L}_{resp}$\nmatch response",
        fc=PALETTE["orange_fill"],
        ec=PALETTE["orange"],
        fontsize=9.0,
    )
    draw_box(
        ax,
        (0.785, 0.435),
        0.165,
        0.12,
        "$\\mathcal{L}_{pred}$\nforecast loss",
        fc=PALETTE["blue_fill"],
        ec=PALETTE["blue"],
        fontsize=9.0,
    )
    draw_box(
        ax,
        (0.785, 0.145),
        0.165,
        0.12,
        "$\\mathcal{L}_{dist}$\nnegative control",
        fc=PALETTE["green_fill"],
        ec=PALETTE["green"],
        fontsize=9.0,
    )

    draw_arrow(ax, (0.685, 0.74), (0.785, 0.785), color=PALETTE["orange"], lw=1.25, linestyle=(0, (2, 2)), mutation_scale=10)
    draw_arrow(ax, (0.685, 0.49), (0.785, 0.785), color=PALETTE["orange"], lw=1.05, linestyle=(0, (2, 2)), mutation_scale=10)
    draw_arrow(ax, (0.685, 0.49), (0.785, 0.495), color=PALETTE["blue"], lw=1.25, linestyle=(0, (2, 2)), mutation_scale=10)
    draw_arrow(ax, (0.685, 0.24), (0.785, 0.205), color=PALETTE["green"], lw=1.25, linestyle=(0, (2, 2)), mutation_scale=10)
    draw_arrow(ax, (0.685, 0.49), (0.785, 0.205), color=PALETTE["green"], lw=1.05, linestyle=(0, (2, 2)), mutation_scale=10)

    ax.text(
        0.505,
        0.055,
        "$\\mathcal{L}=\\mathcal{L}_{pred}+\\lambda_1\\mathcal{L}_{resp}+\\lambda_2\\mathcal{L}_{dist}$",
        ha="center",
        va="center",
        fontsize=13,
        color=PALETTE["dark"],
    )
    for x0, y0, c in [(0.074, 0.815, PALETTE["orange"]), (0.074, 0.565, PALETTE["blue"]), (0.074, 0.315, PALETTE["green"])]:
        ax.add_patch(Circle((x0, y0), 0.011, facecolor=c, edgecolor="white", linewidth=0.6, zorder=6))
    savefig("rir_workflow")


def load_duet_rir_seed_curves() -> pd.DataFrame:
    paths = [
        ROOT / "results" / "duet_crr_20260503" / "crr_ci0_lam010_20260503" / "duet_crr_delta_curve.csv",
        ROOT / "results" / "duet_priority_20260504" / "mix_rir_seed20260504_priority_20260504" / "duet_crr_delta_curve.csv",
        ROOT / "results" / "duet_priority_20260504" / "mix_rir_seed20260505_priority_20260504" / "duet_crr_delta_curve.csv",
    ]
    frames = []
    for idx, path in enumerate(paths):
        if path.exists():
            df = pd.read_csv(path)
            df["seed_id"] = idx
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def response_curves() -> None:
    itr_path = ROOT / "results" / "delta_response_itransformer_crr_curve_20260503" / "delta_response_curve_summary.csv"
    duet_path = ROOT / "results" / "duet_crr_20260503" / "duet_baseline_vs_rir_curve.csv"
    itr = pd.read_csv(itr_path)
    duet = pd.read_csv(duet_path)
    duet_rir_seeds = load_duet_rir_seed_curves()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)

    ax = axes[0]
    exp = itr[itr["model"] == "iTransformer"].sort_values("delta")
    ax.plot(exp["delta"], exp["expected_mean"], color=PALETTE["dark"], lw=1.9, linestyle=(0, (4, 3)), label="Expected")
    for model, label, color, marker in [
        ("iTransformer", "iTransformer", PALETTE["gray"], "o"),
        ("iTransformer_CRR_FT01", "iTransformer + RIR", PALETTE["orange"], "s"),
        ("iTransformer_CRR_FT03", "RIR stronger", PALETTE["green"], "^"),
    ]:
        sub = itr[itr["model"] == model].sort_values("delta")
        ax.plot(
            sub["delta"],
            sub["pred_mean"],
            marker=marker,
            markersize=4.0,
            lw=2.0,
            label=label,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    ax.set_title("iTransformer", loc="left", weight="bold")

    ax = axes[1]
    ax.plot(duet["delta"], duet["expected"], color=PALETTE["dark"], lw=1.9, linestyle=(0, (4, 3)), label="Expected")
    ax.plot(
        duet["delta"],
        duet["duet_baseline"],
        marker="o",
        markersize=4.0,
        lw=2.0,
        label="DUET-Mix",
        color=PALETTE["gray"],
        markeredgecolor="white",
        markeredgewidth=0.55,
    )
    if not duet_rir_seeds.empty:
        agg = (
            duet_rir_seeds.groupby("delta", as_index=False)
            .agg(expected=("expected_mean", "mean"), pred_mean=("pred_mean", "mean"), pred_std=("pred_mean", "std"))
            .sort_values("delta")
        )
        ax.fill_between(
            agg["delta"].to_numpy(),
            (agg["pred_mean"] - agg["pred_std"]).to_numpy(),
            (agg["pred_mean"] + agg["pred_std"]).to_numpy(),
            color=PALETTE["green"],
            alpha=0.18,
            linewidth=0,
            label="RIR seed band",
        )
        ax.plot(
            agg["delta"],
            agg["pred_mean"],
            marker="s",
            markersize=4.0,
            lw=2.2,
            label="DUET-Mix + RIR",
            color=PALETTE["green"],
            markeredgecolor="white",
            markeredgewidth=0.55,
        )
    else:
        ax.plot(duet["delta"], duet["duet_rir"], marker="s", lw=2.2, label="DUET-Mix + RIR", color=PALETTE["green"])
    ax.set_title("DUET-Mix", loc="left", weight="bold")

    for ax in axes:
        ax.axhline(0, color="#8A8F94", lw=0.8, zorder=0)
        ax.axvline(0, color="#8A8F94", lw=0.8, zorder=0)
        polish_axes(ax, grid_axis="both")
        ax.set_xlabel("Cause intervention $\\delta$")
        ax.set_xlim(-5.4, 5.4)
        ax.set_ylim(-5.65, 5.65)
        ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderaxespad=0.2)
    axes[0].set_ylabel("Predicted horizon-1 target change")
    fig.suptitle("RIR restores calibrated intervention-response curves", fontsize=12.5, weight="bold", y=1.02, color=PALETTE["dark"])
    savefig("response_curves_itransformer_duet")


if __name__ == "__main__":
    set_style()
    conceptual_diagram()
    sensitivity_plot()
    rir_workflow()
    response_curves()
