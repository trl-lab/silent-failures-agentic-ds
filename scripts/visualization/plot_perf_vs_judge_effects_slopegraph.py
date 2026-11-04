#!/usr/bin/env python3
"""Summary slopegraph for the model-adjusted regression: one shared, zoomed-in plot instead of
plot_perf_vs_judge_effects_residualized.py's 12 noisy point-cloud panels. Two parallel axes --
"absent" on the left, "present" on the right -- one line per failure mode connecting its two group
means on the rank-residual scale.

Usage:
    python scripts/visualization/plot_perf_vs_judge_effects_slopegraph.py
    -> analysis/figures/perf_vs_judge_effects_slopegraph.{pdf,png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from judge_viz_common import FAILURE_MODES  # noqa: E402
from perf_effects import (  # noqa: E402
    LEAKAGE_MODES, load_rows, load_judged, compute_ranks,
)

C_LEAK = "#CD0324"
C_OTHER = "#3468DC"
MIN_N = 15
OUT = PROJECT_ROOT / "analysis" / "figures" / "perf_vs_judge_effects_slopegraph"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


class _Args:
    def __init__(self, csv, source):
        self.csv = csv
        self.source = source


# Manual label y-position override, keyed by failure-mode id (e.g. "FM08") --
# hand-edit these to move a specific label's TEXT up/down without moving its
# dot/line (which always stay at the real fitted value). Value is an offset
# ADDED to the true delta, not an absolute position, so "0.0" always means
# "label sits exactly at its own dot" -- only change the ones that need
# nudging apart. A fid not listed here (e.g. if the underlying data changes
# and a new mode clears the MIN_N threshold) defaults to 0.0 via .get() below,
# so this dict never needs to be exhaustive to avoid crashing.
LABEL_Y_OFFSET = {
    "FM06": 0.0,
    "FM01": 0.0,
    "FM12": 0.0,
    "FM03": 0.02,
    "FM07": 0.0,
    "FM14": 0.02,
    "FM09": 0.0,
    "FM15": 0.08,
    "FM05": 0.0,
    "FM02": -0.05,
    "FM04": -0.1,
    "FM08": 0.0,
}


def main():
    rows = load_rows(_Args(str(PROJECT_ROOT / "results" / "predictive_scores.csv"), None))
    judged = load_judged()
    ranks = compute_ranks(rows, min_configs=3)
    usable = {k: v for k, v in ranks.items() if k in judged}
    fm_index = {fid: i for i, (fid, _, _) in enumerate(FAILURE_MODES)}

    rows_out = []
    for fid, _, _ in FAILURE_MODES:
        i = fm_index[fid]
        recs = []
        for k in usable:
            agent, model, task = k
            s = judged[k][i]["s"]
            if s not in ("present", "absent"):
                continue
            recs.append({"rank": usable[k], "present": 1 if s == "present" else 0,
                         "model": model, "agent": agent, "task": task})
        df = pd.DataFrame.from_records(recs)
        n_present = int(df["present"].sum())
        n_absent = len(df) - n_present
        if n_present < MIN_N or n_absent < MIN_N:
            continue
        m = smf.ols("rank ~ present + C(model) + C(agent)", df).fit(
            cov_type="cluster", cov_kwds={"groups": df["task"]})
        b1, p = m.params["present"], m.pvalues["present"]
        lo, hi = m.conf_int().loc["present"]
        rows_out.append({"fid": fid, "delta": b1, "lo": lo, "hi": hi, "p": p,
                         "n": n_present, "leak": fid in LEAKAGE_MODES})

    rows_out.sort(key=lambda r: r["delta"])
    deltas = [r["delta"] for r in rows_out]

    # PRESENT_X widened from the original 1.0 -- longer diagonal lines are
    # easier to tell apart by slope than a short 0-to-1 run. Figure height is
    # generous and fixed (not tied to label spread) so the real data range
    # (axis_lo/hi) gets stretched over more vertical pixels -- the actual
    # zoom-in the lines needed, rather than just more room for label text.
    PRESENT_X = 2.2
    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    label_ys = [delta + LABEL_Y_OFFSET.get(r["fid"], 0.0)
                for r, delta in zip(rows_out, deltas)]

    axis_lo, axis_hi = min(deltas) - 0.15, max(deltas) + 0.15

    # Light background shading: upper half (positive residual = worse rank,
    # since rank 1 is best) vs lower half (negative = better rank). Spans
    # only x=0 (absent axis) to x=PRESENT_X (present axis) -- fill_between
    # takes explicit data-coordinate x bounds, unlike axhspan's default
    # axes-fraction xmin/xmax, which bled into the label margin on the right
    # and past the y-axis on the left.
    ax.fill_between([0, PRESENT_X], 0, axis_hi, color="#FCE4D6", zorder=-2)  # Orange (Bad/Underperformed)
    ax.fill_between([0, PRESENT_X], axis_lo, 0, color="#DEEBF7", zorder=-2)  # Blue (Good/Overperformed)
    # Worse performance (Dark Rust Orange text over Light Orange background)
    ax.text(PRESENT_X / 2, axis_hi - 0.03 * (axis_hi - axis_lo), "Worse performance (rank)",
            ha="center", va="top", fontsize=26, style="oblique", color="#A63603", zorder=0)

    # Better performance (Deep Navy Blue text over Light Blue background)
    ax.text(PRESENT_X / 2, axis_lo + 0.03 * (axis_hi - axis_lo), "Better performance (rank)",
            ha="center", va="bottom", fontsize=26, style="oblique", color="#08519C", zorder=0)

    for r, delta, ly in zip(rows_out, deltas, label_ys):
        sig = r["p"] < 0.05
        c = C_LEAK if r["leak"] else C_OTHER
        stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""

        ax.plot([0, PRESENT_X], [0, delta], color=c, linewidth=2.6, linestyle="-", zorder=2)
        ax.plot([PRESENT_X], [delta], marker="o", markersize=7, color=c,
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        # If LABEL_Y_OFFSET nudged this one, draw a thin leader line from the
        # dot to the (now offset) label so it's still traceable.
        if abs(ly - delta) > 1e-9:
            ax.plot([PRESENT_X, PRESENT_X + 0.06], [delta, ly], color="#999999",
                    linewidth=0.8, linestyle=":", zorder=1)

        # Id + stars only, placed right next to the dot (or its manual offset),
        # colored to match its line (leak/other) instead of plain black.
        label = f"{r['fid']} {stars}".strip()
        ax.text(PRESENT_X + 0.08, ly, label, va="center", ha="left", fontsize=20,
                fontweight=600, color=c, clip_on=False)

    ax.plot([PRESENT_X, PRESENT_X], [axis_lo, axis_hi], color="black", linewidth=1.6, zorder=1)
    # Baseline fit exactly from the "absent" axis to the "present" axis, not
    # spanning the full figure width (which bled into the label margin).
    ax.plot([0, PRESENT_X], [0, 0], color="#999999", linewidth=1.4, linestyle="-", zorder=1)

    ax.set_xlim(-0.35, PRESENT_X + 0.7)
    # ylim tight to the real data range (not the wider label spread) -- this
    # is what actually stretches the data's own vertical scale; labels that
    # spill slightly beyond it still draw via clip_on=False above.
    ax.set_ylim(axis_lo, axis_hi)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.set_xticks([0, PRESENT_X])
    ax.set_xticklabels(["absent", "present"], fontsize=22, fontweight=600)
    ax.set_ylabel("rank residual", fontsize=30)
    ax.yaxis.set_ticks_position("left")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    # Put the real y-axis (ticks + scale) exactly at x=0, where the "absent"
    # line already sits, instead of matplotlib's default left-edge-of-axes
    # position -- that default left a visually odd gap between the scale and
    # the "absent" line it's supposed to belong to.
    ax.spines["left"].set_position(("data", 0))
    ax.spines["left"].set_bounds(axis_lo, axis_hi)
    ax.spines["left"].set_linewidth(1.6)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=20, width=1.4, length=6)

    # ax.set_title("Adjusted effect on rank residual per failure mode: absent → present", fontsize=19,
    #              fontweight=700, pad=16)
    # fig.text(0.5, 0.965, "stars = cluster-robust (by task) significance: "
    #          "*p<0.05, **p<0.01, ***p<0.001",
    #          ha="center", va="top", fontsize=11, style="italic", color="#555555", transform=fig.transFigure)

    handles = [plt.matplotlib.patches.Patch(facecolor=C_LEAK, edgecolor="black", linewidth=0.8, label="Leakage mode"),
               plt.matplotlib.patches.Patch(facecolor=C_OTHER, edgecolor="black", linewidth=0.8, label="Non-leakage mode")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
             frameon=False, prop={"weight": 600, "size": 20})

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    for ext in ("png", "pdf"):
        p = f"{OUT}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.05)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
