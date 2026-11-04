#!/usr/bin/env python3
"""Added-variable-style plot for the model-adjusted regression in perf_effects.py: instead of
asserting the adjusted "present" coefficient as a number, show it.

Usage:
    python scripts/visualization/plot_perf_vs_judge_effects_residualized.py
    -> analysis/figures/perf_vs_judge_effects_residualized.{pdf,png}
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from judge_viz_common import FAILURE_MODES, FM_LABEL  # noqa: E402
from perf_effects import (  # noqa: E402
    MODEL_ORDER, MODEL_LABEL, MODEL_COLORS, LEAKAGE_MODES,
    load_rows, load_judged, compute_ranks,
)

C_LEAK = "#CD0324"
C_OTHER = "#3468DC"
MIN_N = 15

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


class _Args:
    def __init__(self, csv, source):
        self.csv = csv
        self.source = source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "results" / "predictive_scores.csv"))
    ap.add_argument("--source")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "analysis" / "figures" /
                                          "perf_vs_judge_effects_residualized"))
    args = ap.parse_args()

    rows = load_rows(_Args(args.csv, args.source))
    judged = load_judged()
    ranks = compute_ranks(rows, min_configs=3)
    usable = {k: v for k, v in ranks.items() if k in judged}

    fm_index = {fid: i for i, (fid, _, _) in enumerate(FAILURE_MODES)}

    panels = []
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

        # cluster-robust (by task) for the p-value/significance call, matching
        # perf_effects.py's convention -- the
        # point estimate is identical either way (cov_type only affects SEs).
        full = smf.ols("rank ~ present + C(model) + C(agent)", df).fit()
        full_cl = smf.ols("rank ~ present + C(model) + C(agent)", df).fit(
            cov_type="cluster", cov_kwds={"groups": df["task"]})
        b1 = full.params["present"]
        p = full_cl.pvalues["present"]
        df = df.assign(resid=b1 * df["present"] + full.resid)
        delta = df.loc[df.present == 1, "resid"].mean() - df.loc[df.present == 0, "resid"].mean()
        assert abs(delta - b1) < 1e-8, f"{fid}: partial-residual delta {delta} != coefficient {b1}"
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        panels.append({"fid": fid, "df": df, "delta": delta, "n_present": n_present,
                       "p": p, "stars": stars, "leak": fid in LEAKAGE_MODES})

    panels = sorted([p for p in panels if p["leak"]], key=lambda p: p["delta"]) + \
             sorted([p for p in panels if not p["leak"]], key=lambda p: p["delta"])

    n = len(panels)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(0)

    for ax, panel in zip(axes, panels):
        df, fid = panel["df"], panel["fid"]
        c = C_LEAK if panel["leak"] else C_OTHER
        sig = panel["p"] < 0.05
        for present_val, xc in ((0, 0.0), (1, 1.0)):
            sub = df[df.present == present_val]
            jitter = rng.uniform(-0.13, 0.13, size=len(sub))
            colors = [MODEL_COLORS[m] for m in sub["model"]]
            ax.scatter(xc + jitter, sub["resid"], s=10, c=colors, edgecolor="black",
                      linewidth=0.2, alpha=0.6, zorder=2)
            ax.plot([xc - 0.22, xc + 0.22], [sub["resid"].mean()] * 2, color="black",
                    linewidth=2.0, zorder=4, solid_capstyle="butt")
        ax.plot([0, 1], [df.loc[df.present == 0, "resid"].mean(),
                        df.loc[df.present == 1, "resid"].mean()],
                color=c, linewidth=1.4, linestyle="--", zorder=3)
        ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=0)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["absent", "present"], fontsize=7.5)
        title_suffix = f"  {panel['stars']}" if sig else ""
        ax.set_title(f"{fid}  {FM_LABEL[fid]}{title_suffix}", fontsize=8,
                    fontweight=600, pad=3)
        ax.text(0.5, 0.98, f"$\\Delta$={panel['delta']:+.2f}  $p$={panel['p']:.2g}  (n={panel['n_present']})",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="#cccccc", linewidth=0.5, boxstyle="round,pad=0.2"))
        ax.set_ylabel("rank resid.", fontsize=7.5)
        ax.tick_params(labelsize=7)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("black")

    for ax in axes[n:]:
        ax.axis("off")

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                          markerfacecolor=MODEL_COLORS[m], markeredgecolor="black",
                          markeredgewidth=0.3, label=MODEL_LABEL[m]) for m in MODEL_ORDER]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
                     ncol=3, frameon=True, prop={"weight": 600, "size": 10},
                     edgecolor="black", facecolor="white", framealpha=1)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = f"{args.out}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.05)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
