"""Failure-mode distribution: what % of all detected failure instances is each taxonomy mode
responsible for, pooled across every agent x model config.

Usage:
    python scripts/visualization/plot_failure_mode_distribution.py
    -> writes analysis/figures/failure_mode_distribution.{pdf,svg,png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_viz_common import (  # noqa: E402
    PROJECT_ROOT, AGENTS, MODELS, FAILURE_MODES, FM_LABEL,
    FM_CATEGORY, CATEGORY_COLOR, load_agent_model,
)

# Local drop (gemini-3.6-flash out), matching the other judge figures; doesn't touch
# judge_viz_common.MODELS.
MODELS = [m for m in MODELS if m["key"] != "gemini-3.6-flash"]

OUT_DIR = PROJECT_ROOT / "analysis" / "figures"
OUT_STEM = "failure_mode_distribution"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "font.size": 15,
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "xtick.labelsize": 14,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def main():
    counts = [0] * len(FAILURE_MODES)
    n_judged = 0
    for agent in AGENTS:
        for m in MODELS:
            tasks = load_agent_model(agent, m["key"])
            for t in tasks.values():
                if t["no_code"]:
                    continue
                n_judged += 1
                for i, cell in enumerate(t["cells"]):
                    if cell["s"] == "present":
                        counts[i] += 1

    total = sum(counts)
    if total == 0:
        raise SystemExit("No 'present' findings found across judge_results -- nothing to plot.")

    pct = [c / total * 100 for c in counts]
    fids = [fid for fid, _, _ in FAILURE_MODES]

    # sort descending by percentage; matplotlib barh plots first row at the
    # bottom, so reverse for a natural top-to-bottom (highest % first) read.
    order = sorted(range(len(FAILURE_MODES)), key=lambda i: pct[i], reverse=True)[::-1]
    disp_ids = [fids[i] for i in order]
    labels = [f"{disp_ids[k]}  {FM_LABEL[fids[i]]}" for k, i in enumerate(order)]
    values = [pct[i] for i in order]
    colors = [CATEGORY_COLOR[FM_CATEGORY[fids[i]]] for i in order]
    ns = [counts[i] for i in order]

    n_rows = len(labels)
    y = list(range(n_rows))
    fig_h = 0.5 * n_rows + 1.8
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    max_val = max(values) or 1.0
    ax.set_xlim(0, max_val * 1.4)

    bar_height = 0.68
    ax.barh(y, values, height=bar_height, color=colors, edgecolor="black", linewidth=1.3, zorder=2)

    for yi, v, n in zip(y, values, ns):
        ax.text(v + max_val * 0.016, yi, f"{v:.1f}%", va="center", ha="left",
                fontsize=14, color="#111111", zorder=3)
        ax.text(max_val * 1.38, yi, f"n={n}", va="center", ha="right",
                fontsize=11.5, color="#555555", fontfamily="monospace", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xlabel(f"% of all detected failure instances (n={total})",
                  fontsize=15, weight="medium")

    # Category legend (leak vs other), using CATEGORY_COLOR.
    handles = [plt.matplotlib.patches.Patch(facecolor=CATEGORY_COLOR[c], edgecolor="black",
               linewidth=1.3, label=c) for c in ("leak", "other")]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.01),
                     frameon=True, ncol=2, handlelength=1.2, handleheight=1.2,
                     borderaxespad=0.3, fontsize=13, prop={"weight": "semibold"})
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.1)

    # Full black box around the axes, matching judge_by_agent_fm_x.png.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=4, width=1.1)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT_DIR / OUT_STEM}.{{pdf,svg,png}}  (n={total} findings across {n_judged} judged runs)")


if __name__ == "__main__":
    main()
