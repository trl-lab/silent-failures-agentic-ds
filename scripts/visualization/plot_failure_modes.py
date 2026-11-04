"""
2x3 grouped bar plot: no-hint fail rate (mean +/- SD, n=5) per failure mode,
grouped by agent scaffold, one bar per LLM.

Grouped-by-agent bars per failure-mode panel with value labels above the bars. Reads data from the test suite's judge CSVs via dsts_results.py.

Saved WITHOUT bbox_inches="tight" (same reasoning as v2): tight cropping
trims the canvas unevenly and changes the on-disk 2:1 ratio.

Outputs (to analysis/figures/, matching every other build_*.py script):
  failure_modes_nohint.pdf  (vector, TrueType fonts -> editable text)
  failure_modes_nohint.svg  (text kept as <text>, elements grouped by gid -> Figma-friendly)
  failure_modes_nohint.png  (preview)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsts_results import (  # noqa: E402
    AGENTS, FMS, MODELS_GRID, OUT_DIR,
    build_rep_fail_rate_rows, crashed_cells,
)

_KEEP_MODELS = {"gemini-3.7-flash", "gpt-5.6-sol"}
MODELS_GRID = [(k, lbl) for k, lbl in MODELS_GRID if lbl in _KEEP_MODELS]
MODELS_GRID = sorted(MODELS_GRID, key=lambda km: 0 if km[1] == "gemini-3.7-flash" else 1)

# ----------------------------------------------------------------------------
# Data -- live from judge_results/, not a hardcoded snapshot. Same source
# (load_judge_csv_by_rep, no-hint variant only, REPEATS_ONLY) as
# dsts_results.py's own grid, so the two stay consistent.
# ----------------------------------------------------------------------------
_df = build_rep_fail_rate_rows(AGENTS, MODELS_GRID)
if _df.empty:
    sys.exit("No per-repeat judge CSV data found for any (agent, model) in MODELS_GRID.")

CRASHED_CELLS = crashed_cells(AGENTS, MODELS_GRID)

MODEL_LABELS = [lbl for _, lbl in MODELS_GRID]

# Display-only rename (matches judge_viz_common.AGENT_DISPLAY's
# earlier "vanilla" -> "ReAct" rename for the same reason): AGENTS itself
# stays "Vanilla" (load-bearing -- AGENT_DIR_NAME and every judge_results/
# directory name key off that exact string), only the rendered x-tick label
# changes.
AGENT_DISPLAY_LABELS = {"Vanilla": "ReAct", "AIDE": "AIDE", "MLEvolve": "MLEvolve"}

# Every DSTS cell is run with 5 independent repeats (REPEATS=5 in
# dsts_run_all.sh / the Snellius sweep). A cell's effective n drops below 5
# when 1+ (but not all 5) repeats crashed -- build_rep_fail_rate_rows already
# excludes those from the mean/SD, but silently, so a cell built from 3 real
# repeats looks identical to one built from 5. Surfacing n on the label
# keeps that visible.
FULL_N = 5

_stats = _df.groupby(["fm", "model", "agent"])["fail_rate"].agg(["mean", "std", "count"]).reset_index()
_stats["std"] = _stats["std"].fillna(0.0)


def _series(fm: str, model: str):
    """(means per agent, sds per agent, n per agent), 0.0/0 for any agent
    with no data."""
    sub = _stats[(_stats["fm"] == fm) & (_stats["model"] == model)].set_index("agent")
    means = [sub.loc[a, "mean"] if a in sub.index else 0.0 for a in AGENTS]
    sds = [sub.loc[a, "std"] if a in sub.index else 0.0 for a in AGENTS]
    ns = [int(sub.loc[a, "count"]) if a in sub.index else 0 for a in AGENTS]
    return means, sds, ns


# Only plot -- and only put in the legend -- models that actually have judge
# data for at least one (fm, agent) cell. Unlike the cnsplots grid
# (dsts_fm_grid.png), qwen-3.8-27b is NOT kept as a reserved empty legend
# slot here: it has no judged results yet, so listing it would caption a
# model with nothing behind it.
MODELS_WITH_DATA = [m for m in MODEL_LABELS if m in set(_stats["model"])]

# Colors for the models shown in this figure (red/teal palette).
ACTIVE_COLORS = {
    "gemini-3.6-flash": "#C8323C",  # red -- unused while 3.6 is filtered out above, kept for a quick revert
    "gemini-3.7-flash": "#C8323C",  # red
    "gpt-5.6-sol": "#5BB5A4",       # teal
}
MODEL_COLORS = {m: ACTIVE_COLORS[m] for m in MODELS_WITH_DATA}

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
WIDTH_IN = 7.0                      # ~full text width of a two-column paper
FIGSIZE = (WIDTH_IN, WIDTH_IN / 2)  # 2:1 width:height
BAR_WIDTH = 0.34
CLIP_ERRORBARS = True         # fail rate is bounded in [0, 1]; clip whiskers there
SHOW_VALUES = True
OUT_STEM = "failure_modes_nohint"

SHORT_FM_TITLES = {
    "TF01_goal_imbalance": "TF01 Ignoring class imbalance",
    "TF02_player_rep_norm": "TF02 Not normalizing messy data",
    "TF03_correlation_causation": "TF03 Correlation vs. causation",
    "TF04_balance_torque": "TF04 Missing domain feature eng.",
    "TF05_credit_card_leakage": "TF05 Data leakage in feat. sel.",
    "TF06_metric_select": "TF06 Wrong metric selection",
}

# ----------------------------------------------------------------------------
# Style -- palette matches the rest of the paper's figures (see module
# docstring), but the serif font is back to this script's original choice
# (preferred over the sans-serif swap made to match dsts_results.py
# / judge_by_agent_fm_x.png -- scoped to this chart only, those two keep
# Helvetica).
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": "#000000",
    "axes.linewidth": 0.8,
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "xtick.direction": "out",
    "xtick.major.size": 2.5,      # visible tick under each agent label
    "xtick.major.width": 0.8,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",    # keep text as text in SVG
})

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharey=True)
x = np.arange(len(AGENTS))
n_models = len(MODELS_WITH_DATA)
offsets = (np.arange(n_models) - (n_models - 1) / 2) * BAR_WIDTH

for p_idx, (ax, fm) in enumerate(zip(axes.flat, FMS)):
    for m_idx, model in enumerate(MODELS_WITH_DATA):
        mean, sd, n = _series(fm, model)
        mean, sd, n = np.asarray(mean), np.asarray(sd), np.asarray(n)
        xpos = x + offsets[m_idx]
        crashed = np.array([(fm, agent, model) in CRASHED_CELLS for agent in AGENTS])

        if (~crashed).any():
            ax.bar(
                xpos[~crashed], mean[~crashed], BAR_WIDTH,
                color=MODEL_COLORS[model], edgecolor="black", linewidth=0.8, zorder=2,
                gid=f"bars_{p_idx}_{model}",
            )

        lo = np.clip(mean - sd, 0, 1) if CLIP_ERRORBARS else mean - sd
        hi = np.clip(mean + sd, 0, 1) if CLIP_ERRORBARS else mean + sd
        yerr = np.vstack([mean - lo, hi - mean])
        has_err = (sd > 0) & (~crashed)
        if has_err.any():
            ax.errorbar(
                xpos[has_err], mean[has_err], yerr=yerr[:, has_err],
                fmt="none", ecolor="#000000", elinewidth=0.7,
                capsize=1.8, capthick=0.7, zorder=3,
                gid=f"err_{p_idx}_{model}",
            )

        if SHOW_VALUES:
            for xi, top, m, ni in zip(xpos[~crashed], hi[~crashed], mean[~crashed], n[~crashed]):
                y0 = max(top, m) + 0.02
                ax.text(
                    xi, y0, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=5.5,
                    fontweight="normal", color="black", zorder=4,
                    gid=f"val_{p_idx}_{model}",
                )
                if ni < FULL_N:
                    # Stacked above the value in a smaller/greyer line rather
                    # than appended inline -- an inline "(n=X)" widened the
                    # label enough to collide with the neighboring bar's
                    # label whenever both were close together (confirmed:
                    # MLEvolve pairs in TF01/03/06 overlapped into an
                    # unreadable jumble).
                    ax.text(
                        xi, y0 + 0.13, f"n={ni}",
                        ha="center", va="bottom", fontsize=5,
                        color="#666666", zorder=4,
                        gid=f"n_{p_idx}_{model}",
                    )

        for xi in xpos[crashed]:
            ax.bar(
                xi, 1.0, BAR_WIDTH, color="#EEEEEE", edgecolor="#999999",
                linewidth=0.8, hatch="////", zorder=2,
                gid=f"crashed_{p_idx}_{model}",
            )

    ax.set_title(SHORT_FM_TITLES[fm], fontsize=7.5, fontweight="bold", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY_LABELS[a] for a in AGENTS], fontweight="bold")
    ax.set_xlim(-0.6, len(AGENTS) - 0.4)
    ax.set_ylim(0, 1.36)  # headroom for value + n=X labels stacked above 1.0
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#E5E5E5", zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")

for ax in axes[:, 0]:
    ax.set_ylabel("Fail rate (mean ± SD)", fontweight="bold")

# Legend lists the models with real data, plus (if any cell needed it) a
# hatched "invalid runs" swatch explaining those grey bars right where the
# reader already looks for a color key -- not a separate caption stapled to
# the bottom of the figure.
handles = [
    Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m)
    for m in MODELS_WITH_DATA
]
_any_crashed = bool(CRASHED_CELLS)
if _any_crashed:
    handles.append(
        Patch(facecolor="#EEEEEE", edgecolor="#999999", hatch="////", label="invalid runs")
    )
leg = fig.legend(
    handles=handles, loc="upper center", ncol=len(handles),
    bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=False,
    edgecolor="#000000", prop=dict(weight="bold", size=8),
    handlelength=2.2, columnspacing=2.0,
)
leg.set_gid("legend")

fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.6)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    p = OUT_DIR / f"{OUT_STEM}.{ext}"
    fig.savefig(p, dpi=300)  # no bbox_inches="tight" -- keeps the 2:1 ratio
    print(f"Wrote {p}")
