"""Mean failure rate across FMs per hint level (DSTS), styled like plot_failure_modes.py:
DejaVu Serif, bold titles and tick labels, black-edged legend on top, dashed light y-grid, outward
x ticks.

Data: dsts_results.py.load_judge_csv_by_level_rep -- same pass/fail/crash logic and
per-column polarity (INVERTED_CHECK_COLUMNS) as the no-hint figure; crashed repeats are excluded
from the mean, never counted as passes.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsts_results import AGENTS, OUT_DIR, MODELS, load_judge_csv_by_level_rep  # noqa: E402

# Local rebind, scoped to this figure only (doesn't touch
# dsts_results.py.MODELS, which the sunburst figure still reads in
# full) --, same as plot_failure_modes.py's own
# gemini-3.6-flash drop.
MODELS = [(mk, ml) for mk, ml in MODELS if ml != "gemini-3.6-flash"]

# Agent colors kept from the original hint-level plot (colorblind-safe blue/orange/green); the
# red/teal pair of the no-hint bars encodes MODEL there and is not reused for agents here.
# Keyed on "Vanilla" (not "ReAct") -- must match AGENTS (imported from
# dsts_results.py), which is load-bearing for the stats[(agent,
# model)] lookups below and stays "Vanilla" on disk (directory names etc.).
# A prior hand-edit renamed this dict's key to "ReAct" directly, which broke
# AGENT_COLORS[agent] with a KeyError the moment agent == "Vanilla" -- the
# display-only rename belongs in AGENT_DISPLAY_LABELS below instead, same
# split plot_failure_modes.py already uses for its own "vanilla" -> "ReAct"
# rename.
AGENT_COLORS = {"Vanilla": "#0072B2", "AIDE": "#E69F00", "MLEvolve": "#009E73"}
AGENT_DISPLAY_LABELS = {"Vanilla": "ReAct", "AIDE": "AIDE", "MLEvolve": "MLEvolve"}
WIDTH_IN = 3.0  # wide enough for the 3-item legend row to fit without clipping
HEIGHT_IN_PER_PANEL = 2.12
FIGSIZE = None  # set below, once MODELS_WITH_DATA is known (needs len(MODELS_WITH_DATA))
OUT_STEM = "failure_modes_by_hint"
DODGE = 0.0  # no x offset -- dots sit exactly on the integer hint level for every agent


def level_stats(agent: str, model_key: str) -> dict:
    """level -> dict(mean, sd, n_reps, n_fms, n_fms_design). Per repeat: mean over the FMs that
    have a real judgment at that (level, rep); then mean/SD across repeats."""
    per_fm = load_judge_csv_by_level_rep(agent, model_key)
    levels = sorted({lv for d in per_fm.values() for (lv, _) in d})
    out = {}
    for lv in levels:
        by_rep, fms, design = defaultdict(list), set(), set()
        for fm, d in per_fm.items():
            for (l, rep), c in d.items():
                if l != lv:
                    continue
                design.add(fm)
                if c["pass"] + c["fail"] == 0:
                    continue  # this repeat crashed entirely: no judgment, not a pass
                fms.add(fm)
                by_rep[rep].append(c["fail"] / (c["pass"] + c["fail"]))
        rep_means = [float(np.mean(v)) for v in by_rep.values()]
        if rep_means:
            out[lv] = {"mean": float(np.mean(rep_means)),
                       "sd": float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0,
                       "n_reps": len(rep_means), "n_fms": len(fms), "n_fms_design": len(design)}
    return out


stats = {(a, ml): level_stats(a, mk) for a in AGENTS for mk, ml in MODELS}
MODELS_WITH_DATA = [ml for _, ml in MODELS if any(stats[(a, ml)] for a in AGENTS)]
NOHINT_HEIGHT_IN = 3.8  # must match plot_failure_modes.py's FIGSIZE[1]
FIGSIZE = (WIDTH_IN, NOHINT_HEIGHT_IN)
LEVELS = sorted({lv for v in stats.values() for lv in v})
FMS_AT_LEVEL = {lv: max((v[lv]["n_fms_design"] for v in stats.values() if lv in v), default=0) for lv in LEVELS}
ANY_PARTIAL = any(v[lv]["n_fms"] < v[lv]["n_fms_design"] for v in stats.values() for lv in v)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": "#000000",
    "axes.linewidth": 0.8,
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "xtick.direction": "out",
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(len(MODELS_WITH_DATA), 1, figsize=FIGSIZE, sharex=True, sharey=True)
n_agents = len(AGENTS)
offsets = (np.arange(n_agents) - (n_agents - 1) / 2) * DODGE * 2

for row_idx, (ax, model) in enumerate(zip(np.atleast_1d(axes), MODELS_WITH_DATA)):
    for a_idx, agent in enumerate(AGENTS):
        d = stats[(agent, model)]
        if not d:
            continue
        lvs = sorted(d)
        x = np.array(lvs, float) + offsets[a_idx]
        mean = np.array([d[l]["mean"] for l in lvs])
        sd = np.array([d[l]["sd"] for l in lvs])
        color = AGENT_COLORS[agent]
        lo, hi = np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1)
        ax.plot(x, mean, color=color, linewidth=1.4, zorder=2, gid=f"line_{model}_{agent}")
        has_err = sd > 0
        if has_err.any():
            ax.errorbar(x[has_err], mean[has_err], yerr=np.vstack([mean - lo, hi - mean])[:, has_err],
                        fmt="none", ecolor=color, elinewidth=0.8, capsize=1.8, capthick=0.8,
                        zorder=3, gid=f"err_{model}_{agent}")
        partial = np.array([d[l]["n_fms"] < d[l]["n_fms_design"] for l in lvs])
        ax.scatter(x[~partial], mean[~partial], s=16, color=color, edgecolor="black", linewidth=0.7,
                   zorder=4, clip_on=False, gid=f"pts_{model}_{agent}")
        if partial.any():  # hollow: fewer FMs than the level's design contributed
            ax.scatter(x[partial], mean[partial], s=16, facecolor="white", edgecolor=color, linewidth=1.0,
                       zorder=4, clip_on=False, gid=f"partial_{model}_{agent}")

    # fontsize tuned (not the nominal 7.5 nohint uses) to compensate for
    # plot_failure_modes_combined.py's autocrop+shared-height rescaling: this
    # figure's cropped content is proportionally shorter than nohint's, so an
    # equal nominal fontsize renders visibly larger once both panels are
    # rescaled to the same height in the combined composite -- see that
    # script's docstring. Re-derive this value (see the measurement in this
    # session) if this figure's FIGSIZE/content layout changes materially.
    ax.set_title(model, fontsize=7.18, fontweight="bold", pad=4)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(lv) for lv in LEVELS], fontweight="bold")
    # Only the bottom row keeps its x-tick labels -- sharex=True already
    # syncs the range/ticks across rows, repeating them on every row would
    # just be redundant clutter in a vertical stack.
    if row_idx < len(MODELS_WITH_DATA) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(min(LEVELS) - 0.45, max(LEVELS) + 0.45)
    ax.set_ylim(-0.05, 1.05)  # a little room below 0 so markers at exactly 0 are not clipped by the axis
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.tick_right()  # y-axis (ticks + label) on the right side
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#E5E5E5", zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")

# fig.supylabel/supxlabel (not a per-axes set_ylabel/set_xlabel on whichever
# panel happens to be first or sit near the middle) -- one shared label,
# correctly centered across however many rows there are, in either layout.
# supylabel: x near 1.0 + rotation=-90 (not the default 90) puts it on the
# right edge, matching yaxis.tick_right() above -- text reads top-to-bottom
# on that side rather than upside down.
fig.supylabel("Fail rate (mean ± SD)", fontweight="bold", fontsize=8, x=0.9, rotation=-90, ha="right")

handles = [Line2D([0], [0], color=AGENT_COLORS[a], linewidth=1.4, marker="o", markersize=4,
                  markeredgecolor="black", markeredgewidth=0.7, label=AGENT_DISPLAY_LABELS[a]) for a in AGENTS]
if ANY_PARTIAL:
    handles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=4, markerfacecolor="white",
                          markeredgecolor="#555555", label="fewer FMs (crashed runs excluded)"))
leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.45, 0.96),
                 frameon=True, fancybox=False, edgecolor="#000000", prop=dict(weight="bold", size=8),
                 handlelength=2.2, columnspacing=0.5)
leg.set_gid("legend")

# Absolute-inch margins, converted to fractions of THIS figure's actual
# FIGSIZE -- a fraction tuned for one figure size leaves badly oversized (or
# clipped) margins once the figure's dimensions change, since FIGSIZE now
# scales with model/row count. RIGHT (not LEFT) margin now reserved, since
# the y-axis (ticks + label) moved to the right side above.
RIGHT_MARGIN_IN, BOTTOM_MARGIN_IN, TOP_MARGIN_IN = 0.4, 0.06, 0.38
fig.tight_layout(
    rect=(0, BOTTOM_MARGIN_IN / FIGSIZE[1], 1 - RIGHT_MARGIN_IN / FIGSIZE[0], 1 - TOP_MARGIN_IN / FIGSIZE[1]),
    h_pad=1.5,
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bottom_ax = axes[-1]
axes_bottom_frac = bottom_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
CAPTION_GAP_IN = 0.12  # small breathing room between the tick labels and this caption
fig.supxlabel("Hint level", fontweight="bold", fontsize=8,
              y=axes_bottom_frac - CAPTION_GAP_IN / FIGSIZE[1], rotation=0, ha="center")

print("FMs contributing per hint level:", FMS_AT_LEVEL)
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    p = OUT_DIR / f"{OUT_STEM}.{ext}"
    fig.savefig(p, dpi=300)
    print(f"Wrote {p}")
