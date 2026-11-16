"""Per-config breakdown, plotted: share of judged pipelines with AT LEAST ONE failure mode present (a
per-pipeline any-failure rate), one group of bars per agent, one bar per model -- same underlying
judge_results data and n_judged denominator as plot_findings_per_pipeline.py's average-findings
metric, but answering a different question: not "how many findings does a typical pipeline have"
but "what share of pipelines have zero clean bill of health".

Usage:
    python scripts/visualization/plot_pct_pipelines_with_failure.py
    -> analysis/figures/pct_pipelines_with_failure.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_viz_common import PROJECT_ROOT, AGENTS, MODELS, AGENT_DISPLAY, load_agent_model  # noqa: E402

# Local reorder/drop (qwen first, gemini-3.6-flash out), same rebind pattern
# as plot_findings_per_pipeline.py -- doesn't touch judge_viz_common.MODELS.
MODELS = [m for m in MODELS if m["key"] != "gemini-3.6-flash"]
MODELS = sorted(MODELS, key=lambda m: 0 if "qwen" in m["key"].lower() else 1)

OUT_DIR = PROJECT_ROOT / "analysis" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "bold",
    "figure.titleweight": "bold",
    "font.size": 20,
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "xtick.labelsize": 22,
    "ytick.labelsize": 18,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Same fixed key->hex map as plot_findings_per_pipeline.py, so a model's
# color matches every other figure in this family.
BAR_COLORS = {
    "qwen3.5-4b": "#3468DC", "qwen3.8-27b": "#FFA631",
    "gemini-3.6-flash": "#008019", "gemini-3.7-flash": "#B0173F",
}


def config_stats(agent: str, model: str) -> dict:
    tasks = load_agent_model(agent, model)
    n_judged = 0
    n_with_failure = 0
    for t in tasks.values():
        if t["no_code"]:
            continue
        n_judged += 1
        if any(c["s"] == "present" for c in t["cells"]):
            n_with_failure += 1
    return {"n_judged": n_judged, "n_with_failure": n_with_failure,
            "pct": 100.0 * n_with_failure / n_judged if n_judged else 0.0}


def main() -> None:
    data = {(a, m["key"]): config_stats(a, m["key"]) for a in AGENTS for m in MODELS}

    n_agents, n_models = len(AGENTS), len(MODELS)
    bar_width = 0.85 / n_models
    x = list(range(n_agents))

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for mi, m in enumerate(MODELS):
        offset = (mi - (n_models - 1) / 2) * bar_width
        vals = [data[(a, m["key"])]["pct"] for a in AGENTS]
        ns = [data[(a, m["key"])]["n_judged"] for a in AGENTS]
        xs = [xi + offset for xi in x]
        ax.bar(xs, vals, width=bar_width, color=BAR_COLORS[m["key"]],
               edgecolor="black", linewidth=1.6, zorder=2,
               label=m["label"])
        for xi, v, n in zip(xs, vals, ns):
            ax.text(xi, v + 0.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=18, zorder=3)
            ax.text(xi, v + 7.0, f"n={n}", ha="center", va="bottom", fontsize=14,
                    color="#555555", fontfamily="monospace", zorder=3)

    for xi in range(n_agents + 1):
        ax.axvline(xi - 0.5, color="#8c8c8c", linewidth=1.2, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_DISPLAY[a] for a in AGENTS], fontsize=24, weight="bold")
    ax.set_xlim(-0.5, n_agents - 0.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Pipelines with ≥ 1 failure mode (%)", fontsize=20, weight="medium")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.3)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=4, width=1.1)

    handles = [mpatches.Patch(facecolor=BAR_COLORS[m["key"]], edgecolor="black",
                              linewidth=1.4, label=m["label"]) for m in MODELS]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.48, -0.1),
                     frameon=True, ncol=n_models, handlelength=1.2, handleheight=1.2,
                     borderaxespad=0.3, fontsize=20)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.1)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"pct_pipelines_with_failure.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
