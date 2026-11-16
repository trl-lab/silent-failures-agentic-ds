"""
Data loading and rank statistics for the performance-vs-failure-mode analysis: joins judge
verdicts to predictive scores (results/predictive_scores.csv) and computes within-task
residual ranks. Used by plot_perf_vs_judge_effects_slopegraph.py and
plot_perf_vs_judge_effects_residualized.py.
"""
import csv
import json
from collections import defaultdict
from judge_viz_common import result_dir, cell_state, FM_IDS, MODELS, AGENT_DISPLAY  # noqa: E402

RUN2CFG = {
    "vanilla__gemini-gemini-3.7-flash__few_shot_role": ("vanilla", "gemini-3.7-flash"),
    "aide__gemini-gemini-3.7-flash__10steps":          ("aide", "gemini-3.7-flash"),
    "mlevolve__gemini-gemini-3.7-flash__10steps":      ("mlevolve", "gemini-3.7-flash"),
    "vanilla__vllm-qwen3.5-4b__zero_shot":             ("vanilla", "qwen3.5-4b"),
    "aide__vllm-qwen3.5-4b__10steps":                  ("aide", "qwen3.5-4b"),
    "mlevolve__vllm-qwen3.5-4b__10steps":              ("mlevolve", "qwen3.5-4b"),
    "vanilla__openrouter-qwen-3.8-27b__few_shot_role": ("vanilla", "qwen3.8-27b"),
    "aide__openrouter-qwen-3.8-27b__10steps":          ("aide", "qwen3.8-27b"),
    "mlevolve__openrouter-qwen-3.8-27b__10steps":      ("mlevolve", "qwen3.8-27b"),
}


LEAKAGE_MODES = {"FM02", "FM08", "FM09", "FM04", "FM05", "FM13"}


AGENTS = ["vanilla", "aide", "mlevolve"]


FULL_MODEL_COLOR_MAP = {
    "qwen3.5-4b": "#3468DC", "qwen3.8-27b": "#FFA631",
    "gemini-3.6-flash": "#008019", "gemini-3.7-flash": "#B0173F",
}


_ACTIVE_MODEL_KEYS = {a_m[1] for a_m in RUN2CFG.values()}


MODEL_ORDER = sorted(
    (m["key"] for m in MODELS if m["key"] in _ACTIVE_MODEL_KEYS),
    key=lambda k: 0 if "qwen" in k.lower() else 1,
)


MODEL_LABEL = {m["key"]: m["label"] for m in MODELS if m["key"] in _ACTIVE_MODEL_KEYS}


MODEL_COLORS = {k: FULL_MODEL_COLOR_MAP[k] for k in MODEL_ORDER}


def load_rows(args):
    """(a, m, task) -> ("ok", score, higher_is_better) | ("failed", None, None).
    "failed" = the pipeline never produced a scorable result (predictive_scores.csv
    status == "failed", e.g. crashed, timed out, no submission) -- kept as a
    data point (rank_task() puts it last), not dropped, so a config that fails
    outright isn't invisible to the leak/other-vs-rank relationship."""
    rows = {}
    for r in csv.DictReader(open(args.csv)):
        if r["run_dir"] not in RUN2CFG:
            continue
        if args.source and r["source"] != args.source:
            continue
        a, m = RUN2CFG[r["run_dir"]]
        k = (a, m, r["task_id"])
        if r["status"] == "failed" or r["score"] == "":
            rows[k] = ("failed", None, None)
            continue
        v, hib = float(r["score"]), r["higher_is_better"] == "True"
        if hib and v == 0.0:
            continue  # degenerate label-encoding artifact --
                      # a scoring bug, not a genuine pipeline failure, so it's dropped rather
                      # than counted as either an "ok" score or a "failed" pipeline
        rows[k] = ("ok", v, hib)
    return rows


def is_no_code(cells):
    """True if the judge had NOTHING to go on for this pipeline -- every applicable (non-"na") check came
    back "cannot_determine", typically because the agent crashed before writing any code worth
    inspecting (see judge_viz_common.load_agent_model's own "no_code" flag, same definition).
    leak_other_counts() would score such a cell as "0 leakage modes, 0 other modes" since
    cannot_determine contributes to neither -- indistinguishable from a genuinely clean pipeline
    unless this is checked separately and the cell is dropped before plotting/correlating.
    """
    applicable = [c for c in cells if c["s"] != "na"]
    return len(applicable) > 0 and all(c["s"] == "cannot_determine" for c in applicable)


def load_judged():
    """(a, m, task) -> per-FM cell states, EXCLUDING no-code pipelines (see is_no_code()) -- matches
    judge_viz_common.compute_taxonomy_column's own n_judged (which already excludes them from its
    denominator), so "judged" means the same thing here as it does in the project's other judge
    figures.
    """
    judged = {}
    for cfg in set(RUN2CFG.values()):
        d = result_dir(*cfg)
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            fm = data.get("failure_modes") or {}
            if not fm:
                continue
            cells = [cell_state(fm.get(f)) for f in FM_IDS]
            if is_no_code(cells):
                continue
            judged[(cfg[0], cfg[1], p.stem)] = cells
    return judged


def compute_ranks(rows, min_configs):
    """(a, m, task) -> rank among the configs seen on that task. 1 = best; ties among "ok" pipelines get
    the average of the tied ranks (fractional ranking, same convention scipy.stats.rankdata uses).
    """
    bytask = defaultdict(list)
    for k, r in rows.items():
        bytask[k[2]].append((k, r))
    ranks = {}
    for t, lst in bytask.items():
        n = len(lst)
        if n < min_configs:
            continue
        ok = [(k, r[1], r[2]) for k, r in lst if r[0] == "ok"]
        failed = [k for k, r in lst if r[0] == "failed"]
        if ok:
            hib = ok[0][2]
            vals = [v for _, v, _ in ok]
            for k, v, _ in ok:
                better = sum(1 for x in vals if (x > v if hib else x < v))
                tied = sum(1 for x in vals if x == v)
                ranks[k] = better + (tied + 1) / 2
        for k in failed:
            ranks[k] = float(n)
    return ranks
