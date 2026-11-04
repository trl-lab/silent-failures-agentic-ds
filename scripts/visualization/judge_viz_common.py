"""
Shared constants/loaders for the judge-result visualization scripts. Kept separate from
judge_common.py, which is about producing judge results, not visualizing them.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Judge verdicts: judge_results/ if you ran the judge yourself, else the released copy.
JUDGE_DIR = PROJECT_ROOT / "judge_results"
if not JUDGE_DIR.exists():
    JUDGE_DIR = PROJECT_ROOT / "outputs" / "judge_verdicts"

AGENTS = ["vanilla", "aide", "mlevolve"]

# (id, short label, gate fact or None). Ids are the paper's Table 6 failure-mode ids. Order is
# the plotting order (ungated modes first, then gated modes); SEGMENT_COLORS below is indexed
# positionally against this list.
FAILURE_MODES = [
    ("FM02", "Test peeking", None),
    ("FM03", "No cleaning", None),
    ("FM08", "Scale before split", None),
    ("FM09", "Feat-sel before split", None),
    ("FM04", "Impute before split", None),
    ("FM10", "Wrong model type", None),
    ("FM06", "No val split", None),
    ("FM15", "One metric only", None),
    ("FM01", "Wrong task formulation", None),
    ("FM07", "No feature eng.", None),
    ("FM05", "Ignores temporal order", "has_temporal"),
    ("FM11", "No DL regularization", "used_deep_learning"),
    ("FM13", "Reuses test across variants", "model_comparison"),
    ("FM14", "Single eval, stochastic model", "uses_stochastic_model"),
    ("FM12", "Naive ensembling", "ensembling"),
]
FM_IDS = [fid for fid, _, _ in FAILURE_MODES]

# "key" matches judge_common.MODEL_RUN_DIRS and the on-disk directory name
# judge_results/<agent>/<key>/; "label" is the display name used in figures and tables.
MODELS = [
    {"key": "gemini-3.6-flash", "label": "Gemini-3.6"},
    {"key": "gemini-3.7-flash", "label": "Gemini-3.7"},
    {"key": "qwen3.5-4b", "label": "Qwen-4B"},
    {"key": "qwen3.8-27b", "label": "Qwen-27B"},
]

# Display name for each agent id (AGENTS keeps the lowercase id used in directory names).
AGENT_DISPLAY = {"vanilla": "ReAct", "aide": "AIDE", "mlevolve": "MLEvolve"}

COLUMNS = [(agent, m["key"], m["label"]) for agent in AGENTS for m in MODELS]

FM_LABEL = {
    "FM01": "Wrong task formulation",
    "FM02": "Peeking at test data during EDA",
    "FM03": "Not cleaning data",
    "FM04": "Imputation before splitting",
    "FM05": "Ignoring temporal order",
    "FM06": "No train/val split",
    "FM07": "No feature engineering",
    "FM08": "Scaling before splitting",
    "FM09": "Feature selection before splitting",
    "FM10": "Wrong model type for task",
    "FM11": "No regularisation in DL",
    "FM12": "Naive ensembling",
    "FM13": "Reusing test set across variants",
    "FM14": "Single eval of stochastic model",
    "FM15": "Using only one eval metric",
}
FM_CATEGORY = {
    "FM01": "other", "FM02": "leak", "FM03": "other", "FM04": "leak",
    "FM05": "leak", "FM06": "other", "FM07": "other", "FM08": "leak",
    "FM09": "leak", "FM10": "other", "FM11": "other",
    "FM12": "other", "FM13": "leak", "FM14": "other", "FM15": "other",
}
CATEGORY_COLOR = {"leak": "#CD0324", "other": "#3468DC"}
FM_DISPLAY_ORDER = [f"FM{i:02d}" for i in range(1, 16)]   # paper-table order
assert set(FM_DISPLAY_ORDER) == set(FM_LABEL) == set(FM_CATEGORY) == set(FM_IDS)

SEGMENT_COLORS = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",             # FM02, FM03, FM08, FM09
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",  # FM04, FM10, FM06, FM15, FM01
    "#8c564b",                                              # FM07
    "#e377c2", "#bcbd22", "#dbdb8d",                        # FM05, FM11, FM13
    "#17becf", "#9edae5",                                   # FM14, FM12
]
OTHER_COLOR = "#8a8f99"


def result_dir(agent: str, model: str) -> Path:
    # Verdicts live in <agent>/<model>/; runs of gemini-3.6-flash judged before per-model
    # subdirectories existed sit directly in <agent>/.
    d = JUDGE_DIR / agent / model
    if model == "gemini-3.6-flash" and not d.exists():
        return JUDGE_DIR / agent
    return d


def cell_state(entry: dict) -> dict:
    if entry is None:
        return {"s": "missing"}
    if entry.get("applicable") is False:
        return {"s": "na"}
    present = entry.get("present")
    if present is True:
        return {
            "s": "present",
            "sev": entry.get("severity"),
            "reason": (entry.get("reasoning") or "")[:220],
            "evidence": (entry.get("evidence") or "")[:220],
        }
    if present is False:
        return {"s": "absent"}
    if present == "cannot_determine":
        return {"s": "cannot_determine", "reason": (entry.get("reasoning") or "")[:220]}
    return {"s": "missing"}


def load_agent_model(agent: str, model: str) -> dict:
    """task_id -> {cells: [cell_state per FM_IDS], no_code, other, stage_count, stage_count_max}"""
    tasks = {}
    d = result_dir(agent, model)
    if not d.exists():
        return tasks
    for path in sorted(d.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        fm = data.get("failure_modes", {})
        if not fm and data.get("no_code") is None:
            continue  # parse_error entries have no failure_modes -- not a judged result
        cells = [cell_state(fm.get(fid)) for fid in FM_IDS]
        applicable_cells = [c for c in cells if c["s"] != "na"]
        no_code = data.get("no_code")
        if no_code is None:
            no_code = len(applicable_cells) > 0 and all(c["s"] == "cannot_determine" for c in applicable_cells)
        tasks[path.stem] = {
            "cells": cells,
            "no_code": no_code,
            "other": len(data.get("other_failure_modes", [])),
            "stage_count": data.get("stage_count"),
            "stage_count_max": data.get("stage_count_max"),
        }
    return tasks


def compute_taxonomy_column(agent: str, model: str) -> dict:
    """Average findings-per-task, decomposed by failure mode -- see
    the stacked-bar script's docstring for why this
    (not stacked presence rates) is the statistically sound quantity here."""
    tasks = load_agent_model(agent, model)
    n_judged = sum(1 for t in tasks.values() if not t["no_code"])
    if n_judged == 0:
        return {"n_judged": 0, "segments": [0.0] * len(FAILURE_MODES), "other": 0.0, "n_nocode": len(tasks)}
    segments = []
    for i in range(len(FAILURE_MODES)):
        n_present = sum(1 for t in tasks.values() if not t["no_code"] and t["cells"][i]["s"] == "present")
        segments.append(n_present / n_judged)
    other_total = sum(t["other"] for t in tasks.values() if not t["no_code"])
    return {
        "n_judged": n_judged,
        "segments": segments,
        "other": other_total / n_judged,
        "n_nocode": sum(1 for t in tasks.values() if t["no_code"]),
    }


def compute_taxonomy_column_applicable(agent: str, model: str) -> dict:
    """Per-mode presence rate CONDITIONED on applicability (present / applicable), for a HEATMAP TABLE
    read as independent per-mode rates -- NOT for the stacked bar, which deliberately wants
    compute_taxonomy_column's present / n_judged (see that function's docstring: dividing every mode
    by the same n_judged is what makes the segments summable into one meaningful bar height).

    Returns n_applicable per mode alongside each rate (None where n_applicable==0)
    so callers can flag/suppress cells whose applicable pool is too small to
    trust -- see build_judge_heatmap_table_by_agent.py's MIN_APPLICABLE.
    """
    tasks = load_agent_model(agent, model)
    n_judged = sum(1 for t in tasks.values() if not t["no_code"])
    if n_judged == 0:
        return {"n_judged": 0, "n_applicable": [0] * len(FAILURE_MODES),
                "rates": [None] * len(FAILURE_MODES)}
    judged_tasks = [t for t in tasks.values() if not t["no_code"]]
    n_applicable = []
    rates = []
    for i in range(len(FAILURE_MODES)):
        applicable_cells = [t["cells"][i] for t in judged_tasks if t["cells"][i]["s"] != "na"]
        n_app = len(applicable_cells)
        n_pres = sum(1 for c in applicable_cells if c["s"] == "present")
        n_applicable.append(n_app)
        rates.append(n_pres / n_app if n_app else None)
    return {"n_judged": n_judged, "n_applicable": n_applicable, "rates": rates}
