"""
Loaders for the curated test suite's judge results (outputs/dsts_judge_results/): per-check
pass/fail/crash rows by agent, model, failure mode and hint level. Used by plot_failure_modes.py
and plot_failure_modes_by_hint.py.
"""
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


OUT_DIR = PROJECT_ROOT / "analysis" / "figures"


AGENTS = ["Vanilla", "AIDE", "MLEvolve"]


FMS = ["TF01_goal_imbalance", "TF02_player_rep_norm", "TF03_correlation_causation",
       "TF04_balance_torque", "TF05_credit_card_leakage", "TF06_metric_select"]


JUDGE_RESULTS_DIR = PROJECT_ROOT / "outputs" / "dsts_judge_results"


FM_CHECK_COLUMNS = {
    "TF01_goal_imbalance": ["uses_macro_average", "checks_class_distribution", "uses_resampling",
                             "uses_stratified_split", "uses_class_weight"],
    "TF02_player_rep_norm": ["handles_na_string_in_international_caps", "handles_unit_suffix_in_height"],
    "TF03_correlation_causation": ["column_x_included", "applies_causal_reasoning"],
    "TF04_balance_torque": ["has_torque_features", "has_torque_difference"],
    "TF05_credit_card_leakage": ["used_leakage_features"],
    # TF06 is named for METRIC SELECTION, not imbalance handling -- selection_metric is the
    # field that targets it (an enum incl. 'accuracy', the wrong choice under imbalance, vs.
    # 'macro_f1'/'roc_auc'/'f1_binary'/'f1_weighted'/'average_precision'). The previous version
    # of this line used imbalance_methods, which measures the same construct as TF01 and left
    # selection_metric (present in every judged row) unused. Corrected; see _check_outcome's
    # TF06 branch for how selection_metric == 'accuracy' -> fail, 'other'/missing -> excluded
    # (like a crash) rather than silently counted as pass or fail.
    "TF06_metric_select": ["selection_metric"],
}


INVERTED_CHECK_COLUMNS = {"column_x_included", "used_leakage_features"}


def _check_outcome(col: str, raw_value: str) -> str:
    """"" -> "crash"; else "pass"/"fail", respecting INVERTED_CHECK_COLUMNS."""
    if raw_value == "":
        return "crash"
    is_true = raw_value == "True"
    if col in INVERTED_CHECK_COLUMNS:
        is_true = not is_true
    return "pass" if is_true else "fail"


def _judge_dir_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


LEAKFREE_MODELS = {"gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash", "openrouter/gpt-5.6-sol"}


LEAKFREE_FMS = {"TF04_balance_torque", "TF06_metric_select"}


def _result_dir_for(agent: str, model: str, fm: str) -> Path:
    slug = _judge_dir_slug(model)
    if fm in LEAKFREE_FMS and model in LEAKFREE_MODELS:
        leakfree_dir = JUDGE_RESULTS_DIR / f"Leakfree_{AGENT_DIR_NAME[agent]}_{slug}"
        if leakfree_dir.exists():
            return leakfree_dir
    return JUDGE_RESULTS_DIR / f"{AGENT_DIR_NAME[agent]}_{slug}"


AGENT_DIR_NAME = {"Vanilla": "Vanilla", "AIDE": "Aide", "MLEvolve": "Mlevolve"}


NO_HINT_ONLY = True


_NO_HINT_VARIANT = re.compile(r"_01_rep\d+$")


def load_judge_csv_by_rep(agent: str, model: str) -> dict:
    """(fm) -> {rep_int: {"pass","fail","crash"}} -- same per-check-column
    pass/fail/crash logic as load_judge_csv, but split PER REPEAT instead of
    pooled across all 5. Lets a caller compute a real mean +/- variance
    across the 5 independent repeats per (agent, model, fm), instead of
    load_judge_csv's single pooled rate."""
    import csv as _csv

    out = {}
    for fm, check_cols in FM_CHECK_COLUMNS.items():
        per_rep = defaultdict(lambda: {"pass": 0, "fail": 0, "crash": 0})
        csv_path = _result_dir_for(agent, model, fm) / f"{fm}_eval.csv"
        if not csv_path.exists():
            out[fm] = {}
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                task_id = row.get("task_id", "").strip()
                if NO_HINT_ONLY and not _NO_HINT_VARIANT.search(task_id):
                    continue  # hinted variant (_02/_03/...) -- see NO_HINT_ONLY above
                rep_m = re.search(r"_rep(\d+)$", task_id)
                if not rep_m:
                    continue  # pre-repeats n=1 run -- see REPEATS_ONLY above
                rep = int(rep_m.group(1))
                crashed_row = row.get("pipeline_failed", "").strip() == "True"
                if fm == "TF06_metric_select":
                    if crashed_row:
                        per_rep[rep]["crash"] += 1
                        continue
                    raw = (row.get("selection_metric") or "").strip()
                    if raw in ("", "other"):
                        per_rep[rep]["crash"] += 1
                        continue
                    per_rep[rep]["fail" if raw == "accuracy" else "pass"] += 1
                    continue
                for c in check_cols:
                    if crashed_row:
                        per_rep[rep]["crash"] += 1
                        continue
                    per_rep[rep][_check_outcome(c, row.get(c, "").strip())] += 1
        out[fm] = dict(per_rep)
    return out


def load_judge_csv_by_level_rep(agent: str, model: str) -> dict:
    """(fm) -> {(hint_level:int, rep:int): {"pass","fail","crash"}} -- identical per-check
    pass/fail/crash logic to load_judge_csv_by_rep (same _check_outcome polarity, same crash
    handling, REPEATS_ONLY), but keeps EVERY hint level (_01, _02, ...) instead of only _01,
    keyed by (level, rep). Used by plot_failure_modes_by_hint.py."""
    import csv as _csv

    variant_rep = re.compile(r"_(\d{2})_rep(\d+)$")
    out = {}
    for fm, check_cols in FM_CHECK_COLUMNS.items():
        per = defaultdict(lambda: {"pass": 0, "fail": 0, "crash": 0})
        csv_path = _result_dir_for(agent, model, fm) / f"{fm}_eval.csv"
        if not csv_path.exists():
            out[fm] = {}
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                m = variant_rep.search(row.get("task_id", "").strip())
                if not m:
                    continue  # pre-repeats n=1 run (or no variant suffix)
                key = (int(m.group(1)), int(m.group(2)))
                crashed_row = row.get("pipeline_failed", "").strip() == "True"
                if fm == "TF06_metric_select":
                    if crashed_row:
                        per[key]["crash"] += 1
                        continue
                    raw = (row.get("selection_metric") or "").strip()
                    if raw in ("", "other"):
                        per[key]["crash"] += 1
                        continue
                    per[key]["fail" if raw == "accuracy" else "pass"] += 1
                    continue
                for c in check_cols:
                    if crashed_row:
                        per[key]["crash"] += 1
                        continue
                    per[key][_check_outcome(c, row.get(c, "").strip())] += 1
        out[fm] = dict(per)
    return out


def fm_fail_rate(counts: dict) -> float:
    denom = counts["pass"] + counts["fail"]
    return counts["fail"] / denom if denom else 0.0


MODELS_GRID = [
    ("openrouter/qwen-3.8-27b", "Qwen-3.8-27B"),
    ("gemini/gemini-3.6-flash", "gemini-3.6-flash"),
    ("openrouter/gpt-5.6-sol", "gpt-5.6-sol"),
    ("gemini/gemini-3.7-flash", "gemini-3.7-flash"),
]


def build_rep_fail_rate_rows(agents: list, models: list) -> "pd.DataFrame":
    """Long-form rows: one per (fm, agent, model, rep) with that repeat's own fail rate -- the raw
    replicate-level data cnsplots' barplot(errorbar="sd") needs to compute a real mean+SD per bar,
    instead of load_judge_csv's single pooled rate.
    """
    import pandas as pd

    rows = []
    for agent in agents:
        for model_key, model_label in models:
            per_fm = load_judge_csv_by_rep(agent, model_key)
            for fm, per_rep in per_fm.items():
                for rep, counts in per_rep.items():
                    if counts["pass"] + counts["fail"] == 0:
                        continue  # this rep crashed entirely -- no real judgment
                    rows.append({
                        "fm": fm, "agent": agent, "model": model_label,
                        "rep": rep, "fail_rate": fm_fail_rate(counts),
                    })
    return pd.DataFrame(rows)


def crashed_cells(agents: list, models: list) -> set:
    """Callers should mark these explicitly (e.g. "crashed" instead of a bar) rather than let a missing
    cell silently read as "0.00 fail rate" or just an unexplained gap.
    """
    out = set()
    for agent in agents:
        for model_key, model_label in models:
            per_fm = load_judge_csv_by_rep(agent, model_key)
            for fm, per_rep in per_fm.items():
                if not per_rep:
                    continue  # no CSV / not judged yet -- different from "judged but crashed"
                if all(c["pass"] + c["fail"] == 0 for c in per_rep.values()):
                    out.add((fm, agent, model_label))
    return out


# (model_id, label) pairs for the three models evaluated on the test suite
MODELS = [
    ("gemini/gemini-3.6-flash", "gemini-3.6-flash"),
    ("gemini/gemini-3.7-flash", "gemini-3.7-flash"),
    ("openrouter/gpt-5.6-sol", "gpt-5.6-sol"),
]
