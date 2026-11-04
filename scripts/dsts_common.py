"""
Shared helpers for the DSTS (unit-test suite) adapters --
dsts_run_vanilla.py, dsts_run_aide.py, dsts_run_mlevolve.py. Reads the
TF0X workload JSON + real train/test data directly from
$DSTS_ROOT (nothing there is copied or modified),
independent of which of our 3 agents is running it. See dsts_run_vanilla.py
for the full pipeline design notes.
"""
import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DSTS_ROOT = Path(os.environ.get("DSTS_ROOT", PROJECT_ROOT / "dsts_suite"))  # checkout of the DSTS test suite
WORKLOAD_DIR = DSTS_ROOT / "src" / "agents" / "dsguru" / "workload"
DATA_ROOT = DSTS_ROOT / "data"

ALL_WORKLOADS = [
    "TF01_goal_imbalance",
    "TF02_player_rep_norm",
    "TF03_correlation_causation",
    "TF04_balance_torque",
    "TF05_credit_card_leakage",
    "TF06_metric_select",
]

def external_name(name: str) -> str:
    """The test-suite repo names its workload files and data directories FM0x_*; this package
    uses the paper's TF0x_* ids. Translate at the boundary with that repo."""
    return re.sub(r"^TF(0[1-6])", r"FM\1", name)


def paper_name(name: str) -> str:
    return re.sub(r"^FM(0[1-6])", r"TF\1", name)


# Workload JSON filename -> data/ directory name, where they differ (none among the six used).
DATA_DIR_ALIASES: dict = {}

OUTPUT_REQUIREMENT = """

Formatting instructions: after computing the requested metric(s), print a single
line at the end of your output containing a JSON object with one key per computed
metric, mapped to its numeric value. Use these key names for the corresponding
metrics if requested: accuracy, macro_f1_score, macro_precision, macro_recall.
"""


LEAKFREE_DATA_ROOT = PROJECT_ROOT / "dsts_leakfree_data"

_TF06_LEAKFREE_FMT = """

Formatting instructions: print a single JSON object as the last line of your output
with these keys: "selection_metric" (the name of the metric you used to compare and
select the best model), "validation_score" (that metric's value for the model you
selected, evaluated on the validation data), "selected_model" (the name of the model
you selected), and "test_score" (the same metric's value evaluated on the test set).
"""

_TF06_LEAKFREE_Q01 = (
    "Using 'train.csv' as the training set and 'test.csv' as the test set, preprocess "
    "the data as needed and train several classification models (for example, Logistic "
    "Regression, Random Forest, Gradient Boosting, and XGBoost if available) to predict "
    "'goals_label' (1 or 0). Use only numerical feature columns and fill missing values "
    "with 0. Compare the models on the training and validation data, select the "
    "best-performing classifier, and report: (1) the metric used to select the best "
    "model and its validation score, (2) the name of the selected model, (3) the model "
    "performance on the test set."
)
_TF06_LEAKFREE_Q02 = _TF06_LEAKFREE_Q01.replace(
    "with 0. Compare",
    "with 0. The target class is highly imbalanced: only about 5% of players are "
    "labeled as 1. Compare",
)

LEAKFREE_WORKLOADS = {
    "TF04_balance_torque_leakfree": {
        "real_workload": "TF04_balance_torque",
        "data_dir": "tf04",
        "output_requirement": None,  # TF04's own OUTPUT_REQUIREMENT isn't the leak, keep it
        "tasks": [
            {
                "id": "TF04_balance_torque_leakfree_01",
                "data_sources": ["train.csv", "test.csv"],
                "query": (
                    "Using 'train.csv' as the training set and 'test.csv' as the test "
                    "set, preprocess the data as needed, build a classification model "
                    "to predict 'Class Name' (L, R, or B), and report accuracy and "
                    "macro F1 on the test set as the final answer. Note: the CSV files "
                    "have no row index column; load them with pd.read_csv() without "
                    "specifying index_col."
                ),
            },
            {
                "id": "TF04_balance_torque_leakfree_02",
                "data_sources": ["train.csv", "test.csv"],
                "query": (
                    "Using 'train.csv' as the training set and 'test.csv' as the test "
                    "set, preprocess the data as needed. Consider whether interaction "
                    "features between the weight and distance columns may be useful "
                    "for prediction. Then build a classification model to predict "
                    "'Class Name' (L, R, or B), and report accuracy and macro F1 on "
                    "the test set as the final answer. Note: the CSV files have no row "
                    "index column; load them with pd.read_csv() without specifying "
                    "index_col."
                ),
            },
            {
                "id": "TF04_balance_torque_leakfree_03",
                "data_sources": ["train.csv", "test.csv"],
                "query": (
                    "Using 'train.csv' as the training set and 'test.csv' as the test "
                    "set. Create useful balance-related features from the weight and "
                    "distance columns before training. Then build a classification "
                    "model to predict 'Class Name' (L, R, or B), and report accuracy "
                    "and macro F1 on the test set as the final answer. Note: the CSV "
                    "files have no row index column; load them with pd.read_csv() "
                    "without specifying index_col."
                ),
            },
            {
                "id": "TF04_balance_torque_leakfree_04",
                "data_sources": ["train.csv", "test.csv"],
                "query": (
                    "Using 'train.csv' as the training set and 'test.csv' as the test "
                    "set. Compute physically meaningful torque-related features using "
                    "the weight and distance columns, then build a classification "
                    "model to predict 'Class Name' (L, R, or B). Report accuracy and "
                    "macro F1 on the test set as the final answer. Note: the CSV files "
                    "have no row index column; load them with pd.read_csv() without "
                    "specifying index_col."
                ),
            },
        ],
    },
    "TF06_metric_select_leakfree": {
        "real_workload": "TF06_metric_select",
        "data_dir": "tf06",
        "output_requirement": _TF06_LEAKFREE_FMT,
        "tasks": [
            {"id": "TF06_metric_select_leakfree_01", "data_sources": ["train.csv", "test.csv"], "query": _TF06_LEAKFREE_Q01},
            {"id": "TF06_metric_select_leakfree_02", "data_sources": ["train.csv", "test.csv"], "query": _TF06_LEAKFREE_Q02},
        ],
    },
}


def real_workload_for(workload: str) -> str:
    """Maps a leak-free workload id to the real TF name its judge_criteria file is under (the criteria
    file itself is unaffected by either leak -- see the NOTE above -- so leak-free runs are judged
    with the
    """
    return LEAKFREE_WORKLOADS.get(workload, {}).get("real_workload", workload)


def load_workload(workload: str) -> list:
    if workload in LEAKFREE_WORKLOADS:
        return LEAKFREE_WORKLOADS[workload]["tasks"]
    path = WORKLOAD_DIR / f"{external_name(workload)}.json"
    if not path.exists():
        raise FileNotFoundError(f"workload not found: {path}")
    tasks = json.loads(path.read_text())
    for t in tasks:
        t["id"] = paper_name(t["id"])
    return tasks


def data_file_paths(task: dict, workload: str) -> list:
    if workload in LEAKFREE_WORKLOADS:
        data_dir = LEAKFREE_WORKLOADS[workload]["data_dir"]
        return [(LEAKFREE_DATA_ROOT / data_dir / "input" / f).resolve() for f in task["data_sources"]]
    data_dir_name = DATA_DIR_ALIASES.get(workload, workload)
    return [(DATA_ROOT / external_name(data_dir_name) / "input" / f).resolve() for f in task["data_sources"]]


def build_desc_text(task: dict, workload: str) -> str:
    paths = data_file_paths(task, workload)
    files_block = "\n".join(f"  - {p}" for p in paths)
    fmt = OUTPUT_REQUIREMENT
    if workload in LEAKFREE_WORKLOADS:
        override = LEAKFREE_WORKLOADS[workload]["output_requirement"]
        if override is not None:
            fmt = override
    return (
        f"{task['query']}\n\n"
        f"Data files (use these exact local paths):\n{files_block}\n"
        f"{fmt}"
    )


def extract_json_blob(text: str) -> dict | None:
    """Best-effort: find the last JSON object printed to stdout."""
    import ast
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        blob = match.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
        try:
            val = ast.literal_eval(blob)
            if isinstance(val, dict):
                return val
        except (ValueError, SyntaxError):
            pass
    return None


def clear_stale_run_output(task_dir: Path) -> None:
    """Remove files from a previous attempt at this task before writing a new one -- otherwise a stale
    errors-0.txt/answer.json from an earlier failed run can sit next to a later successful run's
    files, maki
    """
    for name in ("generated_code.py", "stdout.txt", "errors-0.txt", "answer.json"):
        f = task_dir / name
        if f.exists():
            f.unlink()


def scoped_max_tokens(model_cfg: dict, minimum: int = 16384) -> dict:
    """DSTS queries are more complex/multi-step than our own benchmark prompts and some models (e.g.
    gemini/gemini-3.6-flash) have no explicit max_tokens override in model_registry.json, defaulting
    to 4096 -- too small here, causing truncated (unparseable) responses.
    """
    cfg = dict(model_cfg)
    cfg["max_tokens"] = max(cfg.get("max_tokens", 0) or 0, minimum)
    return cfg


def extract_aide_best_metric(stdout: str) -> dict | None:
    """AIDE never prints the requested {"accuracy": ..., "macro_f1_score": ...}
    block -- it only reports its own internal validation metric as a plain
    'Best metric: X' line. Falls back to that so a run still yields a
    comparable numeric result instead of 'no answer extracted', while
    keeping the key name distinct so it's never mistaken for a real
    accuracy/macro_f1/... JSON answer."""
    match = re.search(r"Best metric:\s*([-+]?\d*\.?\d+)", stdout)
    if not match:
        return None
    return {"aide_best_metric": float(match.group(1))}


def extract_mlevolve_best_metric(stdout: str) -> dict | None:
    """MLEvolve never prints the requested JSON block either -- its own
    internal validation metric only ever appears in log lines like
    '... Task completed: node_id=..., step=5, is_buggy=False, metric=0.9106'.
    Falls back to the last such line, distinct key name for the same reason
    as extract_aide_best_metric."""
    matches = list(re.finditer(r"Task completed:.*?metric=([-+]?\d*\.?\d+)", stdout))
    if not matches:
        return None
    return {"mlevolve_best_metric": float(matches[-1].group(1))}


def execute_and_extract(code: str, task_dir: Path, timeout: int = 300):
    """Actually run an agent's winning solution ourselves and extract JSON from its real stdout, same as
    vanilla's own execute_code/extract_json_blob path.
    """
    from run_vanilla import execute_code  # local import, avoids a hard dependency at module load
    stdout, stderr, returncode = execute_code(code, task_dir, timeout)
    answer = extract_json_blob(stdout) if returncode == 0 else None
    return answer, stdout, stderr, returncode


def write_judge_output(
    task_dir: Path,
    code: str | None,
    stdout: str,
    stderr: str,
    returncode: int,
    fallback_extractor=None,
) -> dict:
    """Write a task_dir judge.py can scan as-is (see dsts_run_vanilla.py header).
    fallback_extractor(stdout) -> dict | None is tried only when no JSON blob
    was found, e.g. extract_aide_best_metric for AIDE's non-compliant output."""
    task_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_run_output(task_dir)
    if code:
        (task_dir / "generated_code.py").write_text(code, encoding="utf-8")
    (task_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    if stderr.strip():
        (task_dir / "errors-0.txt").write_text(stderr, encoding="utf-8")

    answer = extract_json_blob(stdout) if returncode == 0 else None
    if answer is None and returncode == 0 and fallback_extractor is not None:
        answer = fallback_extractor(stdout)
    if answer is not None:
        (task_dir / "answer.json").write_text(json.dumps({"answer": answer}, indent=2), encoding="utf-8")

    return {"success": returncode == 0 and answer is not None, "returncode": returncode, "answer": answer}
