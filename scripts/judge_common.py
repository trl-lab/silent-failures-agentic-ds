"""
Shared helpers for the large-scale-experiment LLM-as-judge (run_judge.py). Reads directly
from the large-scale run directories (runs/, aide_runs/, mlevolve_runs/),
not the DSTS suite.
"""
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TARGET_MODEL = "gemini-3.6-flash"

# Real, verified run dir per (target model, agent) -- checked against what's
# actually on disk, since prompt-variant completeness flips between models:
# gemini-3.6-flash's vanilla few_shot_role has only 1 completed task (so
# zero_shot is the real one); gemini-3.7-flash's vanilla zero_shot has only
# 1 completed task (so few_shot_role is the real one there instead). Don't
# assume the same variant name is "the" run across models -- verify via
# completed_tasks() before adding a new model here.
MODEL_RUN_DIRS = {
    "gemini-3.6-flash": {
        "vanilla":  PROJECT_ROOT / "runs" / "vanilla__gemini-3.6-flash__zero_shot",
        "aide":     PROJECT_ROOT / "aide_runs" / "aide__gemini-3.6-flash__10steps",
        "mlevolve": PROJECT_ROOT / "mlevolve_runs" / "mlevolve__gemini-gemini-3.6-flash__10steps",
    },
    "gemini-3.7-flash": {
        "vanilla":  PROJECT_ROOT / "runs" / "vanilla__gemini-gemini-3.7-flash__few_shot_role",
        "aide":     PROJECT_ROOT / "aide_runs" / "aide__gemini-gemini-3.7-flash__10steps",
        "mlevolve": PROJECT_ROOT / "mlevolve_runs" / "mlevolve__gemini-gemini-3.7-flash__10steps",
    },
    "qwen3.5-4b": {
        "vanilla":  PROJECT_ROOT / "runs" / "vanilla__vllm-qwen3.5-4b__zero_shot",
        "aide":     PROJECT_ROOT / "aide_runs" / "aide__vllm-qwen3.5-4b__10steps",
        "mlevolve": PROJECT_ROOT / "mlevolve_runs" / "mlevolve__vllm-qwen3.5-4b__10steps",
    },
    # OpenRouter backend only (single backend + 2h cap for all 220 tasks); the earlier
    # self-hosted vllm-qwen3.8-27b dirs are a different backend/cap and are not mixed in.
    "qwen3.8-27b": {
        "vanilla":  PROJECT_ROOT / "runs" / "vanilla__openrouter-qwen-3.8-27b__few_shot_role",
        "aide":     PROJECT_ROOT / "aide_runs" / "aide__openrouter-qwen-3.8-27b__10steps",
        "mlevolve": PROJECT_ROOT / "mlevolve_runs" / "mlevolve__openrouter-qwen-3.8-27b__10steps",
    },
}

# Default target model's run directories, keyed by agent.
AGENT_CONFIGS = MODEL_RUN_DIRS[DEFAULT_TARGET_MODEL]


def run_dir(agent: str, target_model: str = DEFAULT_TARGET_MODEL) -> Path:
    return MODEL_RUN_DIRS[target_model][agent]

DATASETS = PROJECT_ROOT / "datasets"

# Per-agent: (code filename, [process-log filenames in priority order]).
# Process logs are the agent's own human-readable step/reasoning output --
# NOT the raw journal.json state dumps (internal fields, not prose) and NOT
# the final code -- kept as a clearly separate prompt section so the judge
# never mistakes an earlier discarded attempt for the final solution.
AGENT_FILES = {
    "vanilla":  {"code": "generated_code.py", "logs": ["stdout.txt", "stderr.txt"]},
    "aide":     {"code": "best_solution.py", "logs": ["aide_stdout.txt"]},
    "mlevolve": {"code": None, "logs": ["mlevolve_stdout.txt"]},  # code resolved via glob, see load_code
}


def completed_tasks(agent: str, target_model: str = DEFAULT_TARGET_MODEL) -> set:
    d = run_dir(agent, target_model)
    return set(x.name for x in d.iterdir() if x.is_dir() and (x / "result.json").exists())


def three_way_overlap(target_model: str = DEFAULT_TARGET_MODEL) -> set:
    return (completed_tasks("vanilla", target_model)
            & completed_tasks("aide", target_model)
            & completed_tasks("mlevolve", target_model))


def pick_task_subset(n: int = 50, seed: int = 42, target_model: str = DEFAULT_TARGET_MODEL) -> list:
    """Same n tasks used for all three agents -- fixed seed for reproducibility.
    For small-n smoke tests that want a shared, directly-comparable subset. NOT used by
    run_judge.py's main sweep -- see pick_task_subset_for_agent for why."""
    overlap = sorted(three_way_overlap(target_model))
    rng = random.Random(seed)
    rng.shuffle(overlap)
    return sorted(overlap[:n])


def pick_task_subset_for_agent(agent: str, n: int = 300, seed: int = 42,
                                target_model: str = DEFAULT_TARGET_MODEL) -> list:
    """Each agent's OWN full completed-task list, not the 3-way intersection."""
    ids = sorted(completed_tasks(agent, target_model))
    if n >= len(ids):
        return ids
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:n])


def load_code(agent: str, task_id: str, target_model: str = DEFAULT_TARGET_MODEL) -> str | None:
    task_dir = run_dir(agent, target_model) / task_id
    if agent == "mlevolve":
        # No single fixed filename -- MLEvolve's winning solution lives under
        # mlevolve_logs/<timestamp>_<task>/best_solution.py.
        matches = sorted(task_dir.glob("mlevolve_logs/*/best_solution.py"))
        return matches[-1].read_text(encoding="utf-8", errors="replace") if matches else None
    fname = AGENT_FILES[agent]["code"]
    path = task_dir / fname
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def load_process_log(agent: str, task_id: str, target_model: str = DEFAULT_TARGET_MODEL, max_chars: int = 6000) -> str:
    task_dir = run_dir(agent, target_model) / task_id
    parts = []
    for fname in AGENT_FILES[agent]["logs"]:
        p = task_dir / fname
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                parts.append(f"--- {fname} (last {max_chars} chars) ---\n{content[-max_chars:]}")
    return "\n\n".join(parts) if parts else "(no process log available -- e.g. vanilla is single-shot with no search/reasoning trace)"


def load_result(agent: str, task_id: str, target_model: str = DEFAULT_TARGET_MODEL) -> dict:
    path = run_dir(agent, target_model) / task_id / "result.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_task_info(task_id: str, source: str) -> dict:
    path = DATASETS / source / task_id / "task.json"
    return json.loads(path.read_text()) if path.exists() else {}


def dataset_level_facts(task_id: str, source: str) -> dict:
    """Facts about the DATASET itself, not any agent's code -- depend only on task_id/source, so computed
    identically regardless of which agent is being judged.
    """
    task_info = load_task_info(task_id, source)
    columns = [str(c) for c in task_info.get("columns", [])]
    target = str(task_info.get("target_column", ""))

    has_temporal = any(
        any(kw in c.lower() for kw in ("date", "time", "timestamp", "dt", "year", "month", "week"))
        for c in columns
    )
    return {"has_temporal": has_temporal}


def build_task_context(agent: str, task_id: str, target_model: str = DEFAULT_TARGET_MODEL) -> dict:
    result = load_result(agent, task_id, target_model)
    source = result.get("source", "")
    task_info = load_task_info(task_id, source)
    facts = dataset_level_facts(task_id, source)
    return {
        "task_id": task_id,
        "description": task_info.get("description", task_id),
        "task_type": task_info.get("task_type", result.get("task_type", "unknown")),
        "target_column": result.get("target_column", task_info.get("target_column", "unknown")),
        "metric": result.get("metric", task_info.get("metric", "unknown")),
        "metric_direction": "higher" if result.get("higher_is_better", True) else "lower",
        "domain": task_info.get("domain", "unknown"),
        "has_temporal": facts["has_temporal"],
        "dataset_facts": facts,
        "code": load_code(agent, task_id, target_model) or "(no code found)",
        "process_log": load_process_log(agent, task_id, target_model),
    }


def extract_json(text: str) -> str:
    import re
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        return m.group(1).strip()
    # opening fence without a closing one (or a stray closing fence)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text).strip()
