"""Runs the real run_judge two-stage (structure-extraction + gated failure- mode judging) pipeline
against the three code files used in the child-mind-institute-problematic-internet-use case-study
DAGs (analysis/hsc_kaggle_prep/{vanilla,aide,mlevolve}/child-mind-institute-problematic-internet-use.py).

Run: python3 scripts/run_judge_casestudy.py
Output: analysis/hsc_kaggle_prep/analysis/judge_casestudy_results/{agent}.json
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_vanilla import load_registry, resolve_model_cfg, call_llm  # noqa: E402
from judge_spec import JUDGE_MODEL  # noqa: E402
from judge_common import extract_json, dataset_level_facts  # noqa: E402
from dsts_common import scoped_max_tokens  # noqa: E402
from run_judge import (  # noqa: E402
    load_schema, load_preconditions, run_extraction, is_applicable,
    ALL_FAILURE_MODES, GATED_JUDGE_PROMPT,
)

TASK_ID = "child-mind-institute-problematic-internet-use"
SOURCE = "kaggle"
OUT_DIR = PROJECT_ROOT / "analysis" / "hsc_kaggle_prep" / "analysis" / "judge_casestudy_results"

CASE_STUDY_SOURCES = {
    "vanilla": {
        "code": PROJECT_ROOT / "analysis/hsc_kaggle_prep/vanilla/child-mind-institute-problematic-internet-use.py",
        "log_dir": PROJECT_ROOT / "hsc_runs/vanilla__gemini-3.6-flash__zero_shot_hsc/child-mind-institute-problematic-internet-use",
        "log_files": ["stdout.txt", "stderr.txt"],
    },
    "aide": {
        "code": PROJECT_ROOT / "analysis/hsc_kaggle_prep/aide/child-mind-institute-problematic-internet-use.py",
        "log_dir": PROJECT_ROOT / "hsc_runs/aide__gemini-gemini-3.6-flash__10steps/child-mind-institute-problematic-internet-use",
        "log_files": ["aide_stdout.txt"],
    },
    "mlevolve": {
        "code": PROJECT_ROOT / "analysis/hsc_kaggle_prep/mlevolve/child-mind-institute-problematic-internet-use.py",
        "log_dir": PROJECT_ROOT / "hsc_runs/mlevolve__gemini-gemini-3.6-flash__10steps/child-mind-institute-problematic-internet-use",
        "log_files": ["mlevolve_stdout.txt"],
    },
}


def load_process_log(log_dir: Path, log_files: list, max_chars: int = 6000) -> str:
    parts = []
    for fname in log_files:
        p = log_dir / fname
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                parts.append(f"--- {fname} (last {max_chars} chars) ---\n{content[-max_chars:]}")
    return "\n\n".join(parts) if parts else "(no process log available)"


def build_case_study_ctx(agent: str) -> dict:
    src = CASE_STUDY_SOURCES[agent]
    task_info = json.loads((PROJECT_ROOT / "datasets" / SOURCE / TASK_ID / "task.json").read_text())
    facts = dataset_level_facts(TASK_ID, SOURCE)
    return {
        "task_id": TASK_ID,
        "description": task_info.get("description") or TASK_ID,
        "task_type": task_info.get("task_type", "unknown"),
        "target_column": task_info.get("target_column", "unknown"),
        "metric": task_info.get("metric", "unknown"),
        "metric_direction": "higher" if task_info.get("metric_higher_is_better", True) else "lower",
        "domain": task_info.get("domain", "unknown"),
        "has_temporal": facts["has_temporal"],
        "dataset_facts": facts,
        "code": src["code"].read_text(encoding="utf-8", errors="replace"),
        "process_log": load_process_log(src["log_dir"], src["log_files"]),
    }


def run_one_case_study(agent: str, judge_model_cfg: dict, schema: list, preconditions: dict) -> dict:
    ctx = build_case_study_ctx(agent)

    extracted = run_extraction(ctx["code"], judge_model_cfg, schema)

    gating_facts = {**ctx.get("dataset_facts", {}), **extracted}
    applicable_ids = [fid for fid in ALL_FAILURE_MODES if is_applicable(fid, gating_facts, preconditions)]
    gated_out_ids = [fid for fid in ALL_FAILURE_MODES if fid not in applicable_ids]

    applicable_modes = {fid: ALL_FAILURE_MODES[fid] for fid in applicable_ids}
    failure_mode_list = "\n".join(f"**{fid} — {fm['name']}**\n{fm['definition']}\n" for fid, fm in applicable_modes.items())
    extracted_facts = "\n".join(f"- {k}: {v}" for k, v in extracted.items())

    prompt = GATED_JUDGE_PROMPT.format(**ctx, extracted_facts=extracted_facts, failure_mode_list=failure_mode_list)
    response, _ = call_llm([{"role": "user", "content": prompt}], judge_model_cfg)
    raw = extract_json(response)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": TASK_ID, "agent": agent, "parse_error": True, "raw_response": response,
                "extracted_structure": extracted}

    result.setdefault("failure_modes", {})
    for fid in gated_out_ids:
        result["failure_modes"][fid] = {"applicable": False, "present": None}
    for fid in applicable_ids:
        if fid in result["failure_modes"]:
            result["failure_modes"][fid]["applicable"] = True

    result["agent"] = agent
    result["target_model"] = "gemini-3.6-flash (case-study exact snapshot)"
    result["extracted_structure"] = extracted
    result["stage_count"] = sum(1 for v in extracted.values() if v)
    result["stage_count_max"] = len(extracted)
    result["_code_path"] = str(CASE_STUDY_SOURCES[agent]["code"].relative_to(PROJECT_ROOT))
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_schema()
    preconditions = load_preconditions()
    registry = load_registry()
    model_cfg = scoped_max_tokens(resolve_model_cfg(JUDGE_MODEL, registry), minimum=32768)

    print(f"Judge model: {JUDGE_MODEL}")
    for agent in ["vanilla", "aide", "mlevolve"]:
        print(f"[{agent}] judging case-study code ({CASE_STUDY_SOURCES[agent]['code'].name}) ...", end=" ", flush=True)
        result = run_one_case_study(agent, model_cfg, schema, preconditions)
        out_path = OUT_DIR / f"{agent}.json"
        out_path.write_text(json.dumps(result, indent=2))
        if result.get("parse_error"):
            print("PARSE ERROR")
        else:
            present = [fid for fid, v in result.get("failure_modes", {}).items()
                       if isinstance(v, dict) and v.get("present") is True]
            other = [o["name"] for o in result.get("other_failure_modes", [])]
            print(f"done — present: {present} | other: {other}")


if __name__ == "__main__":
    main()
