"""
LLM-as-judge runner. Two stages per task:
  1. Structure extraction: an LLM call reads the final code and returns yes/no facts about
     which pipeline stages are present (STRUCTURE_SCHEMA in judge_spec.py).
  2. Gated judging: those facts plus dataset-level facts decide which failure modes are
     applicable (FAILURE_MODE_PRECONDITIONS). Gated-out modes are recorded as
     "applicable": false and never asked about; applicable modes get a present / severity /
     confidence assessment, and an "other_failure_modes" list captures anything outside the
     fixed taxonomy.

Usage:
    python scripts/run_judge.py --agent vanilla --tasks titanic   # smoke test
    python scripts/run_judge.py                                   # judge every completed task
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_vanilla import load_registry, resolve_model_cfg, call_llm  # noqa: E402
from judge_spec import (FAILURE_MODES, STRUCTURE_SCHEMA, FAILURE_MODE_PRECONDITIONS,  # noqa: E402
                        EXTRACTION_PROMPT, GATED_JUDGE_PROMPT, JUDGE_MODEL, is_applicable)
from judge_common import pick_task_subset_for_agent, build_task_context, extract_json, AGENT_CONFIGS, MODEL_RUN_DIRS  # noqa: E402
from dsts_common import scoped_max_tokens  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "judge_results"


def build_extraction_prompt(schema: list, code: str) -> str:
    field_list = "\n".join(f"- **{f['name']}**: {f['description']}" for f in schema)
    field_json_template = ",\n".join(f'  "{f["name"]}": true | false' for f in schema)
    return EXTRACTION_PROMPT.format(
        code=code, field_list=field_list, field_json_template=field_json_template,
    )


def run_extraction(code: str, model_cfg: dict, schema: list) -> dict:
    prompt = build_extraction_prompt(schema, code)
    response, _ = call_llm([{"role": "user", "content": prompt}], model_cfg)
    raw = extract_json(response)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {}
    # default any missing/malformed field to False rather than crash downstream
    return {f["name"]: bool(result.get(f["name"], False)) for f in schema}


def run_one(agent: str, task_id: str, judge_model_cfg: dict,
            target_model: str = "gemini-3.6-flash") -> dict:
    ctx = build_task_context(agent, task_id, target_model)

    extracted = run_extraction(ctx["code"], judge_model_cfg, STRUCTURE_SCHEMA)

    # Gating uses code-level facts (extracted) PLUS dataset-level facts
    # (has_temporal -- same for every agent on this
    # task, computed once in build_task_context/dataset_level_facts). Kept
    # separate from `extracted`/stage_count, which stay code-only -- dataset
    # facts aren't pipeline stages the agent chose to include.
    gating_facts = {**ctx.get("dataset_facts", {}), **extracted}
    applicable_ids = [fid for fid in FAILURE_MODES if is_applicable(fid, gating_facts)]
    gated_out_ids = [fid for fid in FAILURE_MODES if fid not in applicable_ids]

    applicable_modes = {fid: FAILURE_MODES[fid] for fid in applicable_ids}
    failure_mode_list = "\n".join(f"**{fid} — {fm['name']}**\n{fm['definition']}\n" for fid, fm in applicable_modes.items())
    extracted_facts = "\n".join(f"- {k}: {v}" for k, v in extracted.items())

    prompt = GATED_JUDGE_PROMPT.format(**ctx, extracted_facts=extracted_facts, failure_mode_list=failure_mode_list)
    response, _ = call_llm([{"role": "user", "content": prompt}], judge_model_cfg)
    raw = extract_json(response)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"task": task_id, "parse_error": True, "raw_response": response}
        result["agent"] = agent
        result["target_model"] = target_model
        result["extracted_structure"] = extracted
        result["stage_count"] = sum(1 for v in extracted.values() if v)
        result["stage_count_max"] = len(extracted)
        return result

    # deterministically record the gated-out modes -- no LLM call wasted guessing about them
    result.setdefault("failure_modes", {})
    for fid in gated_out_ids:
        result["failure_modes"][fid] = {"applicable": False, "present": None}
    for fid in applicable_ids:
        if fid in result["failure_modes"]:
            result["failure_modes"][fid]["applicable"] = True

    result["agent"] = agent
    result["target_model"] = target_model
    result["extracted_structure"] = extracted
    # Flat count, not weighted -- same reasoning as dropping the weighted
    # overall_score elsewhere: a raw "how many of the N pipeline stages are
    # present" is directly comparable across agents without inventing
    # another scoring formula.
    result["stage_count"] = sum(1 for v in extracted.values() if v)
    result["stage_count_max"] = len(extracted)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=list(AGENT_CONFIGS), help="default: all 3")
    ap.add_argument("--model", choices=list(MODEL_RUN_DIRS), default="gemini-3.6-flash",
                     help="which agent-run's outputs to judge (NOT the judge's own LLM -- that's JUDGE_MODEL)")
    ap.add_argument("--tasks", nargs="*", help="override the auto-picked task list (applies to every agent selected)")
    ap.add_argument("--n-tasks", type=int, default=300,
                     help="cap per agent's OWN task list (not a shared intersection -- see pick_task_subset_for_agent); "
                          "default 300 exceeds every agent's real count, so it means 'all'")
    ap.add_argument("--limit", type=int, help="stop after N tasks (per agent), for quick testing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"{len(STRUCTURE_SCHEMA)} schema fields, {len(FAILURE_MODE_PRECONDITIONS)} gated failure modes "
          f"({len(FAILURE_MODES) - len(FAILURE_MODE_PRECONDITIONS)} of {len(FAILURE_MODES)} always applicable).\n")
    print(f"Target model (whose outputs are being judged): {args.model}\n")

    registry = load_registry()
    model_cfg = scoped_max_tokens(resolve_model_cfg(JUDGE_MODEL, registry), minimum=32768)

    agents = [args.agent] if args.agent else list(AGENT_CONFIGS)

    # gemini-3.6-flash keeps the original flat layout (621 already-cached
    # files, in-flight jobs writing here right now) -- new models get their
    # own subdir so nothing collides or needs migrating.
    for agent in agents:
        out_dir = OUT_ROOT / agent if args.model == "gemini-3.6-flash" else OUT_ROOT / agent / args.model
        out_dir.mkdir(parents=True, exist_ok=True)
        # Each agent judged against its OWN full completed-task list, not a
        # 3-way intersection -- judge everything, filter/compare later at
        # figure-build time (see pick_task_subset_for_agent's docstring).
        task_ids = args.tasks if args.tasks else pick_task_subset_for_agent(agent, n=args.n_tasks, target_model=args.model)
        ids = task_ids[: args.limit] if args.limit else task_ids
        print(f"[{agent}] {len(ids)} tasks to judge (model={args.model})")
        for i, task_id in enumerate(ids, 1):
            out_path = out_dir / f"{task_id}.json"
            if out_path.exists() and not args.force:
                run_result = MODEL_RUN_DIRS.get(args.model, {}).get(agent, Path("/nonexistent")) / task_id / "result.json"
                is_stale = run_result.exists() and run_result.stat().st_mtime > out_path.stat().st_mtime
                if not is_stale:
                    print(f"[{agent}][{i}/{len(ids)}] {task_id} [cached]")
                    continue
                print(f"[{agent}][{i}/{len(ids)}] {task_id} [stale cache, re-judging] ...", end=" ", flush=True)
            else:
                print(f"[{agent}][{i}/{len(ids)}] {task_id} ...", end=" ", flush=True)
            try:
                result = run_one(agent, task_id, model_cfg, target_model=args.model)
            except Exception as e:
                print(f"ERROR ({type(e).__name__}), skipping: {e}")
                continue
            out_path.write_text(json.dumps(result, indent=2))
            if result.get("parse_error"):
                print("PARSE ERROR")
            else:
                n_present = sum(1 for v in result.get("failure_modes", {}).values() if isinstance(v, dict) and v.get("present") is True)
                n_other = len(result.get("other_failure_modes", []))
                n_gated = sum(1 for v in result.get("failure_modes", {}).values() if isinstance(v, dict) and v.get("applicable") is False)
                print(f"done — {n_present} present, {n_other} other, {n_gated} gated-out-as-N/A, "
                      f"stage_count {result.get('stage_count')}/{result.get('stage_count_max')}")


if __name__ == "__main__":
    main()
