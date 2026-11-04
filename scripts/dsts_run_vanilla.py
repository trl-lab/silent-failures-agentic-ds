"""Adapts the vanilla baseline to run against the DSTS unit-test suite. See dsts_common.py for the
shared pipeline design notes (workload format, why judge.py needs no changes, output file
conventions).

Usage:
    python scripts/dsts_run_vanilla.py --workload TF01_goal_imbalance --model gemini/gemini-3.6-flash
    python scripts/dsts_run_vanilla.py --workload TF01_goal_imbalance --model gemini/gemini-3.6-flash --task-id TF01_goal_imbalance_01
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_vanilla import load_registry, resolve_model_cfg, call_llm, extract_code, execute_code, model_slug  # noqa: E402
from dsts_common import load_workload, build_desc_text, write_judge_output, scoped_max_tokens  # noqa: E402


def run_task(task: dict, workload: str, model_cfg: dict, out_root: Path, timeout: int, run_label: str | None = None) -> dict:
    task_id = task["id"]
    dir_name = run_label or task_id
    task_dir = out_root / dir_name
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        build_desc_text(task, workload)
        + "\nWrite a complete, self-contained Python script. Output only the "
          "Python code, no explanation or markdown fences."
    )
    (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    MAX_ATTEMPTS = 4
    response = None
    last_exc = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response, _ = call_llm([{"role": "user", "content": prompt}], model_cfg)
        except Exception as exc:
            last_exc = exc
            continue
        if response:
            break
    if not response:
        msg = (f"LLM error: {last_exc}" if last_exc and not response else
               f"LLM returned empty/no content after {MAX_ATTEMPTS} attempts "
               "-- likely exhausted max_tokens on reasoning, or the Gemini "
               "MALFORMED_FUNCTION_CALL issue (see comment above)")
        (task_dir / "errors-0.txt").write_text(msg, encoding="utf-8")
        return {"task_id": task_id, "success": False, "error": msg}

    code = extract_code(response)
    stdout, stderr, returncode = execute_code(code, task_dir, timeout)
    result = write_judge_output(task_dir, code, stdout, stderr, returncode)
    result["task_id"] = dir_name
    result["base_task_id"] = task_id
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="e.g. TF01_goal_imbalance")
    ap.add_argument("--model", required=True, help="must be in scripts/config/model_registry.json")
    ap.add_argument("--task-id", help="run a single task id instead of the whole workload")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "dsts_runs" / "vanilla"))
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each task this many independent times (e.g. 5, for a real pass-rate stat instead of n=1 per task)")
    ap.add_argument("--force", action="store_true", help="re-run even if output already exists (default: skip)")
    args = ap.parse_args()

    registry = load_registry()
    model_cfg = scoped_max_tokens(resolve_model_cfg(args.model, registry))
    out_root = Path(args.out_root) / model_slug(args.model)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = load_workload(args.workload)
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            sys.exit(f"task id not found in workload: {args.task_id}")

    total_runs = len(tasks) * args.repeats
    print(f"Running {len(tasks)} task(s) x {args.repeats} repeat(s) = {total_runs} run(s) from {args.workload} with {args.model}")
    print(f"Output: {out_root}\n")

    results = []
    run_i = 0
    for task in tasks:
        for rep in range(1, args.repeats + 1):
            run_i += 1
            label = task["id"] if args.repeats == 1 else f"{task['id']}_rep{rep}"
            out_path = out_root / label / "generated_code.py"
            if out_path.exists() and not args.force:
                print(f"[{run_i}/{total_runs}] {label} [cached]")
                continue
            print(f"[{run_i}/{total_runs}] {label}")
            try:
                r = run_task(task, args.workload, model_cfg, out_root, args.timeout, run_label=label)
            except Exception as exc:
                # Single-task crash shouldn't take the whole batch down --
                # same defense-in-depth as run_vanilla.py's main() loop.
                # Rerunning with the same command will retry just this task
                # since it never got generated_code.py written.
                print(f"  UNEXPECTED ERROR, skipping: {type(exc).__name__}: {exc}", flush=True)
                r = {"task_id": label, "success": False, "error": f"unexpected error: {type(exc).__name__}: {exc}"}
            results.append(r)
            status = "OK" if r["success"] else "FAILED"
            print(f"  {status}: {r.get('answer') or r.get('error', 'no answer extracted')}\n")

    n_ok = sum(1 for r in results if r["success"])
    print(f"Done: {n_ok}/{len(results)} newly-run tasks produced a usable answer ({total_runs - len(results)} were cached).")


if __name__ == "__main__":
    main()
