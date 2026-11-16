"""Adapts AIDE to run against the DSTS unit-test suite. Reuses AIDE's real bootstrap mechanism from
run_aide.py unchanged (_bootstrap_script, AIDE_PYTHON, WALL_TIMEOUT) -- only the desc_file content
and data_dir contents are DSTS-specific instead of our own benchmark's task.json.

Usage:
    python scripts/dsts_run_aide.py --workload TF01_goal_imbalance --model gemini/gemini-3.6-flash --steps 5
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_vanilla import load_registry, resolve_model_cfg, model_slug  # noqa: E402
from run_aide import _bootstrap_script, AIDE_PYTHON  # noqa: E402
from dsts_common import load_workload, build_desc_text, data_file_paths, write_judge_output, extract_aide_best_metric, execute_and_extract  # noqa: E402

DEFAULT_TIMEOUT = 3600  # TF0X tasks are single-recipe, much smaller than our own 7200s benchmark cap


def run_task(task: dict, workload: str, model_cfg: dict, steps: int, out_root: Path, timeout: int, run_label: str | None = None) -> dict:
    task_id = task["id"]
    dir_name = run_label or task_id
    task_dir = out_root / dir_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # AIDE browses data_dir itself rather than taking explicit paths --
    # symlink the real TF0X CSVs in rather than copying.
    data_dir = task_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for src in data_file_paths(task, workload):
        dst = data_dir / src.name
        if not dst.exists():
            dst.symlink_to(src)

    desc_file = task_dir / "task_desc.md"
    desc_file.write_text(build_desc_text(task, workload), encoding="utf-8")

    aide_log_dir = task_dir / "aide_logs"
    aide_work_dir = task_dir / "aide_workspace"
    aide_log_dir.mkdir(exist_ok=True)
    aide_work_dir.mkdir(exist_ok=True)

    model_name = model_cfg["model_name"]
    bootstrap = task_dir / "_aide_bootstrap.py"
    bootstrap.write_text(
        _bootstrap_script(data_dir, desc_file, model_name, steps, aide_log_dir, aide_work_dir,
                          max_tokens=model_cfg.get("max_tokens") or 16384,
                          request_timeout_s=model_cfg.get("request_timeout_s")),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = model_cfg.get("base_url", "http://localhost:11434/v1")
    env["OPENAI_API_KEY"] = model_cfg.get("api_key", "EMPTY")

    print(f"  Running AIDE ({steps} steps, model={model_name}) ...", flush=True)

    output_lines = []
    proc = subprocess.Popen(
        [str(AIDE_PYTHON), str(bootstrap)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )

    def _drain():
        for line in proc.stdout:
            output_lines.append(line)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        output_lines.append(f"\nTIMEOUT after {timeout}s\n")
        returncode = -1
    reader.join(timeout=5)

    stdout = "".join(output_lines)

    best_solutions = sorted(aide_log_dir.glob("**/best_solution.py"))
    code = best_solutions[-1].read_text(encoding="utf-8") if best_solutions else None

    combined_stdout = stdout
    exec_returncode = returncode
    exec_stderr = ""
    if code:
        _, exec_stdout, exec_stderr, exec_returncode = execute_and_extract(code, task_dir, timeout=900)
        combined_stdout = stdout + "\n" + exec_stdout

    result = write_judge_output(task_dir, code, combined_stdout, exec_stderr, exec_returncode, fallback_extractor=extract_aide_best_metric)
    result["task_id"] = dir_name
    result["base_task_id"] = task_id
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=5, help="AIDE search steps (default: 5, lighter than our 10-step benchmark)")
    ap.add_argument("--task-id", help="run a single task id instead of the whole workload")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "dsts_runs" / "aide"))
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each task this many independent times (e.g. 5, for a real pass-rate stat instead of n=1 per task)")
    ap.add_argument("--force", action="store_true", help="re-run even if output already exists (default: skip)")
    args = ap.parse_args()

    registry = load_registry()
    model_cfg = resolve_model_cfg(args.model, registry)
    out_root = Path(args.out_root) / model_slug(args.model)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = load_workload(args.workload)
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            sys.exit(f"task id not found in workload: {args.task_id}")

    total_runs = len(tasks) * args.repeats
    print(f"Running {len(tasks)} task(s) x {args.repeats} repeat(s) = {total_runs} run(s) from {args.workload} with {args.model} ({args.steps} steps)")
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
            r = run_task(task, args.workload, model_cfg, args.steps, out_root, args.timeout, run_label=label)
            results.append(r)
            status = "OK" if r["success"] else "FAILED"
            print(f"  {status}: {r.get('answer') or r.get('error', 'no answer extracted')}\n")

    n_ok = sum(1 for r in results if r["success"])
    print(f"Done: {n_ok}/{len(results)} newly-run tasks produced a usable answer ({total_runs - len(results)} were cached).")


if __name__ == "__main__":
    main()
