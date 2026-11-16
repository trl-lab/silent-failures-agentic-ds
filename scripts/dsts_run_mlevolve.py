"""
Adapts MLEvolve to run against the DSTS unit-test suite. Reuses
MLEvolve's real bootstrap mechanism from run_mlevolve.py unchanged
(build_mlevolve_config, _bootstrap_script, MLEVOLVE_PYTHON) -- only the
desc_file content and data_dir contents are DSTS-specific instead of our
own benchmark's task.json. Same data_dir/desc_file pattern as
dsts_run_aide.py (see there for why this differs from vanilla's inline
prompt).

Usage:
    python scripts/dsts_run_mlevolve.py --workload TF01_goal_imbalance --model gemini/gemini-3.6-flash --steps 5

Then judge with the test suite's own, unmodified judge.py -- same as dsts_run_vanilla.py.
"""
import argparse
import os
import subprocess
import sys
import threading
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_vanilla import load_registry, resolve_model_cfg, model_slug  # noqa: E402
from run_mlevolve import build_mlevolve_config, _bootstrap_script, MLEVOLVE_PYTHON  # noqa: E402
from dsts_common import load_workload, build_desc_text, data_file_paths, write_judge_output, extract_mlevolve_best_metric, execute_and_extract  # noqa: E402

DEFAULT_TIMEOUT = 3600  # TF0X tasks are single-recipe, much smaller than our own 7200s benchmark cap


def run_task(task: dict, workload: str, model_cfg: dict, steps: int, out_root: Path, timeout: int, run_label: str | None = None) -> dict:
    task_id = task["id"]
    dir_name = run_label or task_id
    task_dir = out_root / dir_name
    task_dir.mkdir(parents=True, exist_ok=True)

    data_dir = task_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for src in data_file_paths(task, workload):
        dst = data_dir / src.name
        if not dst.exists():
            dst.symlink_to(src)

    input_dir = task_dir / "input"
    if not input_dir.exists():
        input_dir.symlink_to(data_dir.resolve())

    desc_file = task_dir / "task_desc.md"
    desc_file.write_text(build_desc_text(task, workload), encoding="utf-8")

    log_dir = task_dir / "mlevolve_logs"
    workspace_dir = task_dir / "mlevolve_workspace"
    log_dir.mkdir(exist_ok=True)
    workspace_dir.mkdir(exist_ok=True)

    cfg_dict = build_mlevolve_config(data_dir, desc_file, dir_name, log_dir, workspace_dir, model_cfg, steps)
    config_path = task_dir / "mlevolve_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)

    bootstrap = task_dir / "_mlevolve_bootstrap.py"
    bootstrap.write_text(_bootstrap_script(config_path), encoding="utf-8")

    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    if model_cfg.get("api_key"):
        env["GOOGLE_API_KEY"] = model_cfg["api_key"]

    model_name = model_cfg["model_name"]
    print(f"  Running MLEvolve ({steps} steps, model={model_name}) ...", flush=True)

    output_lines = []
    proc = subprocess.Popen(
        [str(MLEVOLVE_PYTHON), str(bootstrap)],
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

    best_solutions = sorted(log_dir.glob("**/best_solution.py"))
    code = best_solutions[-1].read_text(encoding="utf-8") if best_solutions else None

    combined_stdout = stdout
    exec_returncode = returncode
    exec_stderr = ""
    if code:
        _, exec_stdout, exec_stderr, exec_returncode = execute_and_extract(code, task_dir, timeout=900)
        combined_stdout = stdout + "\n" + exec_stdout

    result = write_judge_output(task_dir, code, combined_stdout, exec_stderr, exec_returncode, fallback_extractor=extract_mlevolve_best_metric)
    result["task_id"] = dir_name
    result["base_task_id"] = task_id
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=5, help="MLEvolve search steps (default: 5, lighter than our 10-step benchmark)")
    ap.add_argument("--task-id", help="run a single task id instead of the whole workload")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "dsts_runs" / "mlevolve"))
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each task this many independent times (e.g. 5, for a real pass-rate stat instead of n=1 per task)")
    ap.add_argument("--rep", type=int,
                    help="run only this one specific repeat number (1-indexed) instead of the full --repeats range -- "
                         "lets a caller submit one rep per Slurm job for real N-way parallelism (85 tasks x reps as "
                         "85 separate jobs) instead of one job looping through all of them sequentially. Directory "
                         "label is always '<task_id>_rep<N>' when this is set, matching what a --repeats run would "
                         "have produced for that slot, so results land in the same place either way.")
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

    rep_range = [args.rep] if args.rep is not None else list(range(1, args.repeats + 1))
    total_runs = len(tasks) * len(rep_range)
    reps_desc = f"rep {args.rep}" if args.rep is not None else f"{args.repeats} repeat(s)"
    print(f"Running {len(tasks)} task(s) x {reps_desc} = {total_runs} run(s) from {args.workload} with {args.model} ({args.steps} steps)")
    print(f"Output: {out_root}\n")

    results = []
    run_i = 0
    for task in tasks:
        for rep in rep_range:
            run_i += 1
            label = task["id"] if (args.repeats == 1 and args.rep is None) else f"{task['id']}_rep{rep}"
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
