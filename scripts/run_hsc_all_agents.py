#!/usr/bin/env python3
"""Run all three agents (vanilla, AIDE, MLEvolve) on the full HSC task set
(analysis/hsc/hsc_tasks.json), one after another, via the combined HSC wrappers (run_*_hsc.py) so
home-credit-credit-risk-model-stability (multi_table) and open-problems-single-cell-perturbations
(multi_output) are both handled correctly alongside the other 6 single-table tasks.

Usage:
  python scripts/run_hsc_all_agents.py --model gemini/gemini-3.6-flash --steps 10 --variant few_shot_role
  python scripts/run_hsc_all_agents.py --model gemini/gemini-3.6-flash --steps 10 --variant few_shot_role --resume
  python scripts/run_hsc_all_agents.py --skip-vanilla         # only AIDE + MLEvolve
  python scripts/run_hsc_all_agents.py --aide-model gemini/gemini-3.6-flash --mlevolve-model gemini/gemini-3.6-flash

Note: this can take a long time — MLEvolve/AIDE each have up to a 2-hour
wall-clock cap per task, and home-credit-credit-risk-model-stability's
multi-table input is far larger than the other tasks. Consider running
this in the background (e.g. `nohup python scripts/run_hsc_all_agents.py ... &`)
and using --resume to pick back up if interrupted.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR    = PROJECT_ROOT / "scripts"
HSC_TASKS_FILE = PROJECT_ROOT / "analysis" / "hsc" / "hsc_tasks.json"


def load_hsc_task_ids() -> list[str]:
    data = json.loads(HSC_TASKS_FILE.read_text())
    return [t["id"] for t in data["tasks"]]


def run_and_stream(cmd: list[str], label: str) -> tuple[int, float]:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(" ".join(cmd), "\n", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n--- {label}: {status} in {elapsed / 60:.1f} min ---\n", flush=True)
    return proc.returncode, elapsed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="gemini/gemini-3.6-flash",
                     help="Model used for all three agents unless overridden per-agent "
                          "(default: %(default)s)")
    ap.add_argument("--vanilla-model", help="Override model for vanilla")
    ap.add_argument("--aide-model", help="Override model for AIDE")
    ap.add_argument("--mlevolve-model", help="Override model for MLEvolve")
    ap.add_argument("--steps", type=int, default=10,
                     help="AIDE/MLEvolve search steps (default: %(default)s)")
    ap.add_argument("--variant", default="few_shot_role",
                     help="Vanilla prompt variant: zero_shot | one_shot | few_shot | "
                          "few_shot_role (default: %(default)s)")
    ap.add_argument("--resume", action="store_true",
                     help="Skip already-completed tasks for all agents")
    ap.add_argument("--runs-dir", default="hsc_runs",
                     help="Shared output directory for all three agents' HSC runs — "
                          "run-name prefixes (vanilla__/aide__/mlevolve__) keep them "
                          "from colliding, matching the existing hsc_runs/ convention "
                          "(default: %(default)s)")
    ap.add_argument("--skip-vanilla", action="store_true")
    ap.add_argument("--skip-aide", action="store_true")
    ap.add_argument("--skip-mlevolve", action="store_true")
    args = ap.parse_args()

    task_ids = load_hsc_task_ids()
    print(f"HSC task set ({len(task_ids)} tasks): {', '.join(task_ids)}")
    print(f"Runs dir : {args.runs_dir}/\n")

    results = []

    if not args.skip_vanilla:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_vanilla_hsc.py"),
            "--model", args.vanilla_model or args.model,
            "--variant", args.variant,
            "--runs-dir", args.runs_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        for tid in task_ids:
            cmd += ["--tasks", tid]
        rc, dt = run_and_stream(cmd, "VANILLA")
        results.append(("vanilla", rc, dt))

    if not args.skip_aide:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_aide_hsc.py"),
            "--hsc",
            "--model", args.aide_model or args.model,
            "--steps", str(args.steps),
            "--runs-dir", args.runs_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        rc, dt = run_and_stream(cmd, "AIDE")
        results.append(("aide", rc, dt))

    if not args.skip_mlevolve:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_mlevolve_hsc.py"),
            "--hsc",
            "--model", args.mlevolve_model or args.model,
            "--steps", str(args.steps),
            "--runs-dir", args.runs_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        rc, dt = run_and_stream(cmd, "MLEVOLVE")
        results.append(("mlevolve", rc, dt))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, rc, dt in results:
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  {name:10s} {status:22s} {dt / 60:6.1f} min")
    total = sum(dt for _, _, dt in results)
    print(f"\n  total: {total / 60:.1f} min")


if __name__ == "__main__":
    main()
