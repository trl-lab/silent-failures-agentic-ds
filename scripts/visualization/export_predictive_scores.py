#!/usr/bin/env python3
"""Export every real predictive score across all (agent x model) run directories into one tidy CSV.

Usage:
    python scripts/visualization/export_predictive_scores.py
    python scripts/visualization/export_predictive_scores.py --out /tmp/scores.csv --min-tasks 100
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_vanilla import score_predictions  # noqa: E402

AGENT_BASES = {"vanilla": "runs", "aide": "aide_runs", "mlevolve": "mlevolve_runs"}
DATASET_ROOT = PROJECT_ROOT / "datasets"


def find_task_dir(task_id: str, source_hint: str | None):
    """datasets/<source>/<task_id>, preferring the source recorded in result.json."""
    if source_hint:
        p = DATASET_ROOT / source_hint / task_id
        if p.exists():
            return p
    for src_dir in sorted(DATASET_ROOT.iterdir()):
        if src_dir.is_dir() and (src_dir / task_id).exists():
            return src_dir / task_id
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "predictive_scores.csv"))
    ap.add_argument("--min-tasks", type=int, default=100,
                    help="skip run dirs with fewer tasks than this (filters out abandoned/pilot runs)")
    args = ap.parse_args()

    rows = []
    for agent, base in AGENT_BASES.items():
        base_dir = PROJECT_ROOT / base
        if not base_dir.exists():
            continue
        for run_dir in sorted(base_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            task_dirs = [d for d in sorted(run_dir.iterdir()) if d.is_dir()]
            if len(task_dirs) < args.min_tasks:
                continue
            # run dir name is <agent>__<model-slug>__<variant/steps>
            parts = run_dir.name.split("__")
            model = parts[1] if len(parts) > 1 else "?"
            variant = parts[2] if len(parts) > 2 else ""
            print(f"[{agent}] {run_dir.name}: {len(task_dirs)} tasks", flush=True)

            n_stored = n_rescored = n_failed = 0
            for td in task_dirs:
                rj = td / "result.json"
                res = {}
                if rj.exists():
                    try:
                        res = json.loads(rj.read_text())
                    except Exception:
                        res = {}
                task_id = res.get("task_id") or td.name
                source = res.get("source", "")
                metric = res.get("metric", "")
                score = res.get("score")
                status, error = "stored", ""

                if score is None:
                    if (td / "predictions.csv").exists():
                        task_dir = find_task_dir(task_id, source)
                        tc = res.get("target_column")
                        if task_dir and not tc:
                            try:
                                tc = json.loads((task_dir / "task.json").read_text()).get("target_column")
                            except Exception:
                                tc = None
                        if task_dir and tc:
                            try:
                                r = score_predictions(task_dir, td, tc)
                                if r.get("success") and r.get("score") is not None:
                                    score = r["score"]
                                    metric = r.get("metric", metric)
                                    status = "rescored"
                                else:
                                    status, error = "failed", r.get("error", "rescore returned no score")
                            except Exception as exc:
                                status, error = "failed", f"{type(exc).__name__}: {exc}"
                        else:
                            status, error = "failed", "could not locate task dir / target column"
                    else:
                        status = "failed"
                        error = res.get("error", "no result.json") or "no score"

                if status == "stored":
                    n_stored += 1
                elif status == "rescored":
                    n_rescored += 1
                else:
                    n_failed += 1

                rows.append({
                    "agent": agent, "model": model, "variant": variant,
                    "run_dir": run_dir.name, "task_id": task_id, "source": source,
                    "metric": metric,
                    "higher_is_better": res.get("higher_is_better", ""),
                    "score": score if score is not None else "",
                    "status": status, "error": (error or "")[:200],
                })
            print(f"    stored={n_stored}  rescored={n_rescored}  failed={n_failed}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["agent", "model", "variant", "run_dir", "task_id", "source",
              "metric", "higher_is_better", "score", "status", "error"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    tot = len(rows)
    ok = sum(1 for r in rows if r["score"] != "")
    resc = sum(1 for r in rows if r["status"] == "rescored")
    print(f"\nWrote {out}")
    print(f"  {tot} rows, {ok} with a score ({resc} of them recovered by re-scoring), {tot-ok} failed")


if __name__ == "__main__":
    main()
