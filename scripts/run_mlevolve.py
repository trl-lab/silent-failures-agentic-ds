#!/usr/bin/env python3
"""MLEvolve agent runner for the DS agent failure analysis benchmark.

Run directory: mlevolve_runs/mlevolve__<model-slug>__<steps>steps/<task_id>/

Usage:
  python scripts/run_mlevolve.py --task house-prices-advanced-regression-techniques
  python scripts/run_mlevolve.py --tasks cat-in-the-dat --tasks diamonds
  python scripts/run_mlevolve.py --hsc
  python scripts/run_mlevolve.py --model gemini/gemini-3-pro-preview --steps 10
  python scripts/run_mlevolve.py --resume
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CONFIG_DIR      = PROJECT_ROOT / "scripts" / "config"
# DS_DATASETS_DIR lets memorization-alteration runs point at datasets_altered/ (default: unchanged)
DATASETS_ROOT = PROJECT_ROOT / os.environ.get("DS_DATASETS_DIR", "datasets")
MLEVOLVE_ROOT   = Path(os.environ.get("MLEVOLVE_ROOT", PROJECT_ROOT / "external" / "agents" / "MLEvolve"))  # clone of the MLEvolve repo
# sys.executable, not a hardcoded path -- always resolves to whatever
# interpreter is actually running this script, so it works regardless of
# where conda/miniconda happens to be installed on a given machine (local
# Mac dev vs. Snellius's module-provided Miniconda, different install paths).
MLEVOLVE_PYTHON = sys.executable
HSC_TASKS_FILE  = PROJECT_ROOT / "analysis" / "hsc" / "hsc_tasks.json"
SOURCES         = ["strable", "openml", "uci", "kaggle", "mlebench"] + (["altered"] if os.environ.get("DS_DATASETS_DIR") else [])

DEFAULT_MODEL = "gemini/gemini-3-pro-preview"
DEFAULT_STEPS = 10
WALL_TIMEOUT  = 7200  # per-task wall-clock cap (seconds), default for fast hosted APIs


def resolve_wall_timeout_s(model_cfg: dict) -> int:
    """Per-model override (model_registry.json's time_cap_hours.mlevolve) if present, else the module
    default. Reusing that same registry value here instead of inventing a second, easily-drifting
    config key.
    """
    override_h = (model_cfg.get("time_cap_hours") or {}).get("mlevolve")
    return int(override_h * 3600) if override_h else WALL_TIMEOUT

DS_PERSONA = (
    "You are an experienced data scientist working on a predictive modelling competition. "
    "You follow best practices: always create a validation split to estimate generalisation "
    "performance, clean data before modelling, choose metrics appropriate for the task, "
    "and avoid any form of data leakage. Write clean, well-structured Python code."
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    return json.loads((CONFIG_DIR / "model_registry.json").read_text())


def resolve_model_cfg(model_name: str, registry: dict) -> dict:
    if model_name not in registry:
        raise ValueError(
            f"Model '{model_name}' not in model_registry.json. "
            f"Available: {list(registry)}"
        )
    cfg = dict(registry[model_name])
    if "model_name" not in cfg:
        cfg["model_name"] = model_name
    if "api_key_env" in cfg:
        env_val = os.environ.get(cfg["api_key_env"], "")
        if not env_val:
            raise EnvironmentError(
                f"Env var {cfg['api_key_env']} not set for model {model_name}"
            )
        cfg["api_key"] = env_val
    return cfg


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "-").replace("/", "-")


def load_hsc_task_ids() -> list[str]:
    data = json.loads(HSC_TASKS_FILE.read_text())
    return [t["id"] for t in data["tasks"]]


# ---------------------------------------------------------------------------
# Task description builder
# ---------------------------------------------------------------------------

def build_task_desc(task_dir: Path, task_meta: dict) -> str:
    desc = ""
    if (task_dir / "description.md").exists():
        desc = (task_dir / "description.md").read_text(encoding="utf-8").strip()

    target    = task_meta["target_column"]
    metric    = task_meta.get("metric", "rmse")
    higher    = task_meta.get("metric_higher_is_better", True)
    direction = "higher is better" if higher else "lower is better"

    lines = [DS_PERSONA, ""]

    if desc:
        lines += [desc, ""]

    lines += [
        f"**Target column**: `{target}`",
        f"**Evaluation metric**: `{metric}` ({direction})",
        "",
        "**Data**: Train data is at `./input/train.parquet` and test data at "
        "`./input/test.parquet`. Read them with `pd.read_parquet()`.",
        "",
        "**Installed package versions** (use APIs compatible with these):",
        "- `lightgbm==4.6.0`: `early_stopping_rounds` and `verbose_eval` were removed "
        "from `lgb.train()`. Use callbacks: "
        "`callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`",
        "- `scikit-learn==1.5.0`, `pandas==2.1.4`",
        "",
        "**Output**: Write test predictions to `./submission/submission.csv` with exactly "
        f"two columns: `id` (integer row index 0-based) and `{target}` (predicted values).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MLEvolve config builder
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (modifies base in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def build_mlevolve_config(
    data_dir: Path,
    desc_file: Path,
    task_id: str,
    log_dir: Path,
    workspace_dir: Path,
    model_cfg: dict,
    steps: int,
) -> dict:
    """Build a per-run MLEvolve config by merging our overrides into the base config.yaml."""
    with open(MLEVOLVE_ROOT / "config" / "config.yaml") as f:
        base = yaml.safe_load(f)

    model_name = model_cfg["model_name"]
    base_url   = model_cfg.get("base_url", "") or "https://generativelanguage.googleapis.com/"
    api_key    = model_cfg.get("api_key", "")
    temp       = float(model_cfg.get("temperature", 0.8))

    overrides = {
        "data_dir":        str(data_dir),
        "desc_file":       str(desc_file),
        "exp_id":          task_id,
        "exp_name":        task_id,
        "log_dir":         str(log_dir),
        "workspace_dir":   str(workspace_dir),
        "preprocess_data": False,
        # copy_data=True: MLEvolve copies train/test parquet into workspace/input/
        "copy_data":       True,
        "exec": {
            "timeout": 3600,
        },
        "agent": {
            "steps":          steps,
            # Give ~6 min per step, cap at 1 hour
            "time_limit":     min(steps * 360, 3600),
            "initial_drafts": 2,
            "seed":           42,
            "code": {
                "model":    model_name,
                "base_url": base_url,
                "api_key":  api_key,
                "temp":     temp,
            },
            "feedback": {
                "model":    model_name,
                "base_url": base_url,
                "api_key":  api_key,
                "temp":     temp,
            },
            # Disable memory: requires BAAI/bge embedding model download
            "use_global_memory": False,
            # Disable fusion: only useful at 6+ hours, not at 10 steps
            "use_fusion": False,
            "search": {
                "parallel_search_num": 1,
                "num_gpus":            0,
                "num_drafts":          3,
                "num_bugs":            1,
                "num_improves":        2,
                "max_debug_depth":     10,
                # Push fusion time threshold way out so it never triggers
                "fusion_min_time_hours": 999,
            },
        },
        "coldstart": {
            # Disable: depends on MLE-Bench competition tags, not our tasks
            "use_coldstart": False,
        },
    }

    return _deep_merge(base, overrides)


# ---------------------------------------------------------------------------
# Bootstrap script (runs inside ds_failures env)
# ---------------------------------------------------------------------------

def _bootstrap_script(config_path: Path) -> str:
    """
    Generate a self-contained Python script that monkey-patches MLEvolve's
    load_cfg() to point at our per-run config, then calls run.run().

    Patching must happen BEFORE `import run` so that run.py's top-level
    `from config import load_cfg` binds to our patched version.
    """
    return textwrap.dedent(f"""\
        import sys
        from pathlib import Path

        # Add MLEvolve to path
        sys.path.insert(0, {str(MLEVOLVE_ROOT)!r})

        # Patch config.load_cfg BEFORE importing run.py, so that run.py's
        # `from config import load_cfg` picks up the patched function.
        import config as _mlevolve_config
        _orig_load_cfg = _mlevolve_config.load_cfg

        def _patched_load_cfg(*args, **kwargs):
            return _orig_load_cfg(path=Path({str(config_path)!r}))

        _mlevolve_config.load_cfg = _patched_load_cfg

        # Now import and run MLEvolve's main loop
        import run as _mlevolve_run
        _mlevolve_run.run()
    """)


# ---------------------------------------------------------------------------
# Submission discovery and scoring
# ---------------------------------------------------------------------------

def find_best_submission(workspace_dir: Path, task_dir: Path, target_col: str) -> Path | None:
    """Find the best submission CSV from MLEvolve's workspace."""
    # MLEvolve uses a timestamped sub-directory; find the most recent one
    run_subdirs = sorted(workspace_dir.glob("*/"), key=lambda p: p.name)
    if not run_subdirs:
        return None
    run_dir = run_subdirs[-1]

    # Prefer MLEvolve's own best_submission
    best_sub = run_dir / "best_submission" / "submission.csv"
    if best_sub.exists():
        return best_sub

    candidates = sorted(run_dir.glob("submission/submission_*.csv"))
    if not candidates:
        return None

    labels_path = task_dir / "evaluation" / "labels.parquet"
    metric_path = task_dir / "evaluation" / "metric.py"
    if not labels_path.exists() or not metric_path.exists():
        return candidates[-1]

    try:
        spec   = importlib.util.spec_from_file_location("metric", metric_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        labels_df  = pd.read_parquet(labels_path)
        label_cols = [c for c in labels_df.columns if c != "id"]
    except Exception:
        return candidates[-1]

    best_path  = None
    best_score = None
    maximize   = True

    for csv_path in candidates:
        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df.columns) < 2:
                continue

            pred_col = next(
                (c for c in df.columns if c.lower() == target_col.lower()),
                df.columns[1],
            )
            df = df.rename(columns={pred_col: "_pred"})
            if "id" not in df.columns:
                df.insert(0, "id", range(len(df)))

            merged = labels_df[["id"] + label_cols].merge(
                df[["id", "_pred"]], on="id", how="inner"
            )
            if merged.empty:
                continue

            y_true = merged[label_cols[0]] if len(label_cols) == 1 else merged[label_cols]
            y_pred = merged["_pred"]
            result = module.evaluate(y_true, y_pred)

            score    = result.get("score")
            maximize = result.get("higher_is_better", True)
            if score is None:
                continue

            if best_score is None:
                best_score, best_path = score, csv_path
            elif maximize and score > best_score:
                best_score, best_path = score, csv_path
            elif not maximize and score < best_score:
                best_score, best_path = score, csv_path

        except Exception:
            continue

    return best_path or candidates[-1]


def post_process_submission(submission_csv: Path, target_col: str,
                             task_dir: Path | None = None) -> pd.DataFrame | None:
    """Normalise submission CSV to our standard predictions.csv format.

    For multi-output tasks (task.json "multi_output": true), expects id +
    every label column from evaluation/labels.parquet instead of a single
    target_col rename."""
    try:
        df = pd.read_csv(submission_csv)
    except Exception as e:
        print(f"  Could not read submission.csv: {e}", flush=True)
        return None

    if df.empty or len(df.columns) < 2:
        print(f"  submission.csv has unexpected shape {df.shape}", flush=True)
        return None

    if task_dir is not None:
        task_meta = json.loads((task_dir / "task.json").read_text())
        if task_meta.get("multi_output"):
            labels_df = pd.read_parquet(task_dir / "evaluation" / "labels.parquet")
            expected_cols = [c for c in labels_df.columns if c != "id"]
            if "id" not in df.columns:
                df.insert(0, "id", range(len(df)))
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"  submission.csv missing {len(missing)} expected columns, "
                      f"e.g. {missing[:5]}", flush=True)
                return None
            return df[["id"] + expected_cols]

    pred_col = None
    for c in df.columns:
        if c.lower() == target_col.lower():
            pred_col = c
            break
    if pred_col is None:
        for c in df.columns[1:]:
            if c.lower() in ("prediction", "pred", "score", "target", "label", "output"):
                pred_col = c
                break
    if pred_col is None:
        pred_col = df.columns[1]

    return pd.DataFrame({"id": range(len(df)), target_col: df[pred_col].values})


def score_predictions(task_dir: Path, run_dir: Path, target_col: str) -> dict:
    pred_path   = run_dir / "predictions.csv"
    labels_path = task_dir / "evaluation" / "labels.parquet"
    metric_path = task_dir / "evaluation" / "metric.py"

    if not pred_path.exists():
        return {"success": False, "error": "predictions.csv not produced"}

    try:
        labels_df = pd.read_parquet(labels_path)
        preds_df  = pd.read_csv(pred_path)
        label_cols = [c for c in labels_df.columns if c != "id"]

        task_meta = json.loads((task_dir / "task.json").read_text())
        if task_meta.get("multi_output"):
            missing = [c for c in label_cols if c not in preds_df.columns]
            if missing:
                return {"success": False,
                        "error": f"Missing {len(missing)} target columns in predictions.csv, "
                                 f"e.g. {missing[:5]}"}
            merged = labels_df.merge(preds_df, on="id", how="inner", suffixes=("_true", "_pred"))
            if len(merged) == 0:
                return {"success": False, "error": "No rows matched on 'id'"}
            y_true = merged[[c + "_true" for c in label_cols]]
            y_pred = merged[[c + "_pred" for c in label_cols]]

            spec = importlib.util.spec_from_file_location("metric", metric_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.evaluate(y_true, y_pred)
            result["success"]     = True
            result["n_predicted"] = len(merged)
            return result

        if target_col not in preds_df.columns:
            return {
                "success": False,
                "error": (
                    f"Column '{target_col}' missing from predictions.csv. "
                    f"Found: {list(preds_df.columns)}"
                ),
            }

        preds_slim = preds_df[["id", target_col]].rename(columns={target_col: "_pred"})
        merged = labels_df[["id"] + label_cols].merge(preds_slim, on="id", how="inner")

        if len(merged) == 0:
            return {"success": False, "error": "No rows matched on 'id'"}

        y_true = merged[label_cols] if len(label_cols) > 1 else merged[label_cols[0]]
        y_pred = merged["_pred"]

        spec   = importlib.util.spec_from_file_location("metric", metric_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.evaluate(y_true, y_pred)
        result["success"]     = True
        result["n_predicted"] = len(merged)
        return result

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    source: str,
    model_cfg: dict,
    steps: int,
    run_root: Path,
) -> dict:
    task_dir = DATASETS_ROOT / source / task_id
    if not (task_dir / "task.json").exists():
        return {
            "task_id": task_id, "source": source,
            "success": False, "error": "task.json not found",
        }

    task_meta  = json.loads((task_dir / "task.json").read_text())
    target_col = task_meta["target_column"]
    data_dir   = (task_dir / "data").resolve()

    run_dir = run_root / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir       = run_dir / "mlevolve_logs"
    workspace_dir = run_dir / "mlevolve_workspace"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    log_dir.mkdir(exist_ok=True)
    workspace_dir.mkdir(exist_ok=True)
    # Second half of the same fix (see run_aide.py's identical comment):
    # score_predictions() reads run_dir/"predictions.csv" directly, and a
    # previous invocation's copy of submission.csv/predictions.csv at the
    # run_dir TOP LEVEL survives the subdirectory wipe untouched otherwise.
    for _stale_name in ("submission.csv", "predictions.csv"):
        _stale_f = run_dir / _stale_name
        if _stale_f.exists():
            _stale_f.unlink()

    # Write task description (desc_file for MLEvolve's load_task_desc)
    task_desc = build_task_desc(task_dir, task_meta)
    desc_file = run_dir / "task_desc.md"
    desc_file.write_text(task_desc, encoding="utf-8")

    # Build per-run config and write YAML
    cfg_dict    = build_mlevolve_config(
        data_dir, desc_file, task_id, log_dir, workspace_dir, model_cfg, steps
    )
    config_path = run_dir / "mlevolve_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)

    # Write bootstrap
    bootstrap = run_dir / "_mlevolve_bootstrap.py"
    bootstrap.write_text(_bootstrap_script(config_path), encoding="utf-8")

    # Env: pass API key; suppress Rich TUI
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"]     = "dumb"
    if model_cfg.get("api_key"):
        env["GOOGLE_API_KEY"] = model_cfg["api_key"]

    model_name = model_cfg["model_name"]
    wall_timeout_s = resolve_wall_timeout_s(model_cfg)
    print(f"  Running MLEvolve ({steps} steps, model={model_name}) ...", flush=True)
    t_start = time.time()

    output_lines: list[str] = []
    returncode = -1
    proc = subprocess.Popen(
        [str(MLEVOLVE_PYTHON), str(bootstrap)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )

    def _drain_stdout():
        for line in proc.stdout:
            print(f"    {line}", end="", flush=True)
            ts = datetime.now(timezone.utc).isoformat()
            output_lines.append(f"[{ts}] {line}" if line.strip() else line)

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=wall_timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        output_lines.append(f"\nTIMEOUT after {wall_timeout_s}s\n")
        returncode = -1
    reader.join(timeout=5)

    wall_time = round(time.time() - t_start, 1)
    (run_dir / "mlevolve_stdout.txt").write_text("".join(output_lines), encoding="utf-8")

    if returncode != 0:
        print(f"  MLEvolve exited with code {returncode}", flush=True)

    # Find best submission, score it
    best_sub = find_best_submission(workspace_dir, task_dir, target_col)
    if best_sub:
        shutil.copy(best_sub, run_dir / "submission.csv")
        preds_df = post_process_submission(best_sub, target_col, task_dir)
        if preds_df is not None:
            preds_df.to_csv(run_dir / "predictions.csv", index=False)
        print(f"  Best submission: {best_sub.name}", flush=True)
    else:
        print("  No submission.csv found in MLEvolve workspace.", flush=True)

    score = score_predictions(task_dir, run_dir, target_col)
    if score.get("success"):
        print(f"  score={score.get('score'):.4f} ({task_meta.get('metric')})", flush=True)
    else:
        print(f"  FAILED: {score.get('error', '')[:80]}", flush=True)

    result = {
        "task_id":        task_id,
        "source":         source,
        "agent":          "mlevolve",
        "model":          model_name,
        "mlevolve_steps": steps,
        "metric":         task_meta.get("metric"),
        "target_column":  target_col,
        "wall_time_s":    wall_time,
        "returncode":     returncode,
        **score,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_tasks(args) -> list[dict]:
    if args.hsc:
        task_ids = load_hsc_task_ids()
    elif args.all:
        tasks = []
        for src in SOURCES:
            src_dir = DATASETS_ROOT / src
            if src_dir.exists():
                for td in sorted(src_dir.iterdir()):
                    if (td / "task.json").exists():
                        tasks.append({"id": td.name, "source": src})
        return tasks
    else:
        task_ids = list(args.extra_tasks or [])
        if args.task:
            task_ids.insert(0, args.task)
        if not task_ids:
            return []

    tasks = []
    for tid in task_ids:
        found = False
        for src in SOURCES:
            if (DATASETS_ROOT / src / tid / "task.json").exists():
                tasks.append({"id": tid, "source": src})
                found = True
                break
        if not found:
            print(f"Task '{tid}' not found in any dataset source.")
            sys.exit(1)
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Run MLEvolve on benchmark tasks")
    ap.add_argument("--task",     metavar="TASK_ID", help="Single task ID")
    ap.add_argument("--tasks",    action="append", dest="extra_tasks", metavar="TASK_ID",
                    help="Task ID (repeatable)")
    ap.add_argument("--hsc",      action="store_true", help="Run all HSC tasks")
    ap.add_argument("--all",      action="store_true", help="Run all benchmark tasks")
    ap.add_argument("--model",    default=DEFAULT_MODEL,
                    help=f"Model name from model_registry.json (default: {DEFAULT_MODEL})")
    ap.add_argument("--steps",    type=int, default=DEFAULT_STEPS,
                    help=f"MCGS search steps (default: {DEFAULT_STEPS})")
    ap.add_argument("--runs-dir", default="mlevolve_runs",
                    help="Parent directory for run outputs (default: mlevolve_runs/)")
    ap.add_argument("--resume",   action="store_true",
                    help="Skip tasks that already have a successful result.json")
    args = ap.parse_args()

    registry  = load_registry()
    model_cfg = resolve_model_cfg(args.model, registry)
    slug      = model_slug(args.model)
    run_root  = PROJECT_ROOT / args.runs_dir / f"mlevolve__{slug}__{args.steps}steps"

    tasks = collect_tasks(args)
    if not tasks:
        ap.error("Specify --task, --tasks, --hsc, or --all")

    wall_timeout_s = resolve_wall_timeout_s(model_cfg)

    print(f"MLEvolve runner — model={args.model}, steps={args.steps}, tasks={len(tasks)}")
    print(f"Run root: {run_root}\n")

    results = []
    for i, t in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {t['id']} ({t['source']})", flush=True)

        if args.resume:
            result_path = run_root / t["id"] / "result.json"
            if result_path.exists():
                existing = json.loads(result_path.read_text())
                wall_time = existing.get("wall_time_s")
                # A machine that sleeps mid-run doesn't pause wall-clock deadlines --
                # time.time() keeps advancing while suspended, so on wake every
                # timeout looks instantly, wildly overdue. Legit timeout-kills
                # cluster tightly at WALL_TIMEOUT + ~0-40s (thread-join overhead);
                # anything well beyond that is a sleep/suspend artifact, not a
                # real 2-hour timeout, so don't trust it as "done".
                suspect = wall_time is not None and wall_time > wall_timeout_s + 600
                if suspect:
                    print(f"  Existing result has wall_time_s={wall_time:.0f}s, "
                          f"over {wall_timeout_s + 600}s -- likely a sleep/suspend "
                          f"artifact rather than a real timeout. Re-running instead "
                          f"of trusting it.", flush=True)
                elif existing.get("success"):
                    print(f"  Skipping (score={existing.get('score')})", flush=True)
                    results.append(existing)
                    print()
                    continue
                else:
                    print(f"  Existing result was a failure ({existing.get('error')}) -- "
                          f"retrying instead of skipping.", flush=True)

        r = run_task(t["id"], t["source"], model_cfg, args.steps, run_root)
        results.append(r)
        print()

    successes = [r for r in results if r.get("success")]
    print(f"=== Done: {len(successes)}/{len(results)} successful ===")
    for r in results:
        if r.get("success"):
            print(f"  {r['task_id']}: score={r['score']:.4f} ({r.get('metric')})")
        else:
            print(f"  {r['task_id']}: FAILED — {r.get('error', '')[:60]}")

    run_root.mkdir(parents=True, exist_ok=True)
    summary = run_root / "results_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved: {summary}")


if __name__ == "__main__":
    main()
