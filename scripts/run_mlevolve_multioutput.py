#!/usr/bin/env python3
"""
MLEvolve runner variant that adds honest task-description support for
multi-output tasks (currently: open-problems-single-cell-perturbations).

Same rationale as run_aide_multioutput.py: the scoring/post-processing
generalization already lives directly in run_mlevolve.py (pure measurement
code, doesn't change what the agent is told); this wrapper only patches
build_task_desc(), gated on task.json's "multi_output": true, so every other
task falls through to the original, byte-identical function.

Usage: identical to run_mlevolve.py, e.g.
  python scripts/run_mlevolve_multioutput.py --task open-problems-single-cell-perturbations --steps 10
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_as_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_mlevolve = _load_as_module("run_mlevolve", PROJECT_ROOT / "scripts" / "run_mlevolve.py")
_original_build_task_desc = run_mlevolve.build_task_desc


def build_task_desc_multioutput(task_dir: Path, task_meta: dict) -> str:
    desc = ""
    if (task_dir / "description.md").exists():
        desc = (task_dir / "description.md").read_text(encoding="utf-8").strip()

    metric = task_meta.get("metric", "rmse")
    higher = task_meta.get("metric_higher_is_better", True)
    direction = "higher is better" if higher else "lower is better"

    lines = [run_mlevolve.DS_PERSONA, ""]
    if desc:
        lines += [desc, ""]

    lines += [
        f"**Evaluation metric**: `{metric}` ({direction})",
        "",
        "**Data**: Train data is at `./input/train.parquet` and test data at "
        "`./input/test.parquet`. Read them with `pd.read_parquet()`.",
        "",
        "**This is a multi-output task** — there is no single target column. "
        "`./input/sample_submission.csv` shows the exact columns your "
        "predictions must contain (an `id` column plus every target column, "
        "in the expected order/names). Every column in train.parquet that is "
        "NOT in test.parquet is a target you must predict.",
        "",
        "**Installed package versions** (use APIs compatible with these):",
        "- `lightgbm==4.6.0`: `early_stopping_rounds` and `verbose_eval` were removed "
        "from `lgb.train()`. Use callbacks: "
        "`callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`",
        "- `scikit-learn==1.5.0`, `pandas==2.1.4`",
        "",
        "**Output**: Write test predictions to `./submission/submission.csv` with "
        "exactly the same columns as `./input/sample_submission.csv` (`id` plus "
        "every target column), values filled in with your predictions.",
    ]
    return "\n".join(lines)


def build_task_desc_dispatch(task_dir: Path, task_meta: dict) -> str:
    if task_meta.get("multi_output"):
        return build_task_desc_multioutput(task_dir, task_meta)
    return _original_build_task_desc(task_dir, task_meta)


run_mlevolve.build_task_desc = build_task_desc_dispatch


if __name__ == "__main__":
    run_mlevolve.main()
