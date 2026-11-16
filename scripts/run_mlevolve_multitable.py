#!/usr/bin/env python3
"""MLEvolve runner variant that adds honest task-description support for multi-table tasks
(currently: home-credit-credit-risk-model-stability).

Usage: identical to run_mlevolve.py, e.g.
  python scripts/run_mlevolve_multitable.py --task home-credit-credit-risk-model-stability
  python scripts/run_mlevolve_multitable.py --hsc --model gemini/gemini-3-pro-preview --steps 10 --resume
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


def build_task_desc_multitable(task_dir: Path, task_meta: dict) -> str:
    desc = ""
    if (task_dir / "description.md").exists():
        desc = (task_dir / "description.md").read_text(encoding="utf-8").strip()

    target = task_meta["target_column"]
    metric = task_meta.get("metric", "rmse")
    higher = task_meta.get("metric_higher_is_better", True)
    direction = "higher is better" if higher else "lower is better"

    lines = [run_mlevolve.DS_PERSONA, ""]
    if desc:
        lines += [desc, ""]

    lines += [
        f"**Target column**: `{target}`",
        f"**Evaluation metric**: `{metric}` ({direction})",
        "",
        "**Data**: Training data is in `./input/train/` and test data is in "
        "`./input/test/`. Each of these is a directory containing multiple "
        "related parquet files (not a single flat table). Explore the files "
        "to understand their schemas and how they relate to each other before "
        "modelling.",
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


def build_task_desc_dispatch(task_dir: Path, task_meta: dict) -> str:
    if task_meta.get("multi_table"):
        return build_task_desc_multitable(task_dir, task_meta)
    return _original_build_task_desc(task_dir, task_meta)


run_mlevolve.build_task_desc = build_task_desc_dispatch


if __name__ == "__main__":
    run_mlevolve.main()
