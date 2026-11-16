#!/usr/bin/env python3
"""Vanilla-LLM runner variant that adds honest prompt support for multi-output tasks (currently:
open-problems-single-cell-perturbations).

Usage: identical to run_vanilla.py, e.g.
  python scripts/run_vanilla_multioutput.py --model <model> --variant few_shot_role \
      --task open-problems-single-cell-perturbations
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_as_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_vanilla = _load_as_module("run_vanilla", PROJECT_ROOT / "scripts" / "run_vanilla.py")
_original_build_prompt = run_vanilla.build_prompt


def build_prompt_multioutput(task_dir: Path, task_meta: dict,
                              examples: list, supply_data_snippet: bool) -> str:
    desc = (task_dir / "description.md").read_text(encoding="utf-8").strip() \
        if (task_dir / "description.md").exists() else ""
    train_path = (task_dir / "data" / "train.parquet").resolve()
    test_path  = (task_dir / "data" / "test.parquet").resolve()
    sample_sub_path = (task_dir / "data" / "sample_submission.csv").resolve()
    metric = task_meta.get("metric", "rmse")
    feature_cols = task_meta.get("feature_columns", [])

    parts = []

    # Few-shot examples — unchanged, these are single-target by construction.
    for i, ex in enumerate(examples, 1):
        parts.append(f"=== EXAMPLE {i} ===")
        parts.append(ex.get("description", "").strip())
        parts.append(f"\nData files:\n  train: {ex['train_path']}\n  test:  {ex['test_path']}")
        parts.append(f"Target column: {ex['target_column']}")
        parts.append(f"Evaluation metric: {ex['metric']}")
        if ex.get("data_snippet"):
            parts.append(f"\nData sample (first 5 rows of train):\n{ex['data_snippet']}")
        if ex.get("column_dtypes"):
            parts.append(f"\nColumn dtypes:\n{ex['column_dtypes']}")
        parts.append(f"\n=== SOLUTION {i} ===\n```python\n{ex['code'].strip()}\n```\n")

    if examples:
        parts.append("=== YOUR TASK ===")

    parts.append(desc)
    parts.append(f"\nData files:\n  train: {train_path}\n  test:  {test_path}")
    parts.append(
        "\nThis is a MULTI-OUTPUT task — there is no single target column. "
        "Every column in train.parquet that is NOT in test.parquet is a "
        f"target you must predict ({task_meta.get('n_target_columns', 'many')} "
        "target columns total). The exact expected output columns (id + every "
        f"target column, in order) are shown in: {sample_sub_path}"
    )
    parts.append(f"Evaluation metric: {metric}")

    if supply_data_snippet:
        try:
            df = pd.read_parquet(train_path)
            feat_present = [c for c in feature_cols if c in df.columns] or list(df.columns[:5])
            parts.append(
                f"\nFeature columns and dtypes (train.parquet has "
                f"{len(df.columns)} columns total; only the non-target feature "
                f"columns are shown here):\n{df[feat_present].dtypes.to_string()}"
            )
            parts.append(f"\nFeature sample (first 5 rows):\n{df[feat_present].head(5).to_string()}")
        except Exception:
            pass

    parts.append(f"""
Write a complete, self-contained Python script that:
1. Loads the train and test data from the parquet files above using pandas
2. Trains a model to predict ALL target columns (see sample_submission.csv for exact column names)
3. Saves predictions to predictions.csv in the current working directory,
   with exactly the same columns as sample_submission.csv ("id" plus every target column)

Requirements:
- The script must run without any user input or internet access
- Use only standard libraries plus: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost (all pre-installed)
- Handle missing values and categorical columns automatically
- Do not use pip install or access the internet
- Output only the Python code, no explanation or markdown fences
""")

    return "\n".join(parts).strip()


def build_prompt_dispatch(task_dir: Path, task_meta: dict,
                           examples: list, supply_data_snippet: bool) -> str:
    if task_meta.get("multi_output"):
        return build_prompt_multioutput(task_dir, task_meta, examples, supply_data_snippet)
    return _original_build_prompt(task_dir, task_meta, examples, supply_data_snippet)


run_vanilla.build_prompt = build_prompt_dispatch


if __name__ == "__main__":
    run_vanilla.main()
