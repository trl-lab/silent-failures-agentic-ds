#!/usr/bin/env python3
"""Vanilla-LLM runner variant that adds honest prompt support for multi-table tasks (currently: home-
credit-credit-risk-model-stability).

Usage: identical to run_vanilla.py, e.g.
  python scripts/run_vanilla_multitable.py --model <model> --variant few_shot_role \
      --task home-credit-credit-risk-model-stability
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


def build_prompt_multitable(task_dir: Path, task_meta: dict,
                             examples: list, supply_data_snippet: bool) -> str:
    desc = (task_dir / "description.md").read_text(encoding="utf-8").strip() \
        if (task_dir / "description.md").exists() else ""
    train_dir = (task_dir / "data" / "train").resolve()
    test_dir  = (task_dir / "data" / "test").resolve()
    target = task_meta["target_column"]
    metric = task_meta.get("metric", "rmse")

    parts = []

    # Few-shot examples — unchanged, these are single-table by construction.
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
    parts.append(f"\nData directories:\n  train: {train_dir}\n  test:  {test_dir}")

    train_files = sorted(p.name for p in train_dir.glob("*.parquet")) if train_dir.exists() else []
    test_files  = sorted(p.name for p in test_dir.glob("*.parquet")) if test_dir.exists() else []
    parts.append(
        "\nThese are directories, not single files — each contains multiple related "
        "parquet tables (not a single flat table). You must explore them yourself to "
        "understand their schemas and how they relate to each other.\n"
        "Train directory files:\n  " + "\n  ".join(train_files) + "\n"
        "Test directory files:\n  " + "\n  ".join(test_files)
    )
    parts.append(f"Target column: {target}")
    parts.append(f"Evaluation metric: {metric}")

    if supply_data_snippet:
        try:
            lines = ["\nColumn names and dtypes per train file (schema only, no row content):"]
            for fname in train_files:
                df = pd.read_parquet(train_dir / fname)
                lines.append(f"\n  {fname}:")
                for col, dtype in df.dtypes.items():
                    lines.append(f"    {col}: {dtype}")
            parts.append("\n".join(lines))
        except Exception:
            pass

    parts.append(f"""
Write a complete, self-contained Python script that:
1. Loads the train and test data from the parquet files in the directories above using pandas
2. Trains a model to predict "{target}"
3. Saves predictions to predictions.csv in the current working directory,
   with exactly two columns: "id" (integer row index starting from 0) and "{target}"

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
    if task_meta.get("multi_table"):
        return build_prompt_multitable(task_dir, task_meta, examples, supply_data_snippet)
    return _original_build_prompt(task_dir, task_meta, examples, supply_data_snippet)


run_vanilla.build_prompt = build_prompt_dispatch


if __name__ == "__main__":
    run_vanilla.main()
