#!/usr/bin/env python3
"""Fetch and set up UCI ML Repository datasets for the DS agent failure analysis benchmark.

Usage:
  python scripts/fetch_uci.py
  python scripts/fetch_uci.py --limit 5
  python scripts/fetch_uci.py --id 31          # single dataset by UCI ID
  python scripts/fetch_uci.py --skip-download
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

METRIC_CODE = {
    "accuracy": """\
from sklearn.metrics import accuracy_score

def evaluate(y_true, y_pred):
    score = float(accuracy_score(y_true, y_pred))
    return {"score": score, "metric": "accuracy", "higher_is_better": True}
""",
    "roc_auc": """\
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred):
    score = float(roc_auc_score(y_true, y_pred))
    return {"score": score, "metric": "roc_auc", "higher_is_better": True}
""",
    "rmse": """\
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(y_true, y_pred):
    score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"score": score, "metric": "rmse", "higher_is_better": False}
""",
}


def make_split(df, task_type, target_col):
    y = df[target_col]
    stratify = None
    if task_type in ("b-classification", "m-classification"):
        counts = y.value_counts()
        if counts.min() >= 2:
            stratify = y
    try:
        train_idx, test_idx = train_test_split(
            df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def build_description(task_cfg, dataset) -> str:
    """Build a description.md from UCI metadata and variable info."""
    name = task_cfg.get("display_name", task_cfg["name"])
    lines = [f"# {name}\n"]

    # UCI abstract / description
    abstract = getattr(dataset.metadata, "abstract", None)
    if abstract:
        lines.append(abstract)

    # Variable descriptions table
    if dataset.variables is not None and not dataset.variables.empty:
        lines.append("\n## Variable Descriptions\n")
        lines.append("| Name | Role | Type | Description |")
        lines.append("|------|------|------|-------------|")
        for _, row in dataset.variables.iterrows():
            col_name = row.get("name", "")
            role = row.get("role", "")
            dtype = row.get("type", "")
            desc = str(row.get("description", "") or "").replace("\n", " ")[:120]
            lines.append(f"| {col_name} | {role} | {dtype} | {desc} |")

    lines.append(f"\n**UCI Dataset ID:** {task_cfg['id']}")
    lines.append(f"**URL:** https://archive.ics.uci.edu/dataset/{task_cfg['id']}")
    lines.append(f"**Target:** `{task_cfg['target_column']}` | **Task:** {task_cfg['task_type']}")
    return "\n".join(lines)


def write_task(out_dir, task_cfg, df, description_text, skip_download):
    name = task_cfg["name"]
    task_type = task_cfg["task_type"]
    target_col = task_cfg["target_column"]
    metric = task_cfg.get("metric", "rmse")

    task_dir = out_dir / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "data").mkdir(exist_ok=True)
    (task_dir / "evaluation").mkdir(exist_ok=True)

    train_df, test_df = make_split(df, task_type, target_col)

    task_meta = {
        "id": name,
        "display_name": task_cfg.get("display_name", name),
        "source_paper": "new",
        "origin": "uci",
        "uci_id": task_cfg["id"],
        "uci_url": f"https://archive.ics.uci.edu/dataset/{task_cfg['id']}",
        "kaggle_url": None,
        "hf_url": None,
        "task_type": task_type,
        "target_column": target_col,
        "metric": metric,
        "metric_higher_is_better": metric in ("accuracy", "roc_auc"),
        "n_rows": len(df),
        "n_features": len(df.columns) - 1,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "columns": list(df.columns),
        "domain": task_cfg.get("domain"),
        "modality": "tab",
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "stratified" if task_type in ("b-classification", "m-classification") else "random",
        },
    }
    (task_dir / "task.json").write_text(
        json.dumps(task_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (task_dir / "description.md").write_text(description_text or "", encoding="utf-8")

    if not skip_download:
        train_table = pa.Table.from_pandas(train_df, preserve_index=False)
        pq.write_table(train_table, task_dir / "data" / "train.parquet")

        test_features = test_df.drop(columns=[target_col])
        test_table = pa.Table.from_pandas(test_features, preserve_index=False)
        pq.write_table(test_table, task_dir / "data" / "test.parquet")

        labels_df = pd.DataFrame({"id": range(len(test_df)), target_col: test_df[target_col].values})
        pq.write_table(
            pa.Table.from_pandas(labels_df, preserve_index=False),
            task_dir / "evaluation" / "labels.parquet",
        )

    metric_key = metric if metric in METRIC_CODE else "rmse"
    (task_dir / "evaluation" / "metric.py").write_text(METRIC_CODE[metric_key], encoding="utf-8")

    dummy = 0.0 if task_type == "regression" else 0
    pd.DataFrame({"id": range(len(test_df)), target_col: [dummy] * len(test_df)}).to_csv(
        task_dir / "sample_submission.csv", index=False
    )

    return task_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/uci")
    parser.add_argument("--config", default="scripts/config/uci_tasks.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id", type=int, default=None, help="Process a single dataset by UCI ID")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    from ucimlrepo import fetch_ucirepo

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((project_root / args.config).read_text())
    tasks = config["datasets"]

    if args.id:
        tasks = [t for t in tasks if t["id"] == args.id]
    elif args.limit:
        tasks = tasks[: args.limit]

    results = []
    failed = []

    for i, task_cfg in enumerate(tasks, 1):
        uci_id = task_cfg["id"]
        name = task_cfg["name"]
        target_col = task_cfg["target_column"]
        print(f"[{i:3d}/{len(tasks)}] {name}  (uci_id={uci_id})")

        try:
            print(f"      fetching from UCI ...")
            dataset = fetch_ucirepo(id=uci_id)

            description_text = build_description(task_cfg, dataset)

            if not args.skip_download:
                X = dataset.data.features
                y = dataset.data.targets

                # Deduplicate column names (e.g. istanbul-stock-exchange has two 'ISE' columns)
                if X.columns.duplicated().any():
                    seen = {}
                    new_cols = []
                    for col in X.columns:
                        if col in seen:
                            seen[col] += 1
                            new_cols.append(f"{col}_{seen[col]}")
                        else:
                            seen[col] = 0
                            new_cols.append(col)
                    X.columns = new_cols
                    print(f"      [warn] deduplicated columns: {[c for c in new_cols if '_' in c and c.split('_')[-1].isdigit()]}")

                if y is None:
                    # Target not separated by UCI API — must be a column in X
                    df = X.copy()
                    if target_col not in df.columns:
                        raise ValueError(
                            f"Target '{target_col}' not found in features. "
                            f"Available: {list(df.columns)[:10]}"
                        )
                else:
                    # Handle multi-column targets — use only the specified target column
                    if hasattr(y, "columns"):
                        if target_col in y.columns:
                            y = y[target_col]
                        else:
                            y = y.iloc[:, 0]
                            print(f"      [warn] target '{target_col}' not found, using '{y.name}'")
                    df = X.copy()
                    df[target_col] = y.values

                # Drop rows with missing target
                before = len(df)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                if len(df) < before:
                    print(f"      dropped {before - len(df)} rows with missing target")

                print(f"      shape: {df.shape}")
            else:
                df = pd.DataFrame()

            print(f"      writing task directory ...")
            meta = write_task(out_dir, task_cfg, df, description_text, args.skip_download)
            results.append(meta)
            print(f"      n_train={meta['n_train']}  n_test={meta['n_test']}  done.\n")

        except Exception as exc:
            print(f"      ERROR: {exc}\n")
            failed.append((name, str(exc)))
            results.append(None)

        time.sleep(0.5)

    n_ok = sum(1 for r in results if r is not None)
    print("=" * 60)
    print(f"Done. {n_ok}/{len(tasks)} datasets processed successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")

    meta_out = project_root / "docs" / "uci_metadata.json"
    meta_out.write_text(
        json.dumps([r for r in results if r], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata written to {meta_out}")


if __name__ == "__main__":
    main()
