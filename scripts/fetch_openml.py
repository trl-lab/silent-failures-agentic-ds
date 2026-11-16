#!/usr/bin/env python3
"""Fetch and set up OpenML datasets for the DS agent failure analysis benchmark.

Usage:
  python scripts/fetch_openml.py
  python scripts/fetch_openml.py --limit 5
  python scripts/fetch_openml.py --did 31          # single dataset by OpenML ID
  python scripts/fetch_openml.py --skip-download   # metadata only
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
    "log_loss": """\
from sklearn.metrics import log_loss

def evaluate(y_true, y_pred):
    score = float(log_loss(y_true, y_pred))
    return {"score": score, "metric": "log_loss", "higher_is_better": False}
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
        "origin": "openml",
        "openml_did": task_cfg["did"],
        "openml_url": f"https://www.openml.org/d/{task_cfg['did']}",
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

    # description.md — use OpenML dataset description if available
    if description_text:
        (task_dir / "description.md").write_text(description_text, encoding="utf-8")
    else:
        (task_dir / "description.md").write_text(
            f"# {task_meta['display_name']}\n\n"
            f"<!-- No description available. See: {task_meta['openml_url']} -->\n"
            f"<!-- Target: `{target_col}` | Task: {task_type} | Metric: {metric} -->\n",
            encoding="utf-8",
        )

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
    parser.add_argument("--output", default="datasets/openml")
    parser.add_argument("--config", default="scripts/config/openml_tasks.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--did", type=int, default=None, help="Process a single dataset by OpenML ID")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    import openml
    # Cache downloaded datasets locally to avoid repeated network hits
    openml.config.cache_directory = str(Path.home() / ".openml" / "cache")

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((project_root / args.config).read_text())
    tasks = config["datasets"]

    if args.did:
        tasks = [t for t in tasks if t["did"] == args.did]
    elif args.limit:
        tasks = tasks[: args.limit]

    results = []
    failed = []

    for i, task_cfg in enumerate(tasks, 1):
        did = task_cfg["did"]
        name = task_cfg["name"]
        target_col = task_cfg["target_column"]
        task_type = task_cfg["task_type"]
        print(f"[{i:3d}/{len(tasks)}] {name}  (did={did})")

        try:
            print(f"      fetching from OpenML ...")
            dataset = openml.datasets.get_dataset(
                did,
                download_data=not args.skip_download,
                download_qualities=True,
                download_features_meta_data=True,
            )

            # Build description from OpenML metadata
            desc_parts = [f"# {task_cfg.get('display_name', name)}\n"]
            if dataset.description:
                desc_parts.append(dataset.description)
            desc_parts.append(f"\n**OpenML dataset ID:** {did}")
            desc_parts.append(f"**URL:** https://www.openml.org/d/{did}")
            desc_parts.append(f"**Target:** `{target_col}` | **Task:** {task_type}")
            description_text = "\n\n".join(desc_parts)

            if not args.skip_download:
                try:
                    result = dataset.get_data(
                        target=target_col,
                        dataset_format="dataframe",
                    )
                    X, y = result[0], result[1]
                    df = X.copy()
                    df[target_col] = y
                except (IndexError, KeyError, TypeError) as get_err:
                    print(f"      [warn] get_data(target=...) failed ({get_err}), retrying without target")
                    result = dataset.get_data(dataset_format="dataframe")
                    df = result[0] if isinstance(result, tuple) else result
                    if target_col not in df.columns:
                        lower_map = {c.lower(): c for c in df.columns}
                        if target_col.lower() in lower_map:
                            actual = lower_map[target_col.lower()]
                            df = df.rename(columns={actual: target_col})
                            print(f"      [warn] renamed '{actual}' -> '{target_col}'")
                        else:
                            raise ValueError(
                                f"Target '{target_col}' not found. Available: {list(df.columns)[:10]}"
                            )

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

        time.sleep(0.5)  # be polite to OpenML API

    n_ok = sum(1 for r in results if r is not None)
    print("=" * 60)
    print(f"Done. {n_ok}/{len(tasks)} datasets processed successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")

    meta_out = project_root / "docs" / "openml_metadata.json"
    meta_out.write_text(
        json.dumps([r for r in results if r], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata written to {meta_out}")


if __name__ == "__main__":
    main()
