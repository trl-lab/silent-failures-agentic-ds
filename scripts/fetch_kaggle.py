#!/usr/bin/env python3
"""Fetch and set up Kaggle competition tasks for the DS agent failure analysis benchmark.

Usage:
  python scripts/fetch_kaggle.py
  python scripts/fetch_kaggle.py --limit 3
  python scripts/fetch_kaggle.py --id titanic
  python scripts/fetch_kaggle.py --skip-download
"""

import argparse
import json
import zipfile
import tempfile
import io
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
    \"\"\"y_pred should be predicted probabilities for the positive class.\"\"\"
    score = float(roc_auc_score(y_true, y_pred))
    return {"score": score, "metric": "roc_auc", "higher_is_better": True}
""",
    "log_loss": """\
from sklearn.metrics import log_loss

def evaluate(y_true, y_pred):
    score = float(log_loss(y_true, y_pred))
    return {"score": score, "metric": "log_loss", "higher_is_better": False}
""",
    "rmse": """\
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(y_true, y_pred):
    score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"score": score, "metric": "rmse", "higher_is_better": False}
""",
    "rmsle": """\
import numpy as np
from sklearn.metrics import mean_squared_log_error

def evaluate(y_true, y_pred):
    # Clip predictions to avoid log(negative)
    y_pred = np.maximum(y_pred, 0)
    score = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
    return {"score": score, "metric": "rmsle", "higher_is_better": False}
""",
    "mae": """\
from sklearn.metrics import mean_absolute_error

def evaluate(y_true, y_pred):
    score = float(mean_absolute_error(y_true, y_pred))
    return {"score": score, "metric": "mae", "higher_is_better": False}
""",
    "r2": """\
from sklearn.metrics import r2_score

def evaluate(y_true, y_pred):
    score = float(r2_score(y_true, y_pred))
    return {"score": score, "metric": "r2", "higher_is_better": True}
""",
    "normalized_gini": """\
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred):
    \"\"\"Normalized Gini coefficient = 2 * AUC - 1.\"\"\"
    auc = float(roc_auc_score(y_true, y_pred))
    score = 2 * auc - 1
    return {"score": score, "metric": "normalized_gini", "higher_is_better": True}
""",
    "qwk": """\
from sklearn.metrics import cohen_kappa_score

def evaluate(y_true, y_pred):
    \"\"\"Quadratic Weighted Kappa. y_pred should be integer class labels.\"\"\"
    import numpy as np
    y_pred_rounded = np.round(y_pred).astype(int)
    score = float(cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic"))
    return {"score": score, "metric": "qwk", "higher_is_better": True}
""",
    "ndcg": """\
from sklearn.metrics import ndcg_score
import numpy as np

def evaluate(y_true, y_pred):
    \"\"\"NDCG@5 for ranking tasks.\"\"\"
    score = float(ndcg_score(np.array([y_true]), np.array([y_pred]), k=5))
    return {"score": score, "metric": "ndcg@5", "higher_is_better": True}
""",
    "wmae": """\
import numpy as np

def evaluate(y_true, y_pred, weights=None):
    \"\"\"Weighted Mean Absolute Error. Falls back to MAE if no weights given.\"\"\"
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if weights is None:
        weights = np.ones(len(y_true))
    score = float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))
    return {"score": score, "metric": "wmae", "higher_is_better": False}
""",
    "concordance_index": """\
from lifelines.utils import concordance_index as ci

def evaluate(y_true, y_pred):
    \"\"\"C-index for survival analysis.
    y_true: DataFrame with 'efs' (event indicator) and 'efs_time' (time to event) columns.
    y_pred: predicted risk scores (higher = higher risk).
    \"\"\"
    event_times = y_true["efs_time"]
    event_observed = y_true["efs"]
    score = float(ci(event_times, -y_pred, event_observed))
    return {"score": score, "metric": "concordance_index", "higher_is_better": True}
""",
    "map12": """\
import numpy as np

def _apk(actual_set, predicted, k=12):
    predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual_set), k) if actual_set else 0.0

def evaluate(y_true, y_pred):
    \"\"\"MAP@12 for ranking/recommendation tasks.
    y_true: list of lists of relevant item ids per query.
    y_pred: list of lists of predicted item ids per query (ranked).
    \"\"\"
    score = float(np.mean([_apk(set(t), p) for t, p in zip(y_true, y_pred)]))
    return {"score": score, "metric": "map@12", "higher_is_better": True}
""",
    "mcrmse": """\
import numpy as np

def evaluate(y_true, y_pred):
    \"\"\"Mean Columnwise Root Mean Squared Error for multi-target regression.
    y_true, y_pred: 2-D arrays of shape (n_samples, n_targets).
    \"\"\"
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    col_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    score = float(np.mean(col_rmse))
    return {"score": score, "metric": "mcrmse", "higher_is_better": False}
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
    comp_id = task_cfg["id"]
    task_type = task_cfg["task_type"]
    target_col = task_cfg["target_column"]
    metric = task_cfg.get("metric", "rmse")

    task_dir = out_dir / comp_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "data").mkdir(exist_ok=True)
    (task_dir / "data" / "original").mkdir(exist_ok=True)
    (task_dir / "evaluation").mkdir(exist_ok=True)

    train_df, test_df = make_split(df, task_type, target_col)

    task_meta = {
        "id": comp_id,
        "display_name": task_cfg.get("display_name", comp_id),
        "source_paper": task_cfg.get("source_paper", "new"),
        "origin": "mlebench" if task_cfg.get("source_paper") == "mlebench" else "kaggle",
        "kaggle_url": f"https://www.kaggle.com/competitions/{comp_id}",
        "hf_url": None,
        "task_type": task_type,
        "target_column": target_col,
        "metric": metric,
        "metric_higher_is_better": metric in ("accuracy", "roc_auc", "r2", "normalized_gini", "qwk", "ndcg"),
        "n_rows": len(df),
        "n_features": len(df.columns) - 1,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "columns": list(df.columns),
        "domain": task_cfg.get("domain"),
        "modality": "tab",
        "kaggle_category": task_cfg.get("kaggle_category"),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "stratified" if task_type in ("b-classification", "m-classification") else "random",
        },
    }
    (task_dir / "task.json").write_text(
        json.dumps(task_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # description.md — use scraped competition description if available
    desc_content = description_text or (
        f"# {task_meta['display_name']}\n\n"
        f"<!-- Description not retrieved. Visit: {task_meta['kaggle_url']} -->\n"
        f"<!-- Target: `{target_col}` | Task: {task_type} | Metric: {metric} -->\n"
    )
    (task_dir / "description.md").write_text(desc_content, encoding="utf-8")

    if not skip_download:
        # data/train.parquet — agent's training set (with target)
        train_table = pa.Table.from_pandas(train_df, preserve_index=False)
        pq.write_table(train_table, task_dir / "data" / "train.parquet")

        # data/test.parquet — agent's test set (no target)
        test_features = test_df.drop(columns=[target_col])
        test_table = pa.Table.from_pandas(test_features, preserve_index=False)
        pq.write_table(test_table, task_dir / "data" / "test.parquet")

        # evaluation/labels.parquet — held-out ground truth (may include secondary label cols)
        secondary = task_cfg.get("secondary_label_columns", [])
        labels_dict = {"id": range(len(test_df)), target_col: test_df[target_col].values}
        for col in secondary:
            if col in test_df.columns:
                labels_dict[col] = test_df[col].values
        labels_df = pd.DataFrame(labels_dict)
        pq.write_table(pa.Table.from_pandas(labels_df, preserve_index=False),
                       task_dir / "evaluation" / "labels.parquet")

    # evaluation/metric.py
    metric_key = metric if metric in METRIC_CODE else "rmse"
    (task_dir / "evaluation" / "metric.py").write_text(METRIC_CODE[metric_key], encoding="utf-8")

    # sample_submission.csv
    dummy = 0.0 if task_type == "regression" else 0
    pd.DataFrame({"id": range(len(test_df)), target_col: [dummy] * len(test_df)}).to_csv(
        task_dir / "sample_submission.csv", index=False
    )

    return task_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/kaggle")
    parser.add_argument("--config", default="scripts/config/kaggle_tasks.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id", default=None, help="Process a single competition by ID")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--registry", default="docs/dataset_registry.md")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output  # default: datasets/kaggle
    out_dir.mkdir(parents=True, exist_ok=True)
    mlebench_dir = project_root / "datasets" / "mlebench"
    mlebench_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((project_root / args.config).read_text())
    tasks = config["competitions"]

    if args.id:
        tasks = [t for t in tasks if t["id"] == args.id]
    elif args.limit:
        tasks = tasks[: args.limit]

    # Deduplicate (config has a duplicate from copy-paste)
    seen = set()
    unique_tasks = []
    for t in tasks:
        if t["id"] not in seen:
            seen.add(t["id"])
            unique_tasks.append(t)
    tasks = unique_tasks

    # Lazy import — only needed when actually downloading
    if not args.skip_download:
        try:
            from kaggle import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("Kaggle API authenticated.\n")
        except Exception as e:
            print(f"ERROR: Could not authenticate Kaggle API: {e}")
            print("Ensure ~/.kaggle/kaggle.json exists and is valid.")
            return

    results = []
    failed = []

    for i, task_cfg in enumerate(tasks, 1):
        comp_id = task_cfg["id"]
        target_col = task_cfg["target_column"]
        train_file = task_cfg.get("file", "train.csv")
        task_out_dir = mlebench_dir if task_cfg.get("source_paper") == "mlebench" else out_dir
        print(f"[{i:3d}/{len(tasks)}] {comp_id}  -> datasets/{task_out_dir.name}/")

        try:
            description_text = None

            if not args.skip_download:
                # --- Fetch competition description ---
                try:
                    response = api.competitions_list(search=comp_id)
                    # Handle both old (list) and new (ApiListCompetitionsResponse) API formats
                    if hasattr(response, "competitions"):
                        comps = response.competitions
                    else:
                        comps = list(response) if response else []
                    if comps:
                        comp_obj = comps[0]
                        title = getattr(comp_obj, "title", comp_id)
                        desc = getattr(comp_obj, "description", "") or ""
                        eval_metric = getattr(comp_obj, "evaluationMetric", "")
                        description_text = (
                            f"# {title}\n\n"
                            f"{desc}\n\n"
                            f"**Evaluation metric:** {eval_metric}\n"
                            f"\n*Source: Kaggle competition — {comp_id}*\n"
                        )
                        print(f"      description: {len(desc)} chars")
                except Exception as desc_err:
                    print(f"      [description] failed: {desc_err}")

                # --- Download training file ---
                print(f"      downloading {train_file} ...")
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    try:
                        api.competition_download_file(comp_id, train_file, path=tmp_path, quiet=True)
                    except Exception:
                        # Fall back to downloading all files and finding the train file
                        api.competition_download_files(comp_id, path=tmp_path, quiet=True)

                    # Find and load the training file — extract all zips recursively
                    csv_candidates = list(tmp_path.rglob(train_file))
                    if not csv_candidates:
                        changed = True
                        while changed:
                            changed = False
                            for zf in list(tmp_path.rglob("*.zip")):
                                try:
                                    with zipfile.ZipFile(zf, "r") as z:
                                        z.extractall(zf.parent)
                                    zf.unlink()
                                    changed = True
                                except Exception:
                                    pass
                        csv_candidates = list(tmp_path.rglob(train_file))

                    if not csv_candidates:
                        raise FileNotFoundError(
                            f"Could not find {train_file} in downloaded files. "
                            f"Found: {list(tmp_path.rglob('*.csv')) + list(tmp_path.rglob('*.json'))}"
                        )

                    csv_path = csv_candidates[0]

                    # Kaggle inconsistently zip-wraps large files even when they
                    # keep their original extension (e.g. a "train.csv" that is
                    # actually zip data) — detect and extract before reading.
                    if zipfile.is_zipfile(csv_path):
                        extract_dir = csv_path.parent / f"_unzipped_{csv_path.stem}"
                        extract_dir.mkdir(exist_ok=True)
                        with zipfile.ZipFile(csv_path) as z:
                            z.extractall(extract_dir)
                        inner_candidates = list(extract_dir.rglob(train_file))
                        if not inner_candidates:
                            raise FileNotFoundError(
                                f"{csv_path.name} was zip-wrapped but {train_file} "
                                f"was not found inside it."
                            )
                        csv_path = inner_candidates[0]

                    print(f"      loading {csv_path.name} ...")
                    if train_file.endswith(".json"):
                        try:
                            df = pd.read_json(csv_path)
                        except ValueError:
                            df = pd.read_json(csv_path, lines=True)
                    elif train_file.endswith(".tsv"):
                        df = pd.read_csv(csv_path, sep="\t", low_memory=False)
                    else:
                        df = pd.read_csv(csv_path, low_memory=False)
                    print(f"      shape: {df.shape}")

                    # Also copy original file to data/original/
                    orig_dir = out_dir / comp_id / "data" / "original"
                    orig_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(csv_path, orig_dir / train_file)

                if target_col not in df.columns:
                    raise ValueError(
                        f"Target '{target_col}' not found. Available: {list(df.columns)[:10]}"
                    )

                # Optional: map raw string/categorical labels to numeric
                # (e.g. "Presence"/"Absence" -> 1/0) for metrics like roc_auc
                # that require numeric targets.
                target_mapping = task_cfg.get("target_mapping")
                if target_mapping:
                    df[target_col] = df[target_col].map(target_mapping)

                # Drop rows where target is missing
                before = len(df)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                if len(df) < before:
                    print(f"      dropped {before - len(df)} rows with missing target")

            else:
                df = pd.DataFrame()

            print("      writing task directory ...")
            meta = write_task(task_out_dir, task_cfg, df, description_text, args.skip_download)
            results.append(meta)
            print(f"      n_train={meta['n_train']}  n_test={meta['n_test']}  done.\n")

        except Exception as exc:
            print(f"      ERROR: {exc}\n")
            failed.append((comp_id, str(exc)))
            results.append(None)

        # Be polite to the Kaggle API
        time.sleep(1)

    # --- Summary ---
    n_ok = sum(1 for r in results if r is not None)
    print("=" * 60)
    print(f"Done. {n_ok}/{len(tasks)} tasks processed successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")

    # Write metadata JSON
    meta_out = project_root / "docs" / "kaggle_metadata.json"
    meta_out.write_text(
        json.dumps([r for r in results if r], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata written to {meta_out}")


if __name__ == "__main__":
    main()
