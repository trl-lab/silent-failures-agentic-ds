#!/usr/bin/env python3
"""Fetch and set up STRABLE benchmark tasks.

Usage:
    python scripts/fetch_strable.py
    python scripts/fetch_strable.py --limit 5            # process only first N datasets
    python scripts/fetch_strable.py --dataset wine-dataset  # single dataset
    python scripts/fetch_strable.py --skip-download      # metadata only, no parquet download
"""

import argparse
import json
import struct
import io
import time
import sys
from pathlib import Path

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = "inria-soda/STRABLE-benchmark"
HF_RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
HF_TREE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
HF_DATASET_DIR = f"https://huggingface.co/datasets/{REPO}/tree/main"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# task field in config.json  →  (metric_name, higher_is_better)
METRIC_MAP = {
    "regression":      ("rmse",     False),
    "b-classification": ("roc_auc",  True),
    "m-classification": ("accuracy", True),
}

# Metric implementations written verbatim into each task's evaluation/metric.py
METRIC_CODE = {
    "rmse": '''\
import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate(y_true, y_pred):
    """Root Mean Squared Error. Lower is better."""
    score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"score": score, "metric": "rmse", "higher_is_better": False}
''',
    "roc_auc": '''\
from sklearn.metrics import roc_auc_score


def evaluate(y_true, y_pred):
    """ROC-AUC score. Higher is better.
    y_pred should be predicted probabilities for the positive class."""
    score = float(roc_auc_score(y_true, y_pred))
    return {"score": score, "metric": "roc_auc", "higher_is_better": True}
''',
    "accuracy": '''\
from sklearn.metrics import accuracy_score


def evaluate(y_true, y_pred):
    """Accuracy score. Higher is better."""
    score = float(accuracy_score(y_true, y_pred))
    return {"score": score, "metric": "accuracy", "higher_is_better": True}
''',
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ds-failure-analysis/1.0"})
    return s


SESSION = _session()


def _get(url: str, *, stream: bool = False, retries: int = 3, **kwargs):
    """GET with retries and redirect following."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, allow_redirects=True, stream=stream, timeout=60, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"      [retry {attempt+1}/{retries}] {exc} — waiting {wait}s")
            time.sleep(wait)


def _head(url: str, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            r = SESSION.head(url, allow_redirects=True, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

# ---------------------------------------------------------------------------
# Parquet footer metadata (reads only the footer, not the full file)
# ---------------------------------------------------------------------------

def get_parquet_metadata(url: str) -> dict | None:
    """Read parquet file metadata (n_rows, n_columns, column_names) by downloading only the file footer
    via HTTP Range requests.
    """
    try:
        # Resolve redirect to get final URL and Content-Length
        head = _head(url)
        final_url = head.url
        size = int(head.headers.get("Content-Length", 0))
        if size == 0:
            return None

        # Read last 8 bytes: footer_length (4) + magic (4)
        tail = _get(final_url, headers={"Range": f"bytes={size - 8}-{size - 1}"})
        if tail.status_code not in (200, 206):
            raise ValueError(f"Range request returned {tail.status_code}")

        footer_len = struct.unpack("<I", tail.content[:4])[0]
        magic = tail.content[4:]
        if magic != b"PAR1":
            raise ValueError(f"Bad magic bytes: {magic!r}")

        # Read footer + magic
        footer_start = size - footer_len - 8
        footer_resp = _get(
            final_url,
            headers={"Range": f"bytes={footer_start}-{size - 1}"},
        )
        # pyarrow needs a complete parquet byte stream; prepend PAR1 magic
        buf = io.BytesIO(b"PAR1" + footer_resp.content)
        meta = pq.read_metadata(buf)
        schema = pq.read_schema(io.BytesIO(b"PAR1" + footer_resp.content))
        return {
            "n_rows": meta.num_rows,
            "n_columns": meta.num_columns,
            "column_names": schema.names,
        }

    except Exception as exc:
        # Fallback: download the full file
        print(f"      [range-request failed: {exc}] — falling back to full download")
        r = _get(url)
        buf = io.BytesIO(r.content)
        meta = pq.read_metadata(buf)
        schema = pq.read_schema(io.BytesIO(r.content))
        return {
            "n_rows": meta.num_rows,
            "n_columns": meta.num_columns,
            "column_names": schema.names,
        }

# ---------------------------------------------------------------------------
# HuggingFace dataset folder listing
# ---------------------------------------------------------------------------

def list_dataset_files(dataset_name: str) -> list[dict]:
    """Return all files in the HuggingFace folder for this dataset."""
    r = _get(f"{HF_TREE_API}/{dataset_name}")
    return r.json()

# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_split(
    df: pd.DataFrame,
    task: str,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train (80%) and test (20%).
    Uses stratified split for classification; random for regression.
    Returns (train_df, test_df) — both still contain the target column.
    """
    y = df[target_col]

    stratify = None
    if task in ("b-classification", "m-classification"):
        # Only stratify if every class has at least 2 samples
        counts = y.value_counts()
        if counts.min() >= 2:
            stratify = y
        else:
            print(f"      [split] skipping stratification — some classes have < 2 samples")

    try:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        # Fallback to random split if stratification fails
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sample submission generator
# ---------------------------------------------------------------------------

def make_sample_submission(test_df: pd.DataFrame, target_col: str, task: str) -> pd.DataFrame:
    """
    Build a sample_submission.csv with dummy values in the correct format.
    The agent must replace the dummy values with its predictions.
    """
    n = len(test_df)
    dummy: object
    if task == "regression":
        dummy = 0.0
    elif task == "b-classification":
        dummy = 0.0   # expected to be probability of positive class
    else:
        # m-classification: use first actual class value as dummy
        dummy = 0
    return pd.DataFrame({"id": range(n), target_col: [dummy] * n})


# ---------------------------------------------------------------------------
# Write task directory
# ---------------------------------------------------------------------------

def write_task(
    out_dir: Path,
    dataset_name: str,
    config: dict,
    parquet_meta: dict,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extra_files: list[dict],
    skip_download: bool,
) -> dict:
    """
    Write the complete task directory for one STRABLE dataset.
    Returns the task metadata dict (for registry).
    """
    task_dir = out_dir / dataset_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "data").mkdir(exist_ok=True)
    (task_dir / "evaluation").mkdir(exist_ok=True)
    (task_dir / "docs").mkdir(exist_ok=True)

    task_type = config["task"]
    target_col = config["target_name"]
    metric, higher_is_better = METRIC_MAP.get(task_type, ("rmse", False))
    n_train = len(train_df)
    n_test = len(test_df)
    n_features = parquet_meta["n_columns"] - 1  # exclude target column

    # -- task.json --
    task_meta = {
        "id": dataset_name,
        "display_name": dataset_name.replace("-", " ").replace("_", " ").title(),
        "source_paper": "STRABLE",
        "origin": config.get("source", ""),
        "hf_url": f"https://huggingface.co/datasets/{REPO}/tree/main/{dataset_name}",
        "kaggle_url": None,
        "task_type": task_type,
        "target_column": target_col,
        "metric": metric,
        "metric_higher_is_better": higher_is_better,
        "n_rows": parquet_meta["n_rows"],
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,
        "columns": parquet_meta["column_names"],
        "domain": None,  # to be filled in manually / future enrichment
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

    # -- description.md  (empty — STRABLE has no per-dataset descriptions) --
    desc_path = task_dir / "description.md"
    if not desc_path.exists():
        desc_path.write_text(
            f"# {task_meta['display_name']}\n\n"
            f"<!-- No description available from source ({config.get('source', 'unknown')}). -->\n"
            f"<!-- Target column: `{target_col}` | Task: {task_type} | Metric: {metric} -->\n",
            encoding="utf-8",
        )

    if not skip_download:
        # -- data/data.parquet  (original, untouched) --
        data_url = f"{HF_RESOLVE}/{dataset_name}/data.parquet"
        raw_bytes = _get(data_url).content
        (task_dir / "data" / "data.parquet").write_bytes(raw_bytes)

        # -- data/train.parquet --
        train_table = pa.Table.from_pandas(train_df, preserve_index=False)
        pq.write_table(train_table, task_dir / "data" / "train.parquet")

        # -- data/test.parquet  (target column removed) --
        test_features = test_df.drop(columns=[target_col])
        test_table = pa.Table.from_pandas(test_features, preserve_index=False)
        pq.write_table(test_table, task_dir / "data" / "test.parquet")

        # -- evaluation/labels.parquet  (held-out test targets) --
        labels_df = pd.DataFrame({
            "id": range(n_test),
            target_col: test_df[target_col].values,
        })
        labels_table = pa.Table.from_pandas(labels_df, preserve_index=False)
        pq.write_table(labels_table, task_dir / "evaluation" / "labels.parquet")

    # -- evaluation/metric.py --
    (task_dir / "evaluation" / "metric.py").write_text(
        METRIC_CODE[metric], encoding="utf-8"
    )

    # -- sample_submission.csv --
    sample_sub = make_sample_submission(test_df, target_col, task_type)
    sample_sub.to_csv(task_dir / "sample_submission.csv", index=False)

    # -- docs/  (any extra files found alongside data.parquet in HF) --
    known_files = {"config.json", "data.parquet"}
    doc_files = [
        f for f in extra_files
        if Path(f["path"]).name not in known_files
    ]
    if doc_files and not skip_download:
        for file_info in doc_files:
            filename = Path(file_info["path"]).name
            file_url = f"{HF_RESOLVE}/{file_info['path']}"
            try:
                content = _get(file_url).content
                (task_dir / "docs" / filename).write_bytes(content)
                print(f"      [docs] saved {filename}")
            except Exception as exc:
                print(f"      [docs] failed to fetch {filename}: {exc}")

    return task_meta


# ---------------------------------------------------------------------------
# Registry update
# ---------------------------------------------------------------------------

def update_registry(registry_path: Path, results: list[dict]) -> None:
    """
    Rewrite the STRABLE section of dataset_registry.md with populated metadata.
    Rows that failed (result is None) are preserved from the original.
    """
    if not registry_path.exists():
        print(f"  [registry] {registry_path} not found — skipping update")
        return

    text = registry_path.read_text(encoding="utf-8")

    # Build lookup by id
    meta_by_id = {r["id"]: r for r in results if r}

    lines = text.splitlines(keepends=True)
    new_lines = []
    in_strable_table = False

    for line in lines:
        # Detect start of STRABLE table body (after the header row)
        if "### A — Agent-K" in line:
            in_strable_table = False
        if "### B — STRABLE" in line:
            in_strable_table = True

        if in_strable_table and line.startswith("|") and not line.startswith("| id") and not line.startswith("|---"):
            # Parse the id from the first cell
            cells = [c.strip() for c in line.split("|")]
            if len(cells) > 1:
                row_id = cells[1].strip()
                if row_id in meta_by_id:
                    m = meta_by_id[row_id]
                    task_map = {
                        "regression": "RG",
                        "b-classification": "BC",
                        "m-classification": "MC",
                    }
                    new_line = (
                        f"| {m['id']} "
                        f"| {task_map.get(m['task_type'], m['task_type'])} "
                        f"| Tab "
                        f"| {m.get('domain') or '?'} "
                        f"| {m['metric']} "
                        f"| {m['n_rows']:,} "
                        f"| {m['n_features']} "
                        f"| N/A "
                        f"| pending "
                        f"| origin: {m['origin']} |\n"
                    )
                    new_lines.append(new_line)
                    continue

        new_lines.append(line)

    registry_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"  [registry] updated {registry_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="datasets/strable",
        help="Output directory for task folders (default: datasets/strable)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N datasets (useful for testing)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Process a single dataset by name",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Fetch metadata and write task.json/metric.py only; skip parquet downloads",
    )
    parser.add_argument(
        "--registry", default="docs/dataset_registry.md",
        help="Path to dataset_registry.md to update (default: docs/dataset_registry.md)",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root (parent of this script)
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output
    registry_path = project_root / args.registry

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Get the list of all datasets ----
    print("Fetching dataset list from HuggingFace...")
    tree = _get(f"https://huggingface.co/api/datasets/{REPO}/tree/main").json()
    all_datasets = sorted(
        entry["path"] for entry in tree if entry["type"] == "directory"
    )
    print(f"Found {len(all_datasets)} datasets.\n")

    if args.dataset:
        datasets = [args.dataset]
    elif args.limit:
        datasets = all_datasets[: args.limit]
    else:
        datasets = all_datasets

    results = []
    failed = []

    for i, dataset_name in enumerate(datasets, 1):
        print(f"[{i:3d}/{len(datasets)}] {dataset_name}")

        try:
            # 1. config.json
            print("      fetching config.json ...")
            config_url = f"{HF_RESOLVE}/{dataset_name}/config.json"
            config = _get(config_url).json()
            task_type = config.get("task", "unknown")
            target_col = config.get("target_name", "")
            print(f"      task={task_type}  target='{target_col}'  source={config.get('source', '')}")

            # 2. Parquet metadata via range request
            print("      reading parquet metadata ...")
            parquet_url = f"{HF_RESOLVE}/{dataset_name}/data.parquet"
            parquet_meta = get_parquet_metadata(parquet_url)
            if parquet_meta is None:
                raise ValueError("Could not read parquet metadata")
            print(f"      n_rows={parquet_meta['n_rows']}  n_columns={parquet_meta['n_columns']}")

            # 3. List all files in HF folder (to detect extra docs)
            print("      listing folder contents ...")
            folder_files = list_dataset_files(dataset_name)

            if not args.skip_download:
                # 4. Download full parquet and load into DataFrame
                print("      downloading data.parquet ...")
                r = _get(f"{HF_RESOLVE}/{dataset_name}/data.parquet")
                full_df = pd.read_parquet(io.BytesIO(r.content))

                # Validate target column exists
                if target_col not in full_df.columns:
                    raise ValueError(
                        f"Target column '{target_col}' not found in data. "
                        f"Available: {list(full_df.columns)}"
                    )

                # 5. Create train/test split
                print("      creating 80/20 split ...")
                train_df, test_df = make_split(full_df, task_type, target_col)
                print(f"      train={len(train_df)}  test={len(test_df)}")
            else:
                # Metadata-only mode: create minimal DataFrames just for task.json
                full_df = pd.DataFrame(columns=parquet_meta["column_names"])
                train_df = full_df.copy()
                test_df = full_df.copy()

            # 6. Write task directory
            print("      writing task directory ...")
            meta = write_task(
                out_dir=out_dir,
                dataset_name=dataset_name,
                config=config,
                parquet_meta=parquet_meta,
                full_df=full_df,
                train_df=train_df,
                test_df=test_df,
                extra_files=folder_files,
                skip_download=args.skip_download,
            )
            results.append(meta)
            print(f"      done.\n")

        except Exception as exc:
            print(f"      ERROR: {exc}\n")
            failed.append((dataset_name, str(exc)))
            results.append(None)

    # ---- Summary ----
    n_ok = sum(1 for r in results if r is not None)
    print("=" * 60)
    print(f"Done. {n_ok}/{len(datasets)} datasets processed successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")

    # ---- Write metadata JSON ----
    meta_out = project_root / "docs" / "strable_metadata.json"
    meta_out.write_text(
        json.dumps([r for r in results if r], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nMetadata written to {meta_out}")

    # ---- Update registry markdown ----
    update_registry(registry_path, [r for r in results if r])


if __name__ == "__main__":
    main()
