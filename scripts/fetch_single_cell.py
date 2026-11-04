#!/usr/bin/env python3
"""Fetch and prepare the open-problems-single-cell-perturbations task.

Output structure:
  datasets/kaggle/open-problems-single-cell-perturbations/
    data/
      train.parquet   # feature cols + all 18211 gene target cols, 80% of rows
      test.parquet    # feature cols only + sequential "id", 20% of rows
    evaluation/
      labels.parquet  # id + all 18211 gene target cols (ground truth)
      metric.py       # mrrmse (mean ROWWISE rmse across genes)
    task.json
    description.md
    sample_submission.csv

Usage:
  python scripts/fetch_single_cell.py
"""

import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

COMP_ID = "open-problems-single-cell-perturbations"
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COLS = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_ROOT / "datasets" / "kaggle" / COMP_ID

METRIC_MRRMSE = """\
import numpy as np

def evaluate(y_true, y_pred):
    \"\"\"Mean Rowwise Root Mean Squared Error: RMSE across genes computed
    per row, then averaged across rows (not per-column).\"\"\"
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    row_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    score = float(np.mean(row_rmse))
    return {"score": score, "metric": "mrrmse", "higher_is_better": False}
"""


def download_file(api, remote_name: str, dest_dir: Path, max_retries: int = 3) -> Path:
    out_path = dest_dir / remote_name
    last_err = None
    for attempt in range(1, max_retries + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                api.competition_download_file(COMP_ID, remote_name, path=tmp, quiet=True)
                dpath = tmp / remote_name
                if not dpath.exists():
                    raise FileNotFoundError(f"{remote_name} not written to {tmp}")
                if zipfile.is_zipfile(dpath):
                    with zipfile.ZipFile(dpath) as z:
                        inner = z.namelist()[0]
                        with z.open(inner) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                else:
                    dpath.replace(out_path)
                return out_path
            except Exception as exc:
                last_err = exc
                print(f"    [retry {attempt}/{max_retries}] {remote_name} failed: {exc}")
    raise RuntimeError(f"Could not download {remote_name} after {max_retries} attempts") from last_err


def main():
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated.\n")

    (TASK_DIR / "data").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "evaluation").mkdir(parents=True, exist_ok=True)

    raw_dir = TASK_DIR / "_raw_download"
    raw_dir.mkdir(exist_ok=True)

    local_path = raw_dir / "de_train.parquet"
    if local_path.exists():
        print("de_train.parquet — already downloaded, skipping")
    else:
        print("downloading de_train.parquet ...")
        download_file(api, "de_train.parquet", raw_dir)

    df = pd.read_parquet(local_path)
    print(f"de_train.parquet: {df.shape}")

    target_cols = [c for c in df.columns if c not in FEATURE_COLS]
    print(f"feature cols: {FEATURE_COLS}")
    print(f"target cols: {len(target_cols)} genes")

    try:
        train_idx, test_idx = train_test_split(
            df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=df["cell_type"],
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    print(f"train split: {len(train_df)}  test split: {len(test_df)}")

    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False),
                    TASK_DIR / "data" / "train.parquet")

    test_features = test_df[FEATURE_COLS].copy()
    test_features.insert(0, "id", range(len(test_features)))
    pq.write_table(pa.Table.from_pandas(test_features, preserve_index=False),
                    TASK_DIR / "data" / "test.parquet")

    labels_df = pd.concat(
        [pd.DataFrame({"id": range(len(test_df))}), test_df[target_cols].reset_index(drop=True)],
        axis=1,
    )
    pq.write_table(pa.Table.from_pandas(labels_df, preserve_index=False),
                    TASK_DIR / "evaluation" / "labels.parquet")

    (TASK_DIR / "evaluation" / "metric.py").write_text(METRIC_MRRMSE, encoding="utf-8")

    sample_sub = pd.concat(
        [pd.DataFrame({"id": range(len(test_df))}),
         pd.DataFrame(0.0, index=range(len(test_df)), columns=target_cols)],
        axis=1,
    )
    sample_sub.to_csv(TASK_DIR / "sample_submission.csv", index=False)
    # Also inside data/ so it's copied into the agent's workspace — the exact
    # target column names (18,211 genes) need to come from a real file the
    # agent can read, not be enumerated in the prompt.
    sample_sub.to_csv(TASK_DIR / "data" / "sample_submission.csv", index=False)

    task_meta = {
        "id": COMP_ID,
        "display_name": "Open Problems — Single Cell Perturbations",
        "source_paper": "new",
        "origin": "kaggle",
        "kaggle_url": f"https://www.kaggle.com/competitions/{COMP_ID}",
        "task_type": "multi_output_regression",
        "target_column": "__multi_output__",
        "feature_columns": FEATURE_COLS,
        "n_target_columns": len(target_cols),
        "metric": "mrrmse",
        "metric_higher_is_better": False,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "domain": "bioinformatics",
        "modality": "tab",
        "kaggle_category": "Featured",
        "multi_output": True,
        "notes": (
            "Multi-output regression: predict all 18,211 gene-expression "
            "differential-expression columns jointly for each "
            "(cell_type, sm_name) pair. train.parquet has feature_columns "
            "plus all target gene columns; test.parquet has only "
            "feature_columns + sequential 'id'. evaluation/labels.parquet has "
            "'id' + all gene columns. NOT YET RUNNABLE through "
            "vanilla/AIDE/MLEvolve as-is: their build_task_desc/build_prompt "
            "assume a single target_column for the output-format instructions, "
            "and score_predictions() hardcodes a single predicted column "
            "(preds_df[['id', target_col]]) even though the ground-truth side "
            "already supports multiple label columns. Needs a multi-output-aware "
            "wrapper (task description + scoring) before running, analogous to "
            "the multi_table wrappers."
        ),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "stratified on cell_type (fallback: random)",
        },
    }
    (TASK_DIR / "task.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")

    (TASK_DIR / "description.md").write_text(
        "# Open Problems — Single Cell Perturbations\n\n"
        "Predict differential gene expression (18,211 genes) for each "
        "(cell_type, sm_name) drug-cell-type combination. This is a "
        "**multi-output regression** task — the prediction target is a full "
        "row of 18,211 values per test example, not a single column. "
        "Feature columns available: cell_type, sm_name, sm_lincs_id, SMILES, "
        "control.\n\n"
        "**Evaluation metric:** mrrmse (mean rowwise RMSE across genes)\n\n"
        f"*Source: Kaggle competition — {COMP_ID}*\n",
        encoding="utf-8",
    )

    print(f"\nDone. Task dir: {TASK_DIR}")
    print(f"Raw downloaded files kept at: {raw_dir} (safe to delete once satisfied)")


if __name__ == "__main__":
    main()
