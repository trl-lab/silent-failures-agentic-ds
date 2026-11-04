#!/usr/bin/env python3
"""Fetch and prepare the home-credit-credit-risk-model-stability task as a multi-table dataset
(unlike the other Kaggle tasks in this benchmark, which use a single train.parquet/test.parquet).

Usage:
  python scripts/fetch_home_credit.py
"""

import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

COMP_ID = "home-credit-credit-risk-model-stability"
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_ROOT / "datasets" / "kaggle" / COMP_ID

METRIC_ROC_AUC = """\
from sklearn.metrics import roc_auc_score

def evaluate(y_true, y_pred):
    \"\"\"y_pred should be predicted probabilities for the positive class.\"\"\"
    score = float(roc_auc_score(y_true, y_pred))
    return {"score": score, "metric": "roc_auc", "higher_is_better": True}
"""


def list_train_files(api) -> list[str]:
    names = []
    token = None
    while True:
        resp = (
            api.competition_list_files(COMP_ID, page_token=token)
            if token
            else api.competition_list_files(COMP_ID)
        )
        files = resp.files if hasattr(resp, "files") else resp
        names.extend(f.name for f in files)
        token = getattr(resp, "nextPageToken", None)
        if not token:
            break
    return sorted(n for n in names if n.startswith("parquet_files/train/"))


def download_table(api, remote_name: str, dest_dir: Path) -> Path:
    """Download one parquet file. Kaggle inconsistently zip-wraps some files
    (large ones observed so far) but not others — handle both."""
    fname = remote_name.split("/")[-1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        api.competition_download_file(COMP_ID, remote_name, path=tmp, quiet=True)
        dpath = tmp / fname
        out_path = dest_dir / fname
        if zipfile.is_zipfile(dpath):
            with zipfile.ZipFile(dpath) as z:
                inner = z.namelist()[0]
                with z.open(inner) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
        else:
            dpath.replace(out_path)
    return out_path


def main():
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated.\n")

    (TASK_DIR / "data" / "train").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "data" / "test").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "evaluation").mkdir(parents=True, exist_ok=True)

    raw_dir = TASK_DIR / "_raw_train_tables"
    raw_dir.mkdir(exist_ok=True)

    remote_names = list_train_files(api)
    print(f"Found {len(remote_names)} train tables.\n")

    local_paths = {}
    for i, remote in enumerate(remote_names, 1):
        fname = remote.split("/")[-1]
        local = raw_dir / fname
        if local.exists():
            print(f"  [{i}/{len(remote_names)}] {fname} — already downloaded, skipping")
        else:
            print(f"  [{i}/{len(remote_names)}] downloading {fname} ...")
            download_table(api, remote, raw_dir)
        local_paths[fname] = local

    # --- Split case_ids 80/20 on the base table, stratified on target ---
    base_df = pd.read_parquet(local_paths["train_base.parquet"])
    print(f"\nbase table: {base_df.shape}")

    train_idx, test_idx = train_test_split(
        base_df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=base_df["target"],
    )
    train_base = base_df.loc[train_idx].reset_index(drop=True)
    test_base = base_df.loc[test_idx].reset_index(drop=True)

    train_case_ids = set(train_base["case_id"])
    test_case_ids = set(test_base["case_id"])
    print(f"train case_ids: {len(train_case_ids)}  test case_ids: {len(test_case_ids)}")

    # --- Write base tables ---
    pq.write_table(
        pa.Table.from_pandas(train_base, preserve_index=False),
        TASK_DIR / "data" / "train" / "train_base.parquet",
    )

    test_base_out = test_base.drop(columns=["target"]).copy()
    test_base_out.insert(0, "id", range(len(test_base_out)))
    pq.write_table(
        pa.Table.from_pandas(test_base_out, preserve_index=False),
        TASK_DIR / "data" / "test" / "test_base.parquet",
    )

    # --- Filter and write all join tables by case_id membership ---
    for fname, path in local_paths.items():
        if fname == "train_base.parquet":
            continue
        df = pd.read_parquet(path)
        if "case_id" not in df.columns:
            print(f"  [warn] {fname} has no case_id column, skipping split")
            continue
        train_part = df[df["case_id"].isin(train_case_ids)].reset_index(drop=True)
        test_part = df[df["case_id"].isin(test_case_ids)].reset_index(drop=True)

        out_name = fname.replace("train_", "", 1)
        pq.write_table(
            pa.Table.from_pandas(train_part, preserve_index=False),
            TASK_DIR / "data" / "train" / f"train_{out_name}",
        )
        pq.write_table(
            pa.Table.from_pandas(test_part, preserve_index=False),
            TASK_DIR / "data" / "test" / f"test_{out_name}",
        )
        print(f"  {fname}: train={len(train_part)} test={len(test_part)}")

    # --- evaluation/labels.parquet, aligned to test_base_out's "id" order ---
    labels_df = pd.DataFrame({"id": range(len(test_base)), "target": test_base["target"].values})
    pq.write_table(
        pa.Table.from_pandas(labels_df, preserve_index=False),
        TASK_DIR / "evaluation" / "labels.parquet",
    )

    (TASK_DIR / "evaluation" / "metric.py").write_text(METRIC_ROC_AUC, encoding="utf-8")

    pd.DataFrame({"id": range(len(test_base)), "target": [0.0] * len(test_base)}).to_csv(
        TASK_DIR / "sample_submission.csv", index=False
    )

    # --- task.json ---
    table_names = sorted({p.name.replace("train_", "", 1).replace("test_", "", 1)
                           for p in (TASK_DIR / "data" / "train").glob("*.parquet")})
    task_meta = {
        "id": COMP_ID,
        "display_name": "Home Credit — Credit Risk Model Stability",
        "source_paper": "new",
        "origin": "kaggle",
        "kaggle_url": f"https://www.kaggle.com/competitions/{COMP_ID}",
        "task_type": "b-classification",
        "target_column": "target",
        "metric": "roc_auc",
        "metric_higher_is_better": True,
        "n_train": len(train_base),
        "n_test": len(test_base),
        "domain": "finance",
        "modality": "tab",
        "kaggle_category": "Featured",
        "multi_table": True,
        "join_key": "case_id",
        "tables": table_names,
        "notes": (
            "Multi-table task: base table (case_id, date_decision, MONTH, WEEK_NUM, target) "
            "carries no predictive signal on its own — real features live in the join tables "
            "(applprev, credit_bureau_a/b, person, static, tax_registry, debitcard, deposit, "
            "other), all keyed by case_id. Agent must join them itself. data/train/ and "
            "data/test/ each contain all 31 tables filtered to our 80/20 case_id split; "
            "data/test/test_base.parquet has target dropped and carries the sequential 'id' "
            "column (0-based) that evaluation/labels.parquet and predictions are keyed on."
        ),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "stratified on target, by case_id",
        },
    }
    (TASK_DIR / "task.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")

    (TASK_DIR / "description.md").write_text(
        "# Home Credit — Credit Risk Model Stability\n\n"
        "Predict binary loan default risk (`target`). Multi-table dataset: the base table "
        "(`train_base.parquet`/`test_base.parquet`) only has `case_id`, `date_decision`, "
        "`MONTH`, `WEEK_NUM` and (train only) `target` — all real features are in separate "
        "join tables keyed by `case_id` (previous applications, credit bureau records, "
        "person/static features, tax registry, debit card, deposit, other). You must join "
        "the relevant tables on `case_id` yourself.\n\n"
        f"**Evaluation metric:** roc_auc\n\n"
        f"*Source: Kaggle competition — {COMP_ID}*\n",
        encoding="utf-8",
    )

    print(f"\nDone. Task dir: {TASK_DIR}")
    print(f"Raw downloaded tables kept at: {raw_dir} (safe to delete once satisfied)")


if __name__ == "__main__":
    main()
