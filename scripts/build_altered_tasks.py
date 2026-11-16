"""Build memorization-intervention copies of tasks: random, meaning-free column names.

Usage: python scripts/build_altered_tasks.py --tasks kaggle/titanic openml/cmc [--seed 0]
       (programmatic: build(bucket, tid, seed) / eligible(bucket, tid))
"""
import argparse
import hashlib
import json
import random
import shutil
import string
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets"
OUT = ROOT / "datasets_altered"
BUCKET = "altered"


def alt_id(tid: str, seed: int) -> str:
    return "alt_" + hashlib.sha1(f"{seed}:{tid}".encode()).hexdigest()[:8]


def eligible(bucket: str, tid: str):
    """(ok, reason). Needs local data, a single label column that is a train column."""
    d = SRC / bucket / tid
    for rel in ("task.json", "data/train.parquet", "data/test.parquet", "evaluation/labels.parquet", "evaluation/metric.py"):
        if not (d / rel).exists():
            return False, f"missing {rel}"
    meta = json.loads((d / "task.json").read_text())
    if meta.get("multi_output"):
        return False, "multi_output"
    labels = pd.read_parquet(d / "evaluation" / "labels.parquet")
    label_cols = [c for c in labels.columns if c != "id"]
    train_cols = list(pd.read_parquet(d / "data" / "train.parquet").columns)
    if len(label_cols) != 1:
        return False, f"{len(label_cols)} label columns"
    if label_cols[0] != meta["target_column"] or label_cols[0] not in train_cols:
        return False, "target not a single train column"
    test_cols = list(pd.read_parquet(d / "data" / "test.parquet").columns)
    if not set(test_cols) <= set(train_cols):
        return False, "test has columns absent from train"
    text = (d / "evaluation" / "metric.py").read_text()
    if any(f'"{c}"' in text or f"'{c}'" in text for c in train_cols):
        return False, "metric.py references column names"
    return True, ""


def random_names(n: int, rng: random.Random) -> list:
    alphabet = string.ascii_lowercase + string.digits
    names = set()
    while len(names) < n:
        names.add(rng.choice(string.ascii_lowercase) + "".join(rng.choice(alphabet) for _ in range(5)))
    return sorted(names, key=lambda _: rng.random())


def build(bucket: str, tid: str, seed: int = 0) -> dict:
    ok, why = eligible(bucket, tid)
    if not ok:
        raise ValueError(f"{bucket}/{tid} not eligible: {why}")
    src = SRC / bucket / tid
    aid = alt_id(tid, seed)
    dst = OUT / BUCKET / aid
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    (dst / "evaluation").mkdir()
    meta = json.loads((src / "task.json").read_text())
    train = pd.read_parquet(src / "data" / "train.parquet")
    test = pd.read_parquet(src / "data" / "test.parquet")
    labels = pd.read_parquet(src / "evaluation" / "labels.parquet")

    rng = random.Random(f"{seed}:{tid}")
    cols = list(train.columns)
    order = cols[:]
    rng.shuffle(order)
    cmap = dict(zip(order, random_names(len(order), rng)))  # original -> opaque, in permuted order

    train_a = train[order].rename(columns=cmap)
    test_a = test[[c for c in order if c in test.columns]].rename(columns=cmap)
    labels_a = labels.rename(columns=cmap)  # keeps the pipeline's own "id" column untouched
    train_a.to_parquet(dst / "data" / "train.parquet", index=False)
    test_a.to_parquet(dst / "data" / "test.parquet", index=False)
    labels_a.to_parquet(dst / "evaluation" / "labels.parquet", index=False)
    shutil.copy(src / "evaluation" / "metric.py", dst / "evaluation" / "metric.py")

    task_type = meta.get("task_type", "tabular")
    (dst / "description.md").write_text(
        "Tabular prediction task.\n\n"
        f"Task type: {task_type}.\n"
        "You are given a training table (with the target column) and a test table (without it). "
        "Predict the target for every row of the test table using the other columns. "
        "All columns have anonymous names.\n")
    (dst / "task.json").write_text(json.dumps({
        "id": aid, "task_type": task_type, "target_column": cmap[meta["target_column"]],
        "metric": meta.get("metric"), "metric_higher_is_better": meta.get("metric_higher_is_better"),
        "n_rows": meta.get("n_rows"), "n_train": len(train), "n_test": len(test),
        "columns": list(train_a.columns),
    }, indent=1))
    return {"alt_id": aid, "task_id": tid, "bucket": bucket, "seed": seed, "col_map": cmap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, help="bucket/id, e.g. kaggle/titanic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mp_path = OUT / "mapping.json"
    mapping = json.loads(mp_path.read_text()) if mp_path.exists() else {}
    OUT.mkdir(exist_ok=True)
    for t in a.tasks:
        bucket, tid = t.split("/", 1)
        rec = build(bucket, tid, a.seed)
        mapping[rec["alt_id"]] = rec
        print(f"{t} -> {rec['alt_id']}")
    mp_path.write_text(json.dumps(mapping, indent=1))


if __name__ == "__main__":
    main()
