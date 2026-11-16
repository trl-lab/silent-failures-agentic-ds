"""Skill score over a constant predictor (task-comparable, peer-free).

    lower-is-better (rmse, rmsle, mae): skill = 1 - score / score_of_constant_predictor
        constant = train mean (rmse), train mean of log1p (rmsle), train median (mae)
    accuracy : (acc - majority_acc) / (1 - majority_acc)      (majority class from train)
    roc_auc  : 2 * AUC - 1
    r2       : as is
Other metrics return None. Shared by analyze_memorization_delta.py and select_memorization_tasks.py.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

meta = {}
for f in glob.glob(str(ROOT / "datasets" / "**" / "task.json"), recursive=True):
    d = json.loads(Path(f).read_text())
    d["_dir"] = Path(f).parent
    meta[d["id"]] = d

_ref_cache: dict = {}


def reference(tid: str):
    """Constant-predictor reference score for lower-is-better metrics / majority acc."""
    if tid in _ref_cache:
        return _ref_cache[tid]
    t = meta[tid]
    out = None
    try:
        tr = pd.read_parquet(t["_dir"] / "data" / "train.parquet", columns=[t["target_column"]])[t["target_column"]]
        lab = pd.read_parquet(t["_dir"] / "evaluation" / "labels.parquet")
        y = lab[t["target_column"]] if t["target_column"] in lab.columns else lab.iloc[:, -1]
        m = t["metric"]
        if m == "rmse":
            out = float(np.sqrt(np.mean((y - tr.mean()) ** 2)))
        elif m == "rmsle":
            out = float(np.sqrt(np.mean((np.log1p(y) - np.log1p(tr).mean()) ** 2)))
        elif m == "mae":
            out = float(np.mean(np.abs(y - tr.median())))
        elif m == "accuracy":
            out = float((y == tr.mode().iloc[0]).mean())
    except Exception:
        out = None
    _ref_cache[tid] = out
    return out


def skill(tid: str, metric: str, score: float):
    if metric in ("rmse", "rmsle", "mae"):
        ref = reference(tid)
        return None if not ref else 1 - score / ref
    if metric == "accuracy":
        ref = reference(tid)
        return None if ref is None or ref >= 1 else (score - ref) / (1 - ref)
    if metric == "roc_auc":
        return 2 * score - 1
    if metric == "r2":
        return score
    return None


