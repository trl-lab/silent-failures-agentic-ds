"""AutoGluon reference solutions for the DSTS TF0X unit-test suite -- same methodology and rationale
as run_autogluon_hsc.py (mirrors github.com/KARTIK-N-NAYAK/Benchmarking-Agentic-Data-Scientists's
autogluon_runner_cla.py/reg.py: presets="best_quality", time_limit=1800s default,
eval_metric="roc_auc"/"accuract"/"r2" by problem kind).

Usage:
    python scripts/run_autogluon_dsts.py --dataset TF01_goal_imbalance
    python scripts/run_autogluon_dsts.py --all
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from dsts_common import external_name  # noqa: E402
DSTS_DATA = Path(os.environ.get("DSTS_ROOT", PROJECT_ROOT / "dsts_suite")) / "data"
OUT_ROOT = PROJECT_ROOT / "autogluon_runs" / "dsts"

# One entry per UNIQUE underlying dataset (not per TF0X hint variant --
# see module docstring). "represents" lists which task ids share this data,
# purely for the results doc, not used for anything else.
DATASET_CONFIGS = {
    "TF01_goal_imbalance": {
        "data_dir": "TF01_goal_imbalance", "train": "player_goals_5high_train.csv",
        "test": "player_goals_5high_test.csv", "target": "goals_label", "kind": "binary",
        "represents": ["TF01_goal_imbalance_01/02/03"],
    },
    "TF02_player_rep_norm_clean": {
        "data_dir": "TF02_player_rep_norm", "train": "player_value_train.csv",
        "test": "player_value_test.csv", "target": "market_value_in_eur", "kind": "regression",
        "represents": ["TF02_player_rep_norm_01"],
    },
    "TF02_player_rep_norm_corrupted": {
        "data_dir": "TF02_player_rep_norm", "train": "player_value_rep_norm_train.csv",
        "test": "player_value_rep_norm_test.csv", "target": "market_value_in_eur", "kind": "regression",
        "represents": ["TF02_player_rep_norm_02/03/04"],
    },
    "TF03_correlation_causation": {
        "data_dir": "TF03_correlation_causation", "train": "player_value_corr_causal_train.csv",
        "test": "player_value_corr_causal_test.csv", "target": "market_value_in_eur", "kind": "regression",
        "represents": ["TF03_correlation_causation_01/02"],
    },
    "TF04_balance_torque": {
        "data_dir": "TF04_balance_torque", "train": "balance_torque_train.csv",
        "test": "balance_torque_test.csv", "target": "Class Name", "kind": "multiclass",
        "represents": ["TF04_balance_torque_01/02/03/04"],
    },
    "TF05_credit_card_leakage": {
        "data_dir": "TF05_credit_card_leakage", "train": "credit_card_train.csv",
        "test": "credit_card_test.csv", "target": "card", "kind": "binary",
        "represents": ["TF05_credit_card_leakage_01/02"],
    },
    "TF06_metric_select": {
        "data_dir": "TF06_metric_select", "train": "player_goals_imbalance_train.csv",
        "test": "player_goals_imbalance_test.csv", "target": "goals_label", "kind": "binary",
        "represents": ["TF06_metric_select_01/02"],
    },
}

EVAL_METRIC = {"binary": "roc_auc", "multiclass": "accuracy", "regression": "r2"}
# AutoGluon auto-infers problem_type from the label column's values by
# default, and got it wrong for market_value_in_eur (rounded/discrete-looking
# floats got misdetected as multiclass) -- causing a hard ValueError since
# eval_metric="r2" isn't valid for multiclass. Pass problem_type explicitly
# everywhere instead of trusting the inference.
PROBLEM_TYPE = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}


def run_dataset(name: str, time_limit: int) -> dict:
    from autogluon.tabular import TabularPredictor  # deferred import -- heavy, only needed here

    cfg = DATASET_CONFIGS[name]
    kind = cfg["kind"]
    target = cfg["target"]
    input_dir = DSTS_DATA / external_name(cfg["data_dir"]) / "input"

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / cfg["train"])
    test_df = pd.read_csv(input_dir / cfg["test"])

    eval_metric = EVAL_METRIC[kind]
    print(f"[{name}] train={train_df.shape} test={test_df.shape} target={target} "
          f"kind={kind} eval_metric={eval_metric} time_limit={time_limit}s "
          f"(represents: {cfg['represents']})", flush=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=target, path=str(out_dir / "predictor"), eval_metric=eval_metric,
        problem_type=PROBLEM_TYPE[kind],
    ).fit(train_df, time_limit=time_limit, presets="best_quality")
    fit_time = time.time() - t0

    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    model_best = predictor.model_best

    feature_cols = [c for c in test_df.columns if c != target]
    y_true = test_df[target]

    if kind == "binary":
        proba = predictor.predict_proba(test_df[feature_cols])
        y_score = proba.iloc[:, -1]
        score = roc_auc_score(y_true, y_score)
    elif kind == "multiclass":
        y_pred = predictor.predict(test_df[feature_cols])
        score = accuracy_score(y_true, y_pred)
    else:
        y_pred = predictor.predict(test_df[feature_cols])
        score = r2_score(y_true, y_pred)

    result = {
        "dataset": name, "represents": cfg["represents"], "kind": kind,
        "eval_metric": eval_metric, "score": float(score),
        "fit_time_s": fit_time, "success": True,
        "model_best": model_best, "n_models_trained": len(leaderboard),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] OK in {fit_time/60:.1f}min: {eval_metric}={score:.4f}\n", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help=f"one of {list(DATASET_CONFIGS)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--time-limit", type=int, default=1800,
                     help="AutoGluon fit time budget in seconds (default 1800s/30min)")
    args = ap.parse_args()

    if args.all:
        names = list(DATASET_CONFIGS)
    elif args.dataset:
        if args.dataset not in DATASET_CONFIGS:
            sys.exit(f"'{args.dataset}' not recognized -- one of {list(DATASET_CONFIGS)}")
        names = [args.dataset]
    else:
        sys.exit("specify --dataset <name> or --all")

    for name in names:
        run_dataset(name, args.time_limit)


if __name__ == "__main__":
    main()
