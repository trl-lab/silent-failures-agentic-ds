"""AutoGluon reference solutions for the HSC benchmark -- an independent, non-agent baseline to
validate agent scores against ("is 0.65 good, or is 0.65 just what any competent AutoML gets
here").

Usage:
    python scripts/run_autogluon_hsc.py --task playground-series-s4e2
    python scripts/run_autogluon_hsc.py --all --time-limit 1800
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_vanilla import score_predictions  # noqa: E402

DATASETS = PROJECT_ROOT / "datasets" / "kaggle"
OUT_ROOT = PROJECT_ROOT / "autogluon_runs"

# task_id -> (id column in test.parquet, problem kind)
#   "binary"      -- eval_metric="roc_auc", predict_proba() positive-class column
#   "multiclass"  -- eval_metric="accuracy" (roc_auc not valid for AutoGluon multiclass), predict()
#   "regression"  -- eval_metric="r2", predict()
TASK_CONFIGS = {
    "playground-series-s4e2": {"id_col": "id", "kind": "multiclass"},
    "playground-series-s6e2": {"id_col": "id", "kind": "binary"},
    "equity-post-HCT-survival-predictions": {"id_col": "ID", "kind": "binary"},
    "optiver-trading-at-the-close": {"id_col": "row_id", "kind": "regression"},
}

EVAL_METRIC = {"binary": "roc_auc", "multiclass": "accuracy", "regression": "r2"}
# See run_autogluon_dsts.py's matching comment -- AutoGluon's auto-inferred
# problem_type got a regression target wrong there, so it's passed explicitly
# everywhere rather than trusted to infer correctly.
PROBLEM_TYPE = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}


def run_task(task_id: str, time_limit: int) -> dict:
    from autogluon.tabular import TabularPredictor  # deferred import -- heavy, only needed here

    cfg = TASK_CONFIGS[task_id]
    kind = cfg["kind"]
    task_dir = DATASETS / task_id
    task_meta = json.loads((task_dir / "task.json").read_text())
    target_col = task_meta["target_column"]

    out_dir = OUT_ROOT / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(task_dir / "data" / "train.parquet")
    test_df = pd.read_parquet(task_dir / "data" / "test.parquet")

    id_col = cfg["id_col"]
    # Drop the dataset's own semantic id column from training too -- fit()
    # would otherwise use every non-label column of train_df as a feature by
    # default, including a raw, non-predictive row identifier.
    train_df_fit = train_df.drop(columns=[id_col]) if id_col in train_df.columns else train_df

    eval_metric = EVAL_METRIC[kind]
    print(f"[{task_id}] train={train_df_fit.shape} test={test_df.shape} target={target_col} "
          f"kind={kind} eval_metric={eval_metric} time_limit={time_limit}s", flush=True)

    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        path=str(out_dir / "predictor"),
        eval_metric=eval_metric,
        problem_type=PROBLEM_TYPE[kind],
    ).fit(train_df_fit, time_limit=time_limit, presets="best_quality")
    fit_time = time.time() - t0

    # Save what AutoGluon actually chose -- every trained model + its
    # validation score, and which one/ensemble was selected as final. Lets
    # you compare "what AutoGluon converged on" against the agent's own
    # generated code, not just the two final numeric scores.
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    model_best = predictor.model_best

    # feature_cols excludes id_col -- it's the dataset's own semantic
    # identifier (e.g. a compound stock/date/second string for optiver, an
    # arbitrary row ID elsewhere), not a real feature, and letting AutoGluon
    # train on a raw identifier column is a real leakage/robustness risk
    # regardless of the id-matching fix below.
    feature_cols = [c for c in test_df.columns if c != id_col]
    if kind == "binary":
        proba = predictor.predict_proba(test_df[feature_cols])
        preds = proba.iloc[:, -1]  # positive class -- same convention as the reference repo
    else:
        preds = predictor.predict(test_df[feature_cols])

    pred_id = pd.Series(range(len(test_df)), name="id")
    predictions_df = pd.DataFrame({"id": pred_id, target_col: preds})
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)

    result = score_predictions(task_dir, out_dir, target_col)
    result["task_id"] = task_id
    result["fit_time_s"] = fit_time
    result["model_best"] = model_best
    result["n_models_trained"] = len(leaderboard)
    result["autogluon_eval_metric"] = eval_metric
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    status = "OK" if result.get("success") else "FAILED"
    print(f"[{task_id}] {status} in {fit_time/60:.1f}min: {result}\n", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task id (must be one of the 4 supported)")
    ap.add_argument("--all", action="store_true", help="run all 4 supported tasks")
    ap.add_argument("--time-limit", type=int, default=1800,
                     help="AutoGluon fit time budget in seconds (default 1800s/30min, "
                          "matching the reference repo's default)")
    args = ap.parse_args()

    if args.all:
        task_ids = list(TASK_CONFIGS)
    elif args.task:
        if args.task not in TASK_CONFIGS:
            sys.exit(f"'{args.task}' not supported today -- only {list(TASK_CONFIGS)}")
        task_ids = [args.task]
    else:
        sys.exit("specify --task <id> or --all")

    for task_id in task_ids:
        run_task(task_id, args.time_limit)


if __name__ == "__main__":
    main()
