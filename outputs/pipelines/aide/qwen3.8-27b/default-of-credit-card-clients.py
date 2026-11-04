import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "default payment next month"
ID_COL = "ID"
WORK_DIR = "./working"

os.makedirs(WORK_DIR, exist_ok=True)


def x_num(col):
    if isinstance(col, str) and col.startswith("X") and len(col) > 1:
        try:
            return int(col[1:])
        except ValueError:
            return None
    return None


def prepare(df, has_target=False):
    df = df.copy()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    y = None
    if has_target and TARGET in df.columns:
        y = df[TARGET].copy()
        df = df.drop(columns=[TARGET])

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0.0)

    original_x_cols = [c for c in df.columns if x_num(c) is not None]
    original_x_cols.sort(key=lambda c: x_num(c))
    df = df[original_x_cols]

    bill_cols = [c for c in df.columns if 12 <= x_num(c) <= 17]
    pay_amt_cols = [c for c in df.columns if 18 <= x_num(c) <= 23]
    pay_status_cols = [c for c in df.columns if 6 <= x_num(c) <= 11]

    for c in bill_cols + pay_amt_cols:
        df[c] = np.where(df[c] < 0, 0.0, df[c])

    if pay_status_cols:
        df["pay_status_max"] = df[pay_status_cols].max(axis=1)
        df["pay_status_min"] = df[pay_status_cols].min(axis=1)
        df["pay_status_mean"] = df[pay_status_cols].mean(axis=1)
        df["pay_status_trend"] = (
            df[pay_status_cols].iloc[:, -1] - df[pay_status_cols].iloc[:, 0]
        )

    if bill_cols:
        df["bill_total"] = df[bill_cols].sum(axis=1)
        df["bill_max"] = df[bill_cols].max(axis=1)
        df["bill_std"] = df[bill_cols].std(axis=1).fillna(0.0)

    if pay_amt_cols:
        df["pay_amt_total"] = df[pay_amt_cols].sum(axis=1)
        df["pay_amt_max"] = df[pay_amt_cols].max(axis=1)
        df["pay_amt_std"] = df[pay_amt_cols].std(axis=1).fillna(0.0)

    if "bill_total" in df.columns and "pay_amt_total" in df.columns:
        df["pay_to_bill_ratio"] = df["pay_amt_total"] / (df["bill_total"].abs() + 1.0)
        df["net_amount"] = df["bill_total"] - df["pay_amt_total"]

    if has_target:
        return df, y
    return df


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

X_train_all, y = prepare(train, has_target=True)
X_test = prepare(test, has_target=False)

common_cols = [c for c in X_train_all.columns if c in X_test.columns]
X_train_all = X_train_all[common_cols]
X_test = X_test[common_cols]

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbosity": -1,
}


def make_callbacks():
    return [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y), dtype=float)
fold_aucs = []
best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_all, y), start=1):
    dtrain = lgb.Dataset(X_train_all.iloc[train_idx], label=y.iloc[train_idx])
    dvalid = lgb.Dataset(
        X_train_all.iloc[valid_idx],
        label=y.iloc[valid_idx],
        reference=dtrain,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=make_callbacks(),
    )

    preds = model.predict(
        X_train_all.iloc[valid_idx], num_iteration=model.best_iteration
    )
    oof_preds[valid_idx] = preds
    auc = roc_auc_score(y.iloc[valid_idx], preds)

    fold_aucs.append(auc)
    best_iterations.append(model.best_iteration)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

mean_fold_auc = float(np.mean(fold_aucs))
std_fold_auc = float(np.std(fold_aucs))
oof_auc = float(roc_auc_score(y, oof_preds))

print(f"5-Fold CV ROC AUC (mean): {mean_fold_auc:.6f} (+/- {std_fold_auc:.6f})")
print(f"5-Fold OOF ROC AUC: {oof_auc:.6f}")

final_rounds = max(10, int(np.mean(best_iterations)))
final_model = lgb.train(
    params,
    lgb.Dataset(X_train_all, label=y),
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(period=-1)],
)

test_preds = final_model.predict(X_test, num_iteration=final_rounds)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=int),
        TARGET: test_preds,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print("Saved submission to ./working/submission.csv")
