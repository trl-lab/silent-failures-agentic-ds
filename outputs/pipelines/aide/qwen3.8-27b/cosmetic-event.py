import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

os.makedirs("./working", exist_ok=True)

TARGET = "outcomes"
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

id_like = {"id", "index", "row_id", "sample_id", "uid", "__index_level_0__"}
id_cols = [c for c in train.columns if str(c).lower() in id_like]
train = train.drop(columns=id_cols, errors="ignore")
test = test.drop(columns=[c for c in id_cols if c in test.columns], errors="ignore")

mask = train[TARGET].notna()
train = train.loc[mask].reset_index(drop=True)
y = train[TARGET].reset_index(drop=True)
train = train.drop(columns=[TARGET])

classes = np.unique(y.values)

if len(classes) == 0:
    print("CV accuracy: 0.000000")
    sub = pd.DataFrame({"id": np.arange(len(test)), "outcomes": [None] * len(test)})
    sub.to_csv("./working/submission.csv", index=False)
    raise SystemExit

if len(classes) == 1:
    print("CV accuracy: 1.000000")
    sub = pd.DataFrame({"id": np.arange(len(test)), "outcomes": classes[0]})
    sub.to_csv("./working/submission.csv", index=False)
    raise SystemExit

class_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = y.map(class_to_idx).astype(int)

numeric_cols = []
cat_cols = []

for col in train.columns:
    s = train[col]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        if s.nunique(dropna=True) > 1:
            numeric_cols.append(col)
    elif (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or str(s.dtype).startswith("string")
    ):
        if s.nunique(dropna=True) > 1:
            cat_cols.append(col)

feature_cols = numeric_cols + cat_cols
X_train = pd.DataFrame(index=train.index)
X_test = pd.DataFrame(index=test.index)

for col in numeric_cols:
    s = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = s.median()
    if pd.isna(med):
        med = 0.0

    X_train[col] = s.fillna(med)

    if col in test.columns:
        t = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        t = pd.Series(np.nan, index=test.index)

    X_test[col] = t.fillna(med)

for col in cat_cols:
    train_vals = train[col]
    train_str = (
        train_vals.astype("object").where(train_vals.notna(), "missing").astype(str)
    )
    cats = list(train_str.unique())
    if "missing" not in cats:
        cats.append("missing")

    X_train[col] = train_str.astype("category")

    if col in test.columns:
        test_vals = test[col]
        test_str = (
            test_vals.astype("object").where(test_vals.notna(), "missing").astype(str)
        )
        test_str = test_str.where(test_str.isin(cats), "missing")
    else:
        test_str = pd.Series("missing", index=test.index)

    X_test[col] = test_str.astype("category")
    X_train[col] = X_train[col].cat.set_categories(cats)
    X_test[col] = X_test[col].cat.set_categories(cats)

X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

if X_train.shape[1] == 0:
    majority = y.mode().iloc[0]
    cv_acc = float((y == majority).mean())
    print(f"CV accuracy: {cv_acc:.6f}")
    sub = pd.DataFrame({"id": np.arange(len(test)), "outcomes": majority})
    sub.to_csv("./working/submission.csv", index=False)
    raise SystemExit

params = {
    "objective": "multiclass",
    "num_class": len(classes),
    "metric": "multi_error",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

categorical_feature = cat_cols if cat_cols else None

n_samples = len(y_idx)
min_class_count = int(y_idx.value_counts().min())

if n_samples >= 5 and min_class_count >= 5:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
elif n_samples >= 2 and min_class_count >= 2:
    n_splits = max(2, min(5, min_class_count, n_samples))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
else:
    splitter = KFold(n_splits=2, shuffle=True, random_state=42)

accs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train, y_idx), start=1):
    dtrain = lgb.Dataset(
        X_train.iloc[tr_idx],
        label=y_idx.iloc[tr_idx],
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_train.iloc[va_idx],
        label=y_idx.iloc[va_idx],
        categorical_feature=categorical_feature,
        reference=dtrain,
        free_raw_data=False,
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iter = max(1, int(booster.best_iteration))
    best_iters.append(best_iter)

    pred = booster.predict(X_train.iloc[va_idx], num_iteration=best_iter)
    pred_idx = np.argmax(pred, axis=1)
    pred_labels = classes[pred_idx]

    acc = accuracy_score(y.iloc[va_idx], pred_labels)
    accs.append(acc)

    print(f"Fold {fold} accuracy: {acc:.6f}, best_iteration: {best_iter}")

cv_acc = float(np.mean(accs))
print(f"CV accuracy: {cv_acc:.6f}")

if len(test) == 0:
    sub = pd.DataFrame(
        {"id": np.array([], dtype=int), "outcomes": np.array([], dtype=classes.dtype)}
    )
    sub.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")
    raise SystemExit

final_rounds = int(np.clip(np.mean(best_iters), 1, 2000))

dtrain_full = lgb.Dataset(
    X_train,
    label=y_idx,
    categorical_feature=categorical_feature,
    free_raw_data=False,
)

final_booster = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_booster.predict(X_test)
test_pred_idx = np.argmax(test_pred, axis=1)
test_pred_labels = classes[test_pred_idx]

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "outcomes": test_pred_labels,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
