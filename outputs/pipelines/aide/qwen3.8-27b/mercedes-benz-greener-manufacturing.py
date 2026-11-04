import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "y"
id_cols = {"ID", "id"}

train = train.dropna(subset=[target]).reset_index(drop=True)

feature_candidates = [
    col
    for col in train.columns
    if col != target and col not in id_cols and col in test.columns
]

categorical_features = []

for col in feature_candidates:
    if train[col].dtype == bool or test[col].dtype == bool:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        continue

    if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(
        test[col]
    ):
        continue

    try:
        train_num = pd.to_numeric(train[col], errors="coerce")
        test_num = pd.to_numeric(test[col], errors="coerce")
        if train_num.notna().sum() > 0:
            train[col] = train_num
            test[col] = test_num
            continue
    except Exception:
        pass

    train_categories = pd.Categorical(train[col]).categories
    if len(train_categories) == 0:
        train[col] = pd.Series(np.nan, index=train.index, dtype=float)
        test[col] = pd.Series(np.nan, index=test.index, dtype=float)
    else:
        train[col] = train[col].astype("category")
        test[col] = pd.Categorical(test[col], categories=train_categories)
        categorical_features.append(col)

features = [
    col
    for col in feature_candidates
    if train[col].notna().any() and train[col].nunique(dropna=True) > 1
]
categorical_features = [col for col in categorical_features if col in features]

if len(features) == 0:
    mean_y = train[target].mean()
    oof_preds = np.full(len(train), mean_y, dtype=float)
    test_preds = np.full(len(test), mean_y, dtype=float)
    val_r2 = r2_score(train[target], oof_preds)
    print(f"Validation R2: {val_r2:.6f}")
    submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), "y": test_preds})
    submission.to_csv("./working/submission.csv", index=False)
else:
    X = train[features].copy()
    y = train[target].astype(float).copy()
    X_test = test[features].copy()

    numeric_cols = [col for col in features if pd.api.types.is_numeric_dtype(X[col])]
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": 42,
        "num_threads": -1,
        "verbosity": -1,
    }

    num_boost_round = 5000
    early_stopping_rounds = 50
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train), dtype=float)
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        dtrain = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_features=categorical_features,
        )
        dval = lgb.Dataset(
            X_val_fold,
            label=y_val_fold,
            reference=dtrain,
            categorical_features=categorical_features,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = (
            model.best_iteration
            if model.best_iteration is not None
            else num_boost_round
        )
        best_iterations.append(best_iter)
        oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=best_iter)

    val_r2 = r2_score(y, oof_preds)
    print(f"Validation R2: {val_r2:.6f}")

    final_rounds = max(1, int(np.median(best_iterations)))
    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_features=categorical_features,
    )
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )
    test_preds = final_model.predict(X_test, num_iteration=final_rounds)

    submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), "y": test_preds})
    submission.to_csv("./working/submission.csv", index=False)
