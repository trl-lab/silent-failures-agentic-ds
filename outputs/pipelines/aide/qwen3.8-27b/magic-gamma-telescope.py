import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    known_features = [
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
    ]

    feature_cols = [
        f for f in known_features if f in train.columns and f in test.columns
    ]

    if not feature_cols:
        candidate_cols = [
            c for c in train.columns if c not in {"class", "id"} and c in test.columns
        ]
        feature_cols = []
        for c in candidate_cols:
            converted = pd.to_numeric(train[c], errors="coerce")
            if converted.notna().any():
                feature_cols.append(c)

    target = train["class"]
    if not pd.api.types.is_numeric_dtype(target):
        labels = target.astype(str).str.strip().str.lower()
        y = labels.map(
            {
                "gamma": 1,
                "hadron": 0,
                "g": 1,
                "h": 0,
                "1": 1,
                "0": 0,
            }
        )
        if y.isna().any():
            y = (labels == "gamma").astype(int)
        y = y.astype(int)
    else:
        y = target.astype(int)
        if not set(y.unique()).issubset({0, 1}):
            y = (y > 0).astype(int)

    X_all = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    for c in feature_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        X_all[c] = X_all[c].replace([np.inf, -np.inf], np.nan)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "data_random_seed": 42,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y), start=1):
        X_train_raw = X_all.iloc[train_idx]
        X_val_raw = X_all.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        y_val = y.iloc[val_idx].to_numpy(dtype=int)

        imputer = SimpleImputer(strategy="median").fit(X_train_raw)
        X_train = imputer.transform(X_train_raw).astype(np.float32)
        X_val = imputer.transform(X_val_raw).astype(np.float32)
        X_test_fold = imputer.transform(X_test).astype(np.float32)

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            feature_names=feature_cols,
        )
        val_dataset = lgb.Dataset(
            X_val,
            label=y_val,
            feature_names=feature_cols,
            reference=train_dataset,
        )

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=2000,
            valid_sets=[val_dataset],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iteration = getattr(model, "best_iteration", None)
        best_iteration = (
            max(1, int(best_iteration)) if best_iteration is not None else 2000
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=best_iteration)
        test_preds += model.predict(X_test_fold, num_iteration=best_iteration)

        fold_scores.append(roc_auc_score(y_val, oof_preds[val_idx]))
        print(f"Fold {fold} ROC AUC: {fold_scores[-1]:.6f}")

    test_preds /= skf.n_splits
    test_preds = np.clip(test_preds, 0.0, 1.0)

    val_auc = roc_auc_score(y.to_numpy(dtype=int), oof_preds)
    print(f"Validation ROC AUC: {val_auc:.6f}")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "class": test_preds,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
