import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "loan_status"
RANDOM_STATE = 42
WORK_DIR = "./working"


def is_text_or_categorical_dtype(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.copy()
    test = test.copy()

    if target in train.columns:
        train = train.drop(columns=[target])
    if target in test.columns:
        test = test.drop(columns=[target])

    id_like_exact = {
        "id",
        "loan_id",
        "member_id",
        "id_orig",
        "loan_id_orig",
        "member_id_orig",
        "url",
        "desc",
        "title",
    }

    drop_cols = []
    for col in train.columns:
        col_lower = str(col).lower()
        if col_lower in id_like_exact:
            drop_cols.append(col)
        elif col_lower.endswith("_id") and col_lower != "zip_code":
            drop_cols.append(col)

    common_cols = [
        col for col in train.columns if col in test.columns and col not in drop_cols
    ]

    X = train[common_cols].copy()
    X_test = test[common_cols].copy()

    X = X.dropna(axis=1, how="all")
    X_test = X_test.loc[:, X.columns]

    constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols)
    X_test = X_test.loc[:, X.columns]

    missing_ratio = X.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.95].index
    X = X.drop(columns=high_missing_cols)
    X_test = X_test.loc[:, X.columns]

    categorical_cols = []
    drop_text_cols = []

    for col in X.columns:
        if is_text_or_categorical_dtype(X[col]):
            n_unique = X[col].nunique(dropna=True)
            if n_unique == 0:
                drop_text_cols.append(col)
            elif n_unique > 1000 or n_unique > 0.5 * len(X):
                drop_text_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    if drop_text_cols:
        X = X.drop(columns=drop_text_cols)
        X_test = X_test.loc[:, X.columns]

    for col in categorical_cols:
        X[col] = X[col].astype("category")
        train_categories = X[col].cat.categories
        X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

    X = X.dropna(axis=1, how="all")
    X_test = X_test.loc[:, X.columns]

    categorical_cols = [col for col in categorical_cols if col in X.columns]

    return X, X_test, categorical_cols


def make_binary_target(y_raw: pd.Series) -> pd.Series:
    y_clean = y_raw.dropna()

    if y_clean.nunique() == 2:
        classes = sorted(y_clean.unique().tolist(), key=str)
        positive_class = classes[-1]
        return (y_raw == positive_class).astype(int)

    if y_clean.nunique() > 2:
        value_counts = y_clean.value_counts()
        positive_class = value_counts.idxmin()
        return (y_raw == positive_class).astype(int)

    return y_raw.astype(int)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    y_raw = train[TARGET]
    valid_target_mask = y_raw.notna()
    train = train.loc[valid_target_mask].reset_index(drop=True)
    y_raw = y_raw.loc[valid_target_mask].reset_index(drop=True)

    X, X_test, categorical_cols = prepare_features(train, test, TARGET)
    y = make_binary_target(y_raw)

    if len(X) == 0 or y.nunique() < 2:
        print("CV AUC: 0.500000")
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "loan_status": np.full(len(test), 0.5, dtype=float),
            }
        )
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        return

    min_class_count = int(y.value_counts().min())
    n_splits = min(5, max(2, min_class_count)) if min_class_count >= 2 else 1

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    best_iterations = []
    fold_aucs = []

    if n_splits >= 2:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            dtrain = lgb.Dataset(
                X_train_fold,
                label=y_train_fold,
                categorical_feature=categorical_cols,
            )
            dval = lgb.Dataset(
                X_val_fold,
                label=y_val_fold,
                categorical_feature=categorical_cols,
            )

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dval],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
            )

            best_iter = (
                model.best_iteration if model.best_iteration > 0 else model.num_boosters
            )
            pred_val = model.predict(X_val_fold, num_iteration=best_iter)

            if len(np.unique(y_val_fold)) < 2:
                auc = 0.5
            else:
                auc = roc_auc_score(y_val_fold, pred_val)

            fold_aucs.append(auc)
            best_iterations.append(best_iter)
            print(f"Fold {fold} AUC: {auc:.6f}")

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    print(f"CV AUC: {mean_auc:.6f}")

    if best_iterations:
        final_rounds = int(np.median(best_iterations)) + 100
    else:
        final_rounds = 2000
    final_rounds = int(np.clip(final_rounds, 200, 5000))

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_cols,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_predictions = final_model.predict(X_test)
    test_predictions = np.clip(test_predictions, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "loan_status": test_predictions.astype(float),
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
