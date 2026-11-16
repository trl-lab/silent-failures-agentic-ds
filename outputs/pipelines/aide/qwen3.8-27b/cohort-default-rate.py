import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "Average or Greater than 30"
SEED = 42
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def clean_target(y: pd.Series) -> pd.Series:
    """Convert the target to a binary 0/1 series."""
    y = y.copy()

    if pd.api.types.is_numeric_dtype(y):
        vals = y.dropna().unique()
        if len(vals) == 0:
            return y.astype(int)
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            return y.astype(int)
        if len(vals) == 2:
            positive = max(vals)
            return (y == positive).astype(int)
        return (y > y.median()).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    pos_keys = (
        "1",
        "yes",
        "true",
        "t",
        "greater",
        "30",
        "default",
        "bad",
        "positive",
        "high",
    )
    pos_vals = [v for v in y_str.unique() if any(k in v for k in pos_keys)]
    neg_vals = [v for v in y_str.unique() if not any(k in v for k in pos_keys)]

    if pos_vals and neg_vals:
        return y_str.isin(pos_vals).astype(int)

    if "average" in " ".join(y_str.unique()):
        return (y_str != "average").astype(int)

    codes, uniques = pd.factorize(y_str, sort=False)
    if len(uniques) == 2:
        return (codes == 1).astype(int)
    return (codes == codes.max()).astype(int)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning and type conversion for LightGBM."""
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    id_like = {"id", "index", "row_id", "rowid", "identifier"}
    df = df.drop(
        columns=[c for c in df.columns if c.lower() in id_like], errors="ignore"
    )

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].replace(
            ["NA", "N/A", "null", "None", "?", "missing", ""], np.nan
        )

    df = df.loc[:, df.notna().any()]

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("int64")
        elif df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_na = df[col].notna().sum()
            if non_na > 0 and converted.notna().sum() >= 0.9 * non_na:
                df[col] = converted
            else:
                df[col] = df[col].astype("category")
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            pass
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[:, [col for col in df.columns if df[col].nunique(dropna=True) > 1]]
    return df


def main() -> None:
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    y = train[TARGET].copy()
    train = train.drop(columns=[TARGET])

    valid_rows = y.notna()
    y = y[valid_rows]
    train = train.loc[valid_rows]
    y = clean_target(y)

    if y.nunique() < 2:
        print("CV AUC: 0.500000")
        submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: 0.5})
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        return

    X_train = prepare_features(train)
    X_test = prepare_features(test)

    features = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[features]
    X_test = X_test[features]

    if X_train.shape[1] == 0:
        print("CV AUC: 0.500000")
        submission = pd.DataFrame({"id": np.arange(len(test)), TARGET: 0.5})
        submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        return

    categorical_cols = []
    for col in features:
        is_cat_train = isinstance(X_train[col].dtype, pd.CategoricalDtype)
        is_cat_test = isinstance(X_test[col].dtype, pd.CategoricalDtype)

        if is_cat_train or is_cat_test:
            train_cats = X_train[col].dropna().unique()
            X_train[col] = (
                X_train[col].astype("category").cat.set_categories(train_cats)
            )
            X_test[col] = X_test[col].astype("category").cat.set_categories(train_cats)
            categorical_cols.append(col)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

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
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": SEED,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), start=1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        if len(np.unique(y_val)) < 2:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y_val, val_pred))

        best_iterations.append(model.best_iteration)
        print(f"Fold {fold} AUC: {aucs[-1]:.6f}")

    cv_auc = float(np.mean(aucs))
    print(f"CV AUC: {cv_auc:.6f}")

    final_rounds = max(1, int(np.mean(best_iterations))) if best_iterations else 1000

    dtrain_full = lgb.Dataset(
        X_train,
        label=y,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred)),
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
