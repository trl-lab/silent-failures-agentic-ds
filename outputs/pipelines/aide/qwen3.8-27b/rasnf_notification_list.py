import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

TARGET = "num_countries_affected"
SEED = 42


def preprocess(df, features):
    df = df.copy()
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[features]

    for col in features:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            mask = df[col].notna()
            df[col] = df[col].astype("int64")
            df[col] = df[col].where(mask, np.nan)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            mask = df[col].notna()
            df[col] = df[col].astype("int64")
            df[col] = df[col].where(mask, np.nan)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("int8")
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(
            df[col]
        ):
            if df[col].notna().sum() > 0:
                tmp = df[col].astype(object)
                numeric = pd.to_numeric(tmp, errors="coerce")
                if numeric.notna().sum() > 0.8 * df[col].notna().sum():
                    df[col] = numeric
                else:
                    df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype("category")
        else:
            if not pd.api.types.is_numeric_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.8 * df[col].notna().sum():
                    df[col] = converted
                else:
                    df[col] = df[col].astype("category")

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def save_submission(pred, path="./working/submission.csv"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pred = np.asarray(pred).reshape(-1)
    pd.DataFrame({"id": np.arange(len(pred)), "num_countries_affected": pred}).to_csv(
        path, index=False
    )


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if "id" in train.columns:
        train = train.drop(columns=["id"])
    if "id" in test.columns:
        test = test.drop(columns=["id"])

    if TARGET not in train.columns:
        print("Validation ROC AUC: 0.500000")
        save_submission(np.full(len(test), 0.5))
        return

    y_raw = train[TARGET]
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    y_raw = y_raw.copy()

    X_train = train.drop(columns=[TARGET])
    X_test = test.drop(columns=[TARGET], errors="ignore")

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    features = list(dict.fromkeys(X_train.columns))
    X_train = preprocess(X_train, features)
    X_test = preprocess(X_test, features)

    y_series = pd.to_numeric(y_raw, errors="coerce")
    valid = y_series.notna()
    y_series = y_series[valid].reset_index(drop=True)
    X_train = X_train.loc[valid].reset_index(drop=True)

    drop_cols = []
    for col in X_train.columns:
        if X_train[col].isna().all() or X_train[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols, errors="ignore")

    feature_names = X_train.columns.tolist()
    for col in feature_names:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        elif pd.api.types.is_categorical_dtype(X_train[col]):
            X_test[col] = (
                X_test[col]
                .astype(object)
                .astype("category", categories=X_train[col].cat.categories)
            )
        else:
            X_test[col] = X_test[col].astype("category")
    X_test = X_test[feature_names]

    if X_train.shape[1] == 0 or len(X_train) == 0:
        print("Validation ROC AUC: 0.500000")
        save_submission(np.full(len(X_test), 0.5))
        return

    y_values = y_series.values
    unique = np.unique(y_values)

    if len(unique) == 1:
        print("Validation ROC AUC: 0.500000")
        save_submission(np.full(len(X_test), 0.5))
        return
    elif len(unique) == 2:
        if np.isin(0.0, unique) and max(unique) > 0:
            y = (y_values > 0).astype(int)
        else:
            y = (y_values == max(unique)).astype(int)
    else:
        y = (y_values > 0).astype(int)

    if len(np.unique(y)) == 1:
        print("Validation ROC AUC: 0.500000")
        save_submission(np.full(len(X_test), 0.5))
        return

    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    cat_arg = cat_cols if cat_cols else None

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbosity": -1,
    }

    min_class_count = int(min(np.bincount(y)))
    if min_class_count >= 5 and len(y) >= 20:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(skf.split(X_train, y))
    elif min_class_count >= 2 and len(y) >= 10:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        folds = list(skf.split(X_train, y))
    else:
        if min_class_count >= 2:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y
            )
        else:
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=SEED
            )
        folds = [(tr_idx, va_idx)]

    aucs = []
    best_iters = []

    for tr_idx, va_idx in folds:
        dtrain = lgb.Dataset(
            X_train.iloc[tr_idx],
            label=y[tr_idx],
            feature_name=feature_names,
            categorical_feature=cat_arg,
        )
        dvalid = lgb.Dataset(
            X_train.iloc[va_idx],
            label=y[va_idx],
            reference=dtrain,
            feature_name=feature_names,
            categorical_feature=cat_arg,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_it = getattr(model, "best_iteration", 0) or 0
        if best_it <= 0:
            best_it = 1000
        best_iters.append(best_it)

        pred = np.asarray(
            model.predict(X_train.iloc[va_idx], num_iteration=best_it)
        ).reshape(-1)
        pred = np.clip(np.nan_to_num(pred, nan=0.5), 0.0, 1.0)

        if len(np.unique(y[va_idx])) > 1:
            aucs.append(roc_auc_score(y[va_idx], pred))
        else:
            aucs.append(0.5)

    mean_auc = float(np.mean(aucs))
    print(f"Validation ROC AUC ({len(folds)}-fold): {mean_auc:.6f}")

    if len(X_test) == 0:
        save_submission(np.array([]))
        return

    num_rounds = int(max(100, np.mean(best_iters))) if best_iters else 500

    final_train = lgb.Dataset(
        X_train, label=y, feature_name=feature_names, categorical_feature=cat_arg
    )
    final_model = lgb.train(params, final_train, num_boost_round=num_rounds)

    dtest = lgb.Dataset(
        X_test,
        reference=final_train,
        feature_name=feature_names,
        categorical_feature=cat_arg,
    )
    test_pred = np.asarray(
        final_model.predict(dtest, num_iteration=num_rounds)
    ).reshape(-1)
    test_pred = np.clip(np.nan_to_num(test_pred, nan=0.5), 0.0, 1.0)

    save_submission(test_pred)


if __name__ == "__main__":
    main()
