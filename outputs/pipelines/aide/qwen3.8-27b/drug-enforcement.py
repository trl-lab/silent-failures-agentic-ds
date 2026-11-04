import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "classification"


def prepare_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    train_raw = train[feature_cols].copy()
    test_raw = test.reindex(columns=feature_cols)

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        s = train_raw[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            numeric_cols.append(col)
        else:
            if s.dtype == object or str(s.dtype).startswith("string"):
                converted = pd.to_numeric(s, errors="coerce")
                non_null = int(s.notna().sum())
                if non_null > 0 and converted.notna().sum() / non_null > 0.95:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)

    X_train = pd.DataFrame(index=train.index)
    X_test = pd.DataFrame(index=test.index)

    for col in feature_cols:
        if col in numeric_cols:
            if pd.api.types.is_datetime64_any_dtype(train_raw[col]):
                vals = train_raw[col].astype("int64")
                vals = vals.where(train_raw[col].notna(), np.nan)
                X_train[col] = vals
            else:
                X_train[col] = pd.to_numeric(train_raw[col], errors="coerce")

            if col in test_raw.columns:
                if pd.api.types.is_datetime64_any_dtype(test_raw[col]):
                    vals = test_raw[col].astype("int64")
                    vals = vals.where(test_raw[col].notna(), np.nan)
                    X_test[col] = vals
                else:
                    X_test[col] = pd.to_numeric(test_raw[col], errors="coerce")
            else:
                X_test[col] = np.nan
        else:
            X_train[col] = train_raw[col].astype("category")
            if col in test_raw.columns:
                train_cats = pd.Categorical(train_raw[col]).categories
                if len(train_cats) == 0:
                    X_test[col] = np.nan
                else:
                    X_test[col] = pd.Categorical(test_raw[col], categories=train_cats)
            else:
                X_test[col] = np.nan

    for df in (X_train, X_test):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    drop_cols = []
    for col in X_train.columns:
        if X_train[col].isna().all() or X_train[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols, errors="ignore")

    return X_train, X_test


def predict_codes(model, X, n_classes, label_encoder, num_iteration=None):
    if num_iteration is None:
        num_iteration = getattr(model, "best_iteration", 0)
        if num_iteration <= 0:
            num_iteration = None

    proba = model.predict(X, num_iteration=num_iteration)

    if n_classes == 2:
        codes = (proba > 0.5).astype(int)
    else:
        codes = np.argmax(proba, axis=1)

    labels = label_encoder.inverse_transform(codes)
    return labels


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train[TARGET] = train[TARGET].replace("", np.nan)
    valid_target_mask = train[TARGET].notna()
    train_model = train.loc[valid_target_mask].copy()

    feature_cols = [c for c in train.columns if c != TARGET and c.lower() != "id"]
    X_train_full, X_test = prepare_features(train, test, feature_cols)
    X_train = X_train_full.loc[valid_target_mask].copy()

    y_raw = train_model[TARGET]
    target_dtype = y_raw.dtype

    if len(X_train) == 0 or y_raw.nunique() == 0:
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "classification": (
                    y_raw.mode().iloc[0] if not y_raw.mode().empty else np.nan
                ),
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        print(0.0)
        return

    le = LabelEncoder()
    y_codes = le.fit_transform(y_raw.astype(str))
    n_classes = len(le.classes_)

    if n_classes == 1 or X_train.shape[1] == 0:
        most_common_code = 0
        cv_accuracy = float((y_codes == most_common_code).mean())
        pred_labels = le.inverse_transform(np.full(len(test), most_common_code))
        if pd.api.types.is_numeric_dtype(target_dtype):
            pred_labels = pd.Series(pred_labels).astype(target_dtype).values
        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "classification": pred_labels}
        )
        submission.to_csv("./working/submission.csv", index=False)
        print(cv_accuracy)
        return

    params = {
        "objective": "binary" if n_classes == 2 else "multiclass",
        "metric": "binary_logloss" if n_classes == 2 else "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": RANDOM_STATE,
        "num_threads": -1,
    }
    if n_classes > 2:
        params["num_class"] = n_classes

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_codes), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_codes[tr_idx], y_codes[va_idx]

        train_dataset = lgb.Dataset(
            X_tr, label=y_tr, categorical_feature="auto", free_raw_data=False
        )
        valid_dataset = lgb.Dataset(
            X_va,
            label=y_va,
            reference=train_dataset,
            categorical_feature="auto",
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=10000,
            valid_sets=[valid_dataset],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        pred_va = predict_codes(
            model, X_va, n_classes, le, num_iteration=model.best_iteration
        )
        acc = accuracy_score(y_va, pred_va)
        fold_accuracies.append(acc)
        best_iterations.append(model.best_iteration)

    cv_accuracy = float(np.mean(fold_accuracies))

    final_num_boost = (
        max(1, int(np.median(best_iterations))) if best_iterations else 500
    )
    final_dataset = lgb.Dataset(
        X_train, label=y_codes, categorical_feature="auto", free_raw_data=False
    )
    final_model = lgb.train(
        params,
        final_dataset,
        num_boost_round=final_num_boost,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = predict_codes(
        final_model, X_test, n_classes, le, num_iteration=final_num_boost
    )
    if pd.api.types.is_numeric_dtype(target_dtype):
        test_pred = pd.Series(test_pred).astype(target_dtype).values

    submission = pd.DataFrame({"id": np.arange(len(test)), "classification": test_pred})
    submission.to_csv("./working/submission.csv", index=False)

    print(cv_accuracy)


if __name__ == "__main__":
    main()
