import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score

TARGET = "class"
ID = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"


def replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def add_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sensor_cols = [f"US{i}" for i in range(1, 25) if f"US{i}" in df.columns]

    if len(sensor_cols) < 24:
        return df

    us = df[sensor_cols].astype(float)

    df["us_min"] = us.min(axis=1)
    df["us_max"] = us.max(axis=1)
    df["us_mean"] = us.mean(axis=1)
    df["us_std"] = us.std(axis=1)
    df["us_range"] = df["us_max"] - df["us_min"]

    df["front"] = df["US1"]
    df["back"] = df["US13"]

    right_cols = [f"US{i}" for i in range(2, 13)]
    left_cols = [f"US{i}" for i in range(14, 25)]

    df["right_mean"] = us[right_cols].mean(axis=1)
    df["left_mean"] = us[left_cols].mean(axis=1)
    df["side_asym"] = df["right_mean"] - df["left_mean"]

    df["right_min"] = us[right_cols].min(axis=1)
    df["left_min"] = us[left_cols].min(axis=1)
    df["right_min_asym"] = df["right_min"] - df["left_min"]

    front_right_cols = [f"US{i}" for i in range(2, 7)]
    front_left_cols = [f"US{i}" for i in range(20, 25)]

    df["front_right_mean"] = us[front_right_cols].mean(axis=1)
    df["front_left_mean"] = us[front_left_cols].mean(axis=1)
    df["front_asym"] = df["front_right_mean"] - df["front_left_mean"]

    back_right_cols = [f"US{i}" for i in range(14, 19)]
    back_left_cols = [f"US{i}" for i in range(19, 24)]

    df["back_right_mean"] = us[back_right_cols].mean(axis=1)
    df["back_left_mean"] = us[back_left_cols].mean(axis=1)
    df["back_asym"] = df["back_right_mean"] - df["back_left_mean"]

    df["front_minus_min"] = df["US1"] - df["us_min"]
    df["back_minus_min"] = df["US13"] - df["us_min"]

    df["grad_front_right"] = df["US2"] - df["US6"]
    df["grad_front_left"] = df["US24"] - df["US20"]
    df["grad_back_right"] = df["US14"] - df["US18"]
    df["grad_back_left"] = df["US23"] - df["US19"]

    return df


def prepare_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    train = replace_inf(train)
    test = replace_inf(test)

    train = add_sensor_features(train)
    test = add_sensor_features(test)

    train = train.dropna(subset=[TARGET])
    y_raw = train[TARGET].copy()

    labels = np.unique(y_raw)
    if pd.api.types.is_numeric_dtype(labels):
        labels = labels.astype(int)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = y_raw.map(label_to_idx).astype(int)

    feature_cols = [
        col
        for col in train.columns
        if col not in (TARGET, ID) and pd.api.types.is_numeric_dtype(train[col])
    ]
    feature_cols = [col for col in feature_cols if col in test.columns]
    feature_cols = [col for col in feature_cols if train[col].notna().any()]

    X_train = train[feature_cols].astype(float).reset_index(drop=True)
    X_test = test[feature_cols].astype(float).reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X_train, y, X_test, labels


def main() -> None:
    train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    X_train, y, X_test, labels = prepare_data(train, test)

    params = {
        "objective": "multiclass",
        "num_class": len(labels),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbose": -1,
    }

    try:
        cv = TimeSeriesSplit(n_splits=5, gap=10)
        splits = list(cv.split(X_train))
        if len(splits) < 5:
            raise ValueError("Not enough temporal splits.")
    except Exception:
        cv = TimeSeriesSplit(n_splits=5)
        splits = list(cv.split(X_train))
        if len(splits) < 2:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(cv.split(X_train, y))

    accuracies = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr = X_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration if model.best_iteration > 0 else 1
        best_iterations.append(best_iter)

        val_pred = model.predict(X_va, num_iteration=best_iter)
        val_pred_idx = np.argmax(val_pred, axis=1)
        acc = accuracy_score(y_va, val_pred_idx)

        accuracies.append(acc)
        print(f"Fold {fold}: accuracy={acc:.4f}, best_iteration={best_iter}")

    cv_accuracy = float(np.mean(accuracies))
    cv_std = float(np.std(accuracies))
    print(f"5-fold time-series CV accuracy: {cv_accuracy:.4f} (+/- {cv_std:.4f})")

    final_rounds = int(np.clip(np.mean(best_iterations) * 1.1, 100, 1000))

    dtrain_full = lgb.Dataset(X_train, label=y)
    final_model = lgb.train(params, dtrain_full, num_boost_round=final_rounds)

    test_pred = final_model.predict(X_test)
    test_pred_idx = np.argmax(test_pred, axis=1)
    test_pred_labels = labels[test_pred_idx]

    os.makedirs(WORK_DIR, exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test), dtype=int),
            "class": test_pred_labels,
        }
    )
    submission_path = os.path.join(WORK_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


if __name__ == "__main__":
    main()
