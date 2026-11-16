import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "class" in df.columns:
        df = df.drop(columns=["class"])

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Period" in df.columns:
        period = df["Period"]
        if period.max() <= 1.0:
            period_frac = period
        else:
            period_frac = (period - 1) / 48.0
        df["Period_sin"] = np.sin(2 * np.pi * period_frac)
        df["Period_cos"] = np.cos(2 * np.pi * period_frac)

    if "Day" in df.columns:
        day = df["Day"]
        if day.max() <= 1.0:
            day_frac = day
        else:
            day_frac = (day - 1) / 7.0
        df["Day_sin"] = np.sin(2 * np.pi * day_frac)
        df["Day_cos"] = np.cos(2 * np.pi * day_frac)

    if "NSWprice" in df.columns and "NSWdemand" in df.columns:
        df["NSW_price_demand"] = df["NSWprice"] * df["NSWdemand"]

    if "VICprice" in df.columns and "VICdemand" in df.columns:
        df["VIC_price_demand"] = df["VICprice"] * df["VICdemand"]

    if "transfer" in df.columns:
        df["transfer_centered"] = df["transfer"] - 0.5

    return df


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if "class" not in train.columns:
        raise ValueError("Target column 'class' not found in train data.")

    train = train.dropna(subset=["class"]).reset_index(drop=True)

    if len(train) == 0:
        print("Validation accuracy: 0.000000")
        submission = pd.DataFrame({"id": np.arange(len(test)), "class": []})
        submission.to_csv("./working/submission.csv", index=False)
        return

    if len(train) <= 1:
        only_label = train["class"].iloc[0]
        print("Validation accuracy: 1.000000")
        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "class": np.full(len(test), only_label)}
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    sort_cols = [c for c in ["Date", "Period"] if c in train.columns]
    if sort_cols:
        train = train.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    y_full = train["class"].reset_index(drop=True)
    X_full = make_features(train)

    val_size = min(max(1, int(0.2 * len(X_full))), len(X_full) - 1)

    X_train = X_full.iloc[:-val_size].copy()
    y_train = y_full.iloc[:-val_size].copy()
    X_val = X_full.iloc[-val_size:].copy()
    y_val = y_full.iloc[-val_size:].copy()

    labels = np.unique(y_full)

    if len(labels) == 1:
        print("Validation accuracy: 1.000000")
        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "class": np.full(len(test), labels[0])}
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for i, lab in enumerate(labels)}

    y_train_enc = y_train.map(label_to_idx).astype(int)
    y_val_enc = y_val.map(label_to_idx).astype(int)
    y_full_enc = y_full.map(label_to_idx).astype(int)

    is_binary = len(labels) == 2

    params = {
        "objective": "binary" if is_binary else "multiclass",
        "metric": "binary_logloss" if is_binary else "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "seed": 42,
        "verbose": -1,
    }

    if not is_binary:
        params["num_class"] = len(labels)

    dtrain = lgb.Dataset(X_train, label=y_train_enc, feature_name=list(X_train.columns))
    dval = lgb.Dataset(X_val, label=y_val_enc, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = model.best_iteration
    if best_iter is None or best_iter <= 0:
        best_iter = model.num_trees()
    if best_iter is None or best_iter <= 0:
        best_iter = 3000
    best_iter = int(best_iter)

    val_prob = model.predict(X_val, num_iteration=best_iter)

    if is_binary:
        min_p = float(np.min(val_prob))
        max_p = float(np.max(val_prob))
        quantiles = np.quantile(val_prob, np.linspace(0.0, 1.0, 200))
        candidate_thresholds = np.unique(
            np.concatenate([quantiles, [min_p - 1e-6, max_p + 1e-6, 0.5]])
        )

        best_threshold = 0.5
        val_acc = -1.0
        val_pred = None

        for threshold in candidate_thresholds:
            pred = (val_prob >= threshold).astype(int)
            acc = accuracy_score(y_val_enc, pred)
            if acc > val_acc:
                val_acc = acc
                best_threshold = threshold
                val_pred = pred

        if val_pred is None:
            val_pred = (val_prob >= best_threshold).astype(int)
    else:
        val_pred = np.argmax(val_prob, axis=1)
        val_acc = accuracy_score(y_val_enc, val_pred)

    print(f"Validation accuracy: {val_acc:.6f}")

    dtrain_full = lgb.Dataset(
        X_full, label=y_full_enc, feature_name=list(X_full.columns)
    )
    model_full = lgb.train(
        params,
        dtrain_full,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(-1)],
    )

    X_test = make_features(test)
    X_test = X_test.reindex(columns=X_full.columns)

    if len(test) == 0:
        test_pred_labels = np.array([], dtype=object)
    else:
        test_prob = model_full.predict(X_test, num_iteration=best_iter)

        if is_binary:
            test_pred = (test_prob >= best_threshold).astype(int)
        else:
            test_pred = np.argmax(test_prob, axis=1)

        test_pred_labels = np.array([idx_to_label[i] for i in test_pred])

    submission = pd.DataFrame({"id": np.arange(len(test)), "class": test_pred_labels})
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
