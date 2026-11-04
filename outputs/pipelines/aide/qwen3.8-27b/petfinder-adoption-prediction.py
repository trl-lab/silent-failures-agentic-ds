import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def qwk(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true = np.clip(y_true, 0, 4)
    y_pred = np.nan_to_num(y_pred, nan=2.0)
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)

    t = np.bincount(y_true, minlength=5).astype(float)
    p = np.bincount(y_pred, minlength=5).astype(float)

    h = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            h[i, j] = 1.0 - abs(i - j) / 4.0

    s = np.outer(t, p)
    if s.sum() == 0:
        return 0.0

    return 1.0 - (s * h).sum() / s.sum()


def add_features(df):
    df = df.copy()
    df["Name_len"] = df["Name"].fillna("").astype(str).str.len()
    df["Desc_len"] = df["Description"].fillna("").astype(str).str.len()
    df["Desc_word_count"] = (
        df["Description"].fillna("").astype(str).str.split().str.len()
    )
    df["Breed2_present"] = (df["Breed2"] > 0).astype(int)
    df["Color2_present"] = (df["Color2"] > 0).astype(int)
    df["Color3_present"] = (df["Color3"] > 0).astype(int)
    df["Age_is_255"] = (df["Age"] == 255).astype(int)
    return df


def prepare(df, numeric_cols, categorical_cols):
    df = df.copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in categorical_cols:
        df[c] = df[c].fillna(-1).astype(int)
    return df


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = add_features(train)
    test = add_features(test)

    numeric_cols = [
        "Age",
        "Fee",
        "PhotoAmt",
        "Quantity",
        "VideoAmt",
        "Name_len",
        "Desc_len",
        "Desc_word_count",
        "Breed2_present",
        "Color2_present",
        "Color3_present",
        "Age_is_255",
    ]

    categorical_cols = [
        "Breed1",
        "Breed2",
        "Color1",
        "Color2",
        "Color3",
        "Dewormed",
        "FurLength",
        "Gender",
        "Health",
        "MaturitySize",
        "State",
        "Sterilized",
        "Type",
        "Vaccinated",
    ]

    feature_cols = numeric_cols + categorical_cols

    train = prepare(train, numeric_cols, categorical_cols)
    test = prepare(test, numeric_cols, categorical_cols)

    y = train["AdoptionSpeed"].astype(int)
    X = train[feature_cols]
    X_test = test[feature_cols]

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype=float)
    best_iters = []

    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_cols)
        valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        best_iters.append(model.best_iteration)
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof[va_idx] = pred

    cv_qwk = qwk(y.values, oof)
    print(f"CV QWK: {cv_qwk:.5f}")

    final_rounds = max(1, int(np.mean(best_iters))) if best_iters else 1000
    final_train_set = lgb.Dataset(X, label=y, categorical_feature=categorical_cols)
    final_model = lgb.train(params, final_train_set, num_boost_round=final_rounds)

    pred_test = final_model.predict(X_test)
    pred_test = np.nan_to_num(pred_test, nan=2.0)
    pred_test = np.clip(np.round(pred_test), 0, 4).astype(int)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "AdoptionSpeed": pred_test,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
