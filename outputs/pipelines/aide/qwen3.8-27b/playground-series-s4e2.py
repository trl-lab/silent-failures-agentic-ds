import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

TARGET = "NObeyesdad"


def add_bmi(df):
    df = df.copy()
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    return df


def prepare(df, cat_cols, num_cols):
    out = df.copy()

    for c in cat_cols:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].astype(str)

    for c in num_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if "id" in test.columns:
        test_id = test["id"].astype(int)
    else:
        test_id = np.arange(len(test))

    X = train.drop(columns=[TARGET, "id"], errors="ignore")
    X_test = test.drop(columns=["id"], errors="ignore")

    X = add_bmi(X)
    X_test = add_bmi(X_test)

    y = train[TARGET].astype(str)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X = prepare(X, cat_cols, num_cols)
    X_test = prepare(X_test, cat_cols, num_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.empty(len(y), dtype=object)

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]

        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )

        X_train_cat = encoder.fit_transform(X_train_fold[cat_cols])
        X_val_cat = encoder.transform(X_val_fold[cat_cols])

        X_train_all = np.hstack(
            [X_train_fold[num_cols].to_numpy(dtype=np.float32), X_train_cat]
        )

        X_val_all = np.hstack(
            [X_val_fold[num_cols].to_numpy(dtype=np.float32), X_val_cat]
        )

        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train_all, y_train_fold)
        oof_pred[val_idx] = model.predict(X_val_all)

    cv_accuracy = accuracy_score(y, oof_pred)
    print(f"Validation accuracy: {cv_accuracy:.6f}")

    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32
    )

    X_train_cat = encoder.fit_transform(X[cat_cols])
    X_train_all = np.hstack([X[num_cols].to_numpy(dtype=np.float32), X_train_cat])

    X_test_cat = encoder.transform(X_test[cat_cols])
    X_test_all = np.hstack([X_test[num_cols].to_numpy(dtype=np.float32), X_test_cat])

    final_model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X_train_all, y)
    test_pred = final_model.predict(X_test_all)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": test_id, "NObeyesdad": test_pred})
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
