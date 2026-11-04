import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/test.parquet"


def encode_categorical(train_col: pd.Series, test_col: pd.Series):
    train_col = train_col.astype("object")
    test_col = test_col.astype("object")

    categories = train_col.dropna().unique()
    if len(categories) == 0:
        train_enc = pd.Series(-1, index=train_col.index, dtype=np.int32)
        test_enc = pd.Series(-1, index=test_col.index, dtype=np.int32)
        return train_enc, test_enc

    train_enc = pd.Categorical(train_col, categories=categories).codes.astype(np.int32)
    test_enc = pd.Categorical(test_col, categories=categories).codes.astype(np.int32)
    return train_enc, test_enc


def prepare_data(train_path: str, test_path: str):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    y = train_df["target"].astype(int)

    train_df = train_df.drop(columns=["id", "target"], errors="ignore")
    test_df = test_df.drop(columns=["id", "target"], errors="ignore")

    common_cols = [col for col in train_df.columns if col in test_df.columns]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    categorical_cols = []
    for col in common_cols:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            categorical_cols.append(col)

    for col in categorical_cols:
        train_df[col], test_df[col] = encode_categorical(train_df[col], test_df[col])

    for col in common_cols:
        if col not in categorical_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(np.float32)
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)

    return train_df, test_df, y, categorical_cols


def write_predictions(test_len: int, predictions: np.ndarray):
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions = np.clip(predictions, 0.0, 1.0)
    out = pd.DataFrame({
        "id": np.arange(test_len, dtype=int),
        "target": predictions
    })
    out.to_csv("predictions.csv", index=False)


def main():
    X, X_test, y, categorical_cols = prepare_data(TRAIN_PATH, TEST_PATH)

    if len(X) == 0 or len(X_test) == 0:
        write_predictions(len(X_test), np.full(len(X_test), 0.5))
        return

    if y.nunique() < 2:
        write_predictions(len(X_test), np.full(len(X_test), 0.5))
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.08,
        "n_estimators": 400,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**base_params)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=50,
            categorical_feature=categorical_cols,
        )
    except Exception:
        model.fit(X_train, y_train, categorical_feature=categorical_cols)

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        final_n_estimators = 300
    else:
        final_n_estimators = int(min(max(best_iteration + 1, 100), 400))

    final_params = base_params.copy()
    final_params["n_estimators"] = final_n_estimators

    final_model = lgb.LGBMClassifier(**final_params)
    try:
        final_model.fit(X, y, categorical_feature=categorical_cols)
        predictions = final_model.predict_proba(X_test)[:, 1]
    except Exception:
        predictions = model.predict_proba(X_test)[:, 1]

    write_predictions(len(X_test), predictions)


if __name__ == "__main__":
    main()