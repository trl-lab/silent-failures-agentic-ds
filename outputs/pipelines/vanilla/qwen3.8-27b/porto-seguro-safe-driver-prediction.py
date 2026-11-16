import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/porto-seguro-safe-driver-prediction/data/test.parquet"
TARGET = "target"
ID = "id"
RANDOM_STATE = 42


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare features.

    - Drops id and target if present.
    - Replaces -1 with NaN for numeric columns, which is the missing-value
      convention in this dataset.
    - Handles any object/category columns by filling missing values and
      converting them to category dtype.
    - Downcasts numeric columns to float32 for memory efficiency.
    """
    df = df.copy()

    for col in (ID, TARGET):
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].replace(-1, np.nan)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("missing").astype("category")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].astype(np.float32)

    return df


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized Gini coefficient for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def gini(actual: np.ndarray, pred: np.ndarray) -> float:
        n = len(actual)
        if n == 0 or np.sum(actual) == 0:
            return 0.0
        ap = np.argsort(np.argsort(pred))
        au = np.cumsum(actual[ap]) / np.sum(actual)
        return 2.0 * np.sum((np.arange(1, n + 1) - (n + 1) / 2.0) * (au - np.arange(1, n + 1) / n)) / n

    return gini(y_true, y_pred) / gini(y_true, y_true)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = train_df[TARGET].astype(int)
    X = prepare_features(train_df)
    X_test = prepare_features(test_df)

    # Ensure train and test have the same feature columns.
    X_test = X_test[X.columns]

    # Validation split to estimate generalization performance.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(n_estimators=3000, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Optional validation metric estimate.
    val_pred = model.predict_proba(X_val)[:, 1]
    _ = normalized_gini(y_val, val_pred)

    # Refit on the full training set using the selected number of iterations.
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = model.n_estimators

    final_model = lgb.LGBMClassifier(n_estimators=best_iteration, **params)
    final_model.fit(X, y)

    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "target": test_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()