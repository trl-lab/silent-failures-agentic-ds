import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/test.parquet"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42
MISSING_CAT = "__missing__"


def clean_frame(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    df = df.copy()

    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] < -365 * 30, "DAYS_EMPLOYED"] = np.nan

    return df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y_full = train_df[TARGET].copy()
    X_full = train_df.drop(columns=[TARGET]).copy()
    X_test = test_df.drop(columns=[TARGET], errors="ignore").copy()

    for col in X_full.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan

    X_test = X_test[X_full.columns].copy()

    constant_cols = [
        col
        for col in X_full.columns
        if X_full[col].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        X_full = X_full.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols, errors="ignore")

    categorical_cols = []
    for col in X_full.columns:
        dtype = X_full[col].dtype
        if dtype == object or dtype.name == "category" or str(dtype).startswith("string"):
            categorical_cols.append(col)

    for col in categorical_cols:
        X_full[col] = (
            X_full[col]
            .astype(object)
            .where(X_full[col].notna(), MISSING_CAT)
            .astype(str)
        )
        X_test[col] = (
            X_test[col]
            .astype(object)
            .where(X_test[col].notna(), MISSING_CAT)
            .astype(str)
        )

    numeric_cols = [col for col in X_full.columns if col not in categorical_cols]
    for col in numeric_cols:
        X_full[col] = pd.to_numeric(X_full[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X_full, X_test, y_full, categorical_cols


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = clean_frame(train_df, drop_id=True)
    test_df = clean_frame(test_df, drop_id=True)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")

    X_full, X_test, y_full, categorical_cols = prepare_features(train_df, test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=RANDOM_STATE,
    )

    base_params = dict(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        cat_features=categorical_cols,
    )

    model = CatBoostClassifier(
        **base_params,
        eval_metric="AUC",
        early_stopping_rounds=100,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=0,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    if len(np.unique(y_val)) > 1:
        val_auc = roc_auc_score(y_val, val_proba)
    else:
        val_auc = 0.5
    print(f"Validation AUC: {val_auc:.6f}")

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None:
        final_iterations = base_params["iterations"]
    else:
        final_iterations = max(10, int(best_iteration) + 1)

    final_params = dict(base_params)
    final_params["iterations"] = final_iterations

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X_full, y_full, verbose=0)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_proba = np.clip(test_proba, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_proba), dtype=int),
            "TARGET": test_proba,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()