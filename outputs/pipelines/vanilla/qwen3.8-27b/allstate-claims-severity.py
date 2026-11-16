import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/test.parquet"
TARGET = "loss"
ID_COL = "id"
RANDOM_STATE = 42
MISSING_TOKEN = "__MISSING__"


def is_categorical_dtype(dtype):
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if dtype == np.dtype("O"):
        return True
    if dtype.name in ("string", "object"):
        return True
    return False


def prepare_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if ID_COL in train_df.columns:
        train_df = train_df.drop(columns=[ID_COL])
    if ID_COL in test_df.columns:
        test_df = test_df.drop(columns=[ID_COL])
    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    y_train = train_df[TARGET].copy()
    train_df = train_df.drop(columns=[TARGET])

    valid_target = y_train.notna()
    train_df = train_df.loc[valid_target]
    y_train = y_train.loc[valid_target]

    feature_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[feature_cols]
    test_df = test_df[feature_cols]

    cat_cols = []
    cont_cols = []
    for col in feature_cols:
        if is_categorical_dtype(train_df[col].dtype):
            cat_cols.append(col)
        else:
            cont_cols.append(col)

    keep_cols = []
    for col in feature_cols:
        series = train_df[col]
        if series.isna().all():
            continue
        if col in cat_cols:
            if series.dropna().nunique() <= 1:
                continue
        else:
            if series.nunique() <= 1:
                continue
        keep_cols.append(col)

    train_df = train_df[keep_cols]
    test_df = test_df[keep_cols]
    cat_cols = [c for c in cat_cols if c in keep_cols]
    cont_cols = [c for c in cont_cols if c in keep_cols]

    for col in cont_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        train_df[col] = np.where(np.isfinite(train_df[col]), train_df[col], np.nan)
        test_df[col] = np.where(np.isfinite(test_df[col]), test_df[col], np.nan)

    for col in cat_cols:
        train_series = train_df[col].where(train_df[col].notna(), MISSING_TOKEN).astype(str)
        test_series = test_df[col].where(test_df[col].notna(), MISSING_TOKEN).astype(str)

        categories = sorted(train_series.unique().tolist())
        if MISSING_TOKEN not in categories:
            categories.append(MISSING_TOKEN)

        train_df[col] = pd.Categorical(train_series, categories=categories)
        test_df[col] = pd.Categorical(test_series, categories=categories).fillna(MISSING_TOKEN)

    return train_df, test_df, y_train, cat_cols, cont_cols


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train_full, X_test, y_train_full, cat_cols, cont_cols = prepare_data(train_df, test_df)

    if X_train_full.shape[1] == 0 or len(y_train_full) == 0:
        fallback_pred = float(y_train_full.mean()) if len(y_train_full) > 0 else 0.0
        submission = pd.DataFrame({
            "id": np.arange(len(X_test), dtype=int),
            "loss": fallback_pred
        })
        submission.to_csv("predictions.csv", index=False)
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    max_cat_count = 0
    for col in cat_cols:
        if hasattr(X_train_full[col], "cat"):
            max_cat_count = max(max_cat_count, len(X_train_full[col].cat.categories))
    max_bin = max(255, max_cat_count + 1)

    model_params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        max_bin=max_bin,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    model = lgb.LGBMRegressor(n_estimators=3000, **model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_pred = model.predict(X_val)
    _ = mean_absolute_error(y_val, val_pred)

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration", None)
    if not best_iter or best_iter <= 0:
        best_iter = model.n_estimators

    final_model = lgb.LGBMRegressor(n_estimators=int(max(1, best_iter)), **model_params)
    final_model.fit(
        X_train_full,
        y_train_full,
        categorical_feature=cat_cols
    )

    test_pred = final_model.predict(X_test)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, None)

    submission = pd.DataFrame({
        "id": np.arange(len(X_test), dtype=int),
        "loss": test_pred
    })
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()