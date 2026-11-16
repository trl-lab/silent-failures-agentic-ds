import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/test.parquet"
TARGET = "y"
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    return train_df, test_df


def identify_columns(train_df, target):
    feature_cols = [c for c in train_df.columns if c != target]
    cat_cols = []
    for col in feature_cols:
        dtype = train_df[col].dtype
        if dtype == object or str(dtype).startswith("category"):
            cat_cols.append(col)
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


def fit_categorical_encoder(train_df, cat_cols):
    if not cat_cols:
        return None

    categories = {}
    for col in cat_cols:
        values = train_df[col].astype(str).fillna("missing")
        unique_values = sorted(values.unique().tolist())
        categories[col] = {value: idx for idx, value in enumerate(unique_values)}

    return categories


def transform_categoricals(df, cat_cols, categories):
    if not cat_cols or categories is None:
        return df

    out = df.copy()
    for col in cat_cols:
        mapping = categories.get(col, {})
        out[col] = (
            out[col]
            .astype(str)
            .fillna("missing")
            .map(mapping)
            .fillna(-1)
            .astype(float)
        )
    return out


def prepare_features(df, feature_cols, cat_cols, num_cols, categories):
    X = df.reindex(columns=feature_cols).copy()
    X = transform_categoricals(X, cat_cols, categories)

    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in num_cols:
        if col in X.columns and (col.lower().startswith("theta") or "angle" in col.lower()):
            X[f"{col}_sin"] = np.sin(X[col])
            X[f"{col}_cos"] = np.cos(X[col])

    return X


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_model(n_estimators):
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df, test_df = load_data()

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET])

    feature_cols, cat_cols, num_cols = identify_columns(train_df, TARGET)
    test_df = test_df.reindex(columns=feature_cols)

    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    val_categories = fit_categorical_encoder(train_part, cat_cols)
    X_train = prepare_features(train_part, feature_cols, cat_cols, num_cols, val_categories)
    X_val = prepare_features(val_part, feature_cols, cat_cols, num_cols, val_categories)
    y_train = train_part[TARGET].astype(float)
    y_val = val_part[TARGET].astype(float)

    model = make_model(n_estimators=5000)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_val)
    print(f"Validation RMSE: {rmse(y_val, val_pred):.6f}")

    best_iter = max(1, int(model.best_iteration_))

    full_categories = fit_categorical_encoder(train_df, cat_cols)
    X_full = prepare_features(train_df, feature_cols, cat_cols, num_cols, full_categories)
    X_test = prepare_features(test_df, feature_cols, cat_cols, num_cols, full_categories)
    y_full = train_df[TARGET].astype(float)

    final_model = make_model(n_estimators=best_iter)
    final_model.fit(X_full, y_full)

    predictions = final_model.predict(X_test)
    output = pd.DataFrame({"id": np.arange(len(test_df), dtype=int), "y": predictions})
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()