import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/global-dams-database/data/test.parquet"
TARGET = "Res_capacityM3"
RANDOM_STATE = 42


def clean_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def looks_numeric(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return converted.notna().mean() >= 0.8


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    common_cols = [c for c in train_df.columns if c in test_df.columns and c != TARGET]
    train_df = train_df[common_cols].reset_index(drop=True)
    test_df = test_df[common_cols].reset_index(drop=True)

    if TARGET not in train_df.columns:
        y = pd.Series(np.nan, index=train_df.index)
    else:
        y = train_df[TARGET].copy()
        train_df = train_df.drop(columns=[TARGET])

    y = clean_numeric_series(y).clip(lower=0.0)
    valid = y.notna()
    train_df = train_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        return train_df.iloc[0:0], test_df, y, []

    numeric_cols = []
    categorical_candidates = []

    for col in train_df.columns:
        if looks_numeric(train_df[col]):
            train_df[col] = clean_numeric_series(train_df[col])
            test_df[col] = clean_numeric_series(test_df[col])
            numeric_cols.append(col)
        else:
            categorical_candidates.append(col)

    keep_numeric = []
    for col in numeric_cols:
        s = train_df[col]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        if s.isna().mean() > 0.95:
            continue
        keep_numeric.append(col)

    keep_categorical = []
    max_cardinality = 1000

    for col in categorical_candidates:
        s = train_df[col].astype("string")
        if s.notna().sum() == 0:
            continue
        if s.isna().mean() > 0.95:
            continue

        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            continue
        if nunique > max_cardinality:
            continue
        if nunique > max(50, int(0.5 * len(train_df))):
            continue

        categories = s.dropna().unique()
        mapping = {c: c for c in categories}

        train_df[col] = s.map(mapping).fillna("UNKNOWN")
        test_df[col] = test_df[col].astype("string").map(mapping).fillna("UNKNOWN")
        keep_categorical.append(col)

    feature_cols = keep_numeric + keep_categorical
    if len(feature_cols) == 0:
        return train_df.iloc[0:0], test_df, y, []

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for col in keep_numeric:
        X_train[col] = clean_numeric_series(X_train[col])
        X_test[col] = clean_numeric_series(X_test[col])

    for col in keep_categorical:
        categories = list(X_train[col].astype("string").unique())
        if "UNKNOWN" not in categories:
            categories.append("UNKNOWN")
        X_train[col] = X_train[col].astype("category").cat.set_categories(categories)
        X_test[col] = X_test[col].astype("category").cat.set_categories(categories)

    return X_train, X_test, y, keep_categorical


def make_base_models(cat_cols):
    models = {}

    models["catboost"] = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat_cols,
    )

    models["lightgbm"] = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        categorical_feature=cat_cols,
    )

    models["xgboost"] = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
    )

    return models


def fit_with_early_stopping(model, name, X_tr, y_tr, X_val, y_val):
    if name == "catboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter < 0:
            best_iter = 0
        return model, best_iter + 1

    if name == "lightgbm":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter <= 0:
            best_iter = model.n_estimators
        return model, int(best_iter)

    if name == "xgboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50,
        )
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None or best_iter < 0:
            best_iter = model.n_estimators - 1
        return model, int(best_iter) + 1

    raise ValueError(f"Unknown model: {name}")


def clone_model(name, cat_cols, n_estimators):
    if name == "catboost":
        return CatBoostRegressor(
            iterations=max(100, int(n_estimators)),
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_cols,
        )

    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=max(100, int(n_estimators)),
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            categorical_feature=cat_cols,
        )

    if name == "xgboost":
        return XGBRegressor(
            n_estimators=max(100, int(n_estimators)),
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
        )

    raise ValueError(f"Unknown model: {name}")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, X_test, y, cat_cols = prepare_data(train_df, test_df)

    if len(X_train) == 0 or len(X_test) == 0 or len(y) == 0:
        predictions = np.zeros(len(test_df), dtype=float)
    elif y.nunique(dropna=True) <= 1:
        predictions = np.full(len(test_df), float(y.mean()), dtype=float)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        candidates = []
        base_models = make_base_models(cat_cols)

        for name, model in base_models.items():
            try:
                fitted, best_iter = fit_with_early_stopping(
                    model, name, X_tr, y_tr, X_val, y_val
                )
                val_pred = fitted.predict(X_val)
                val_pred = np.asarray(val_pred, dtype=float)
                val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)
                val_pred = np.clip(val_pred, 0.0, None)

                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
                if np.isfinite(val_rmse):
                    candidates.append((name, best_iter, val_rmse))
            except Exception:
                continue

        if not candidates:
            try:
                model = LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    categorical_feature=cat_cols,
                )
                model.fit(X_train, y)
                predictions = model.predict(X_test)
            except Exception:
                predictions = np.full(len(test_df), float(y.mean()), dtype=float)
        else:
            candidates.sort(key=lambda x: x[2])
            top = candidates[: min(3, len(candidates))]
            weights = np.array([1.0 / max(1e-12, rmse) for _, _, rmse in top])
            weights = weights / weights.sum()

            test_preds = []
            for (name, best_iter, _), weight in zip(top, weights):
                try:
                    full_model = clone_model(name, cat_cols, best_iter)
                    full_model.fit(X_train, y)
                    pred = full_model.predict(X_test)
                    pred = np.asarray(pred, dtype=float)
                    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    pred = np.clip(pred, 0.0, None)
                    test_preds.append(pred * weight)
                except Exception:
                    continue

            if test_preds:
                predictions = np.sum(test_preds, axis=0)
            else:
                predictions = np.full(len(test_df), float(y.mean()), dtype=float)

    predictions = np.asarray(predictions, dtype=float)
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "Res_capacityM3": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()