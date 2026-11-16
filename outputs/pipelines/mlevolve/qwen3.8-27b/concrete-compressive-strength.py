import os
import warnings
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

TARGET = "strength"
SEED = 42
VAL_FRACTION = 0.15


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_float_array(x, dtype=np.float32):
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return arr.astype(dtype, copy=False)


def official_rmse(y_true, y_pred) -> float:
    y_true = to_float_array(y_true, dtype=np.float64)
    y_pred = to_float_array(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def drop_target_and_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in [TARGET, "id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.select_dtypes(include=[np.number])
    return df


def safe_divide(a, b, eps: float = 1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.where(np.abs(b) < eps, np.where(b >= 0, eps, -eps), b)
    return (a / denom).astype(np.float32)


def add_domain_features_if_raw(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    required = {"cement", "blast_furnace_slag", "fly_ash", "water", "age"}
    if not required.issubset(set(X.columns)):
        return X

    cement = X["cement"].to_numpy(dtype=np.float32)
    slag = X["blast_furnace_slag"].to_numpy(dtype=np.float32)
    fly = X["fly_ash"].to_numpy(dtype=np.float32)
    water = X["water"].to_numpy(dtype=np.float32)
    sp = (
        X["superplasticizer"].to_numpy(dtype=np.float32)
        if "superplasticizer" in X.columns
        else np.zeros(len(X), dtype=np.float32)
    )
    coarse = (
        X["coarse_aggregate"].to_numpy(dtype=np.float32)
        if "coarse_aggregate" in X.columns
        else np.zeros(len(X), dtype=np.float32)
    )
    fine = (
        X["fine_aggregate"].to_numpy(dtype=np.float32)
        if "fine_aggregate" in X.columns
        else np.zeros(len(X), dtype=np.float32)
    )
    age = X["age"].to_numpy(dtype=np.float32)

    binder = cement + slag + fly
    aggregate = coarse + fine
    total = binder + water + sp + aggregate

    X["binder_total"] = binder
    X["aggregate_total"] = aggregate
    X["water_cement"] = safe_divide(water, cement)
    X["water_binder"] = safe_divide(water, binder)
    X["binder_water"] = safe_divide(binder, water)
    X["binder_aggregate"] = safe_divide(binder, aggregate)
    X["sp_binder"] = safe_divide(sp, binder)
    X["cement_binder"] = safe_divide(cement, binder)
    X["slag_binder"] = safe_divide(slag, binder)
    X["fly_binder"] = safe_divide(fly, binder)
    X["log_age"] = np.log1p(age).astype(np.float32)
    X["sqrt_age"] = np.sqrt(age).astype(np.float32)
    X["age_cement"] = safe_divide(age, cement)
    X["age_water"] = safe_divide(age, water)
    X["total_mixture"] = total
    X["water_total"] = safe_divide(water, total)

    return X


def prepare_xy_from_frame(df: pd.DataFrame, is_test: bool = False):
    y = None
    if not is_test and TARGET in df.columns:
        y = to_float_array(df[TARGET], dtype=np.float32)

    X = drop_target_and_id(df)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    return X, y


def prepare_data():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    X_all, y_all = prepare_xy_from_frame(train_df, is_test=False)
    X_test, _ = prepare_xy_from_frame(test_df, is_test=True)

    X_all = add_domain_features_if_raw(X_all)
    X_test = add_domain_features_if_raw(X_test)

    X_test = X_test.reindex(columns=X_all.columns, fill_value=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VAL_FRACTION,
        random_state=SEED,
    )

    return X_train, y_train, X_val, y_val, X_test


def make_model():
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=2000,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=SEED,
        )


def fit_model(model, X_train, y_train, X_val, y_val):
    X_train = to_float_array(X_train)
    X_val = to_float_array(X_val)
    y_train = to_float_array(y_train)
    y_val = to_float_array(y_val)

    model_name = type(model).__name__

    if model_name == "LGBMRegressor":
        import lightgbm as lgb

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=callbacks,
            )
        except Exception:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
    else:
        model.fit(X_train, y_train)

    dump(model, "./working/best_model.joblib")
    model = load("./working/best_model.joblib")

    return model


def predict_model(model, X):
    X = to_float_array(X)
    return to_float_array(model.predict(X), dtype=np.float32)


def main():
    set_seed()

    X_train, y_train, X_val, y_val, X_test = prepare_data()

    model = make_model()
    model = fit_model(model, X_train, y_train, X_val, y_val)

    val_pred = predict_model(model, X_val)
    test_pred = predict_model(model, X_test)

    val_score = official_rmse(y_val, val_pred)

    test_pred = to_float_array(test_pred, dtype=np.float32).reshape(-1)
    if len(test_pred) != len(X_test):
        test_pred = to_float_array(
            model.predict(to_float_array(X_test)),
            dtype=np.float32,
        ).reshape(-1)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "strength": test_pred.astype(np.float64),
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {val_score}")


if __name__ == "__main__":
    main()
