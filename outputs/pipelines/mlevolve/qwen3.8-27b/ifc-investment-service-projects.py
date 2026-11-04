import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

TARGET = "Total IFC investment as approved by Board(Million - USD)"
RANDOM_STATE = 42
VAL_FRACTION = 0.20
EARLY_STOPPING_ROUNDS = 100

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


def _numeric_like(series: pd.Series) -> bool:
    if series.dtype.kind in "iufb":
        return True
    if series.dtype.kind in "OUS":
        if series.notna().sum() == 0:
            return False
        converted = pd.to_numeric(series, errors="coerce")
        return converted.notna().sum() >= 0.9 * series.notna().sum()
    return False


def _as_numeric(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.replace([np.inf, -np.inf], np.nan)


def _as_native_categorical(series: pd.Series, categories: list) -> pd.Series:
    return pd.Categorical(series, categories=categories)


def prepare_features(
    train_raw: pd.DataFrame, val_raw: pd.DataFrame, test_raw: pd.DataFrame
):
    train = train_raw.copy()
    val = val_raw.copy()
    test = test_raw.copy()

    for col in train.columns:
        if col not in val.columns:
            val[col] = np.nan
        if col not in test.columns:
            test[col] = np.nan

    cat_cols = []

    for col in train.columns:
        train_series = train[col]

        if _numeric_like(train_series):
            train[col] = _as_numeric(train_series)
            val[col] = _as_numeric(val[col])
            test[col] = _as_numeric(test[col])
        else:
            categories = list(dict.fromkeys(train_series.dropna().unique()))
            if len(categories) == 0:
                train[col] = np.nan
                val[col] = np.nan
                test[col] = np.nan
            else:
                train[col] = _as_native_categorical(train_series, categories)
                val[col] = _as_native_categorical(val[col], categories)
                test[col] = _as_native_categorical(test[col], categories)
                cat_cols.append(col)

    keep_cols = []
    for col in train.columns:
        non_null = train[col].notna().sum()
        if non_null == 0:
            continue
        if train[col].nunique(dropna=True) <= 1:
            continue
        keep_cols.append(col)

    if len(keep_cols) == 0:
        train["constant_feature"] = 1.0
        val["constant_feature"] = 1.0
        test["constant_feature"] = 1.0
        keep_cols = ["constant_feature"]

    cat_cols = [c for c in cat_cols if c in keep_cols]

    return train[keep_cols], val[keep_cols], test[keep_cols], cat_cols


def make_model(n_estimators: int = 6000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_split_gain=0.01,
        max_bin=255,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def _clean_target(train_frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in train_frame.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    train_frame = train_frame.copy()
    train_frame[TARGET] = pd.to_numeric(train_frame[TARGET], errors="coerce")
    train_frame = train_frame[np.isfinite(train_frame[TARGET])]

    if len(train_frame) == 0:
        raise ValueError("No training rows available after target cleaning.")

    return train_frame


def _drop_identifier(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "id" in frame.columns:
        frame = frame.drop(columns=["id"])
    if TARGET in frame.columns:
        frame = frame.drop(columns=[TARGET])
    return frame


def _postprocess_predictions(
    predictions: np.ndarray, lower_bound: float, fallback: float
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.where(np.isfinite(predictions), predictions, fallback)

    if lower_bound >= 0.0:
        predictions = np.clip(predictions, 0.0, None)

    return predictions


def _predict_from_booster(
    model_file: str, frame: pd.DataFrame, num_iteration: int | None = None
) -> np.ndarray:
    booster = lgb.Booster(model_file=model_file)
    if num_iteration is None:
        return booster.predict(frame)
    return booster.predict(frame, num_iteration=int(num_iteration))


def main():
    train_full = pd.read_parquet("./input/train.parquet")
    test_full = pd.read_parquet("./input/test.parquet")

    train_full = _clean_target(train_full)

    y_full = train_full[TARGET].astype(float)
    x_full = _drop_identifier(train_full)

    if len(x_full) == 0:
        raise ValueError("No feature rows available after cleaning.")

    if len(x_full) < 5:
        train_idx = np.arange(len(x_full))
        val_idx = np.array([], dtype=int)
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(x_full)),
            test_size=VAL_FRACTION,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    x_train_raw = x_full.iloc[train_idx]
    y_train = y_full.iloc[train_idx]
    x_val_raw = x_full.iloc[val_idx]
    y_val = y_full.iloc[val_idx]

    test_work = _drop_identifier(test_full)

    x_train, x_val, x_test, categorical_features = prepare_features(
        x_train_raw, x_val_raw, test_work
    )

    _, x_full_prepared, x_test_prepared, full_categorical_features = prepare_features(
        x_train_raw, x_full, test_work
    )

    validation_model_file = "./working/best_validation_model.txt"
    full_model_file = "./working/full_model.txt"

    if len(x_val) > 0:
        val_model = make_model()
        val_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(
                    EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False
                ),
                lgb.log_evaluation(-1),
            ],
            categorical_feature=categorical_features,
        )

        best_iter = int(getattr(val_model, "best_iteration_", 0) or 0)
        best_iter = max(best_iter, 1)
        val_model.booster_.save_model(validation_model_file)
    else:
        best_iter = 1
        val_model = make_model(n_estimators=best_iter)
        val_model.fit(
            x_train,
            y_train,
            categorical_feature=categorical_features,
            callbacks=[lgb.log_evaluation(-1)],
        )
        val_model.booster_.save_model(validation_model_file)

    full_model = make_model(n_estimators=best_iter)
    full_model.fit(
        x_full_prepared,
        y_full,
        categorical_feature=full_categorical_features,
        callbacks=[lgb.log_evaluation(-1)],
    )
    full_model.booster_.save_model(full_model_file)

    lower_bound = float(y_full.min())
    fallback = float(y_full.mean())

    if len(x_val) > 0:
        val_pred = _postprocess_predictions(
            _predict_from_booster(
                validation_model_file, x_val, num_iteration=best_iter
            ),
            lower_bound,
            fallback,
        )
        score = float(root_mean_squared_error(y_val, val_pred))
    else:
        train_pred = _postprocess_predictions(
            _predict_from_booster(
                validation_model_file, x_train, num_iteration=best_iter
            ),
            lower_bound,
            fallback,
        )
        score = float(root_mean_squared_error(y_train, train_pred))

    if len(x_test_prepared) == 0:
        test_pred = np.array([], dtype=float)
    else:
        test_pred = _postprocess_predictions(
            _predict_from_booster(
                full_model_file, x_test_prepared, num_iteration=best_iter
            ),
            lower_bound,
            fallback,
        )

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            TARGET: np.asarray(test_pred, dtype=float),
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
