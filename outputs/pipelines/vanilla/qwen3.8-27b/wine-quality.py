import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/test.parquet"
TARGET = "quality"
RANDOM_STATE = 42


def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_categorical_dtype(series) or series.dtype.kind in "OU"


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
):
    X_train = X_train.copy()
    X_test = X_test.copy()

    if numeric_cols:
        for col in numeric_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        X_train[numeric_cols] = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)

        medians = X_train[numeric_cols].median().fillna(0.0)
        X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    if categorical_cols:
        for col in categorical_cols:
            X_train[col] = X_train[col].fillna("missing").astype(str)
            X_test[col] = X_test[col].fillna("missing").astype(str)

        train_dummies = pd.get_dummies(
            X_train[categorical_cols],
            prefix=categorical_cols,
            drop_first=False,
        )
        test_dummies = pd.get_dummies(
            X_test[categorical_cols],
            prefix=categorical_cols,
            drop_first=False,
        )
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        X_train_out = pd.concat(
            [
                X_train[numeric_cols].reset_index(drop=True),
                train_dummies.reset_index(drop=True),
            ],
            axis=1,
        )
        X_test_out = pd.concat(
            [
                X_test[numeric_cols].reset_index(drop=True),
                test_dummies.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        X_train_out = X_train[numeric_cols].reset_index(drop=True)
        X_test_out = X_test[numeric_cols].reset_index(drop=True)

    X_train_out = X_train_out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test_out = X_test_out.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X_train_out, X_test_out


def safe_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.inf
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    y_train = train_df[TARGET].astype(float)

    ignore_cols = {
        col for col in train_df.columns if col.lower() in {"id", "index", "row_id", "rowid"}
    }
    feature_cols = [col for col in train_df.columns if col != TARGET and col not in ignore_cols]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    categorical_cols = [col for col in feature_cols if is_categorical(train_df[col])]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    if len(X_train_raw) == 0 or len(feature_cols) == 0:
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                "quality": np.full(len(test_df), float(y_train.mean()) if len(y_train) else 0.0),
            }
        )
        submission.to_csv("predictions.csv", index=False)
        return

    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_fit, X_val = prepare_features(X_fit_raw, X_val_raw, numeric_cols, categorical_cols)

    models = {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }

    val_predictions = {}
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        val_predictions[name] = model.predict(X_val)

    val_rmse = {name: safe_rmse(y_val, pred) for name, pred in val_predictions.items()}
    weights = {
        name: (1.0 / score if np.isfinite(score) and score > 0 else 0.0)
        for name, score in val_rmse.items()
    }

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        weights = {name: 1.0 / len(weights) for name in weights}
    else:
        weights = {name: weight / weight_sum for name, weight in weights.items()}

    X_full, X_test = prepare_features(X_train_raw, X_test_raw, numeric_cols, categorical_cols)

    test_predictions = np.zeros(len(X_test), dtype=float)
    for name, model in models.items():
        model.fit(X_full, y_train)
        test_predictions += weights[name] * model.predict(X_test)

    test_predictions = np.clip(test_predictions, 0.0, 10.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "quality": test_predictions,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()