import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "Appliances"
DATE = "date"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

exclude_cols = {DATE, TARGET, "id"}
base_feature_cols = [c for c in train.columns if c not in exclude_cols]

date_feature_cols = [
    "hour",
    "minute",
    "dayofweek",
    "month",
    "day",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

feature_cols = base_feature_cols + date_feature_cols


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if DATE in df.columns:
        df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    else:
        df[DATE] = pd.NaT

    df["hour"] = df[DATE].dt.hour
    df["minute"] = df[DATE].dt.minute
    df["dayofweek"] = df[DATE].dt.dayofweek
    df["month"] = df[DATE].dt.month
    df["day"] = df[DATE].dt.day
    df["dayofyear"] = df[DATE].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def prepare_dataframe(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = add_date_features(df)

    if is_train:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        df = df.dropna(subset=[TARGET])

        if df[DATE].notna().any():
            df = df.dropna(subset=[DATE])

        df = df.sort_values(DATE, na_position="last").reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    return df


train = prepare_dataframe(train, is_train=True)
test = prepare_dataframe(test, is_train=False)

X_train = train[feature_cols].astype(float)
y_train = train[TARGET].astype(float)
X_test = test.reindex(columns=feature_cols).astype(float)

# Remove features that contain no information in the training set.
feature_cols = [c for c in feature_cols if not X_train[c].isna().all()]
X_train = X_train[feature_cols]
X_test = X_test.reindex(columns=feature_cols)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}

num_boost_round = 2000
early_stopping_rounds = 50

if len(X_train) >= 10:
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        dtrain = lgb.Dataset(
            X_train.iloc[train_idx],
            label=y_train.iloc[train_idx],
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_train.iloc[val_idx],
            label=y_train.iloc[val_idx],
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_it = getattr(model, "best_iteration", 0)
        if not best_it or best_it <= 0:
            best_it = num_boost_round

        preds = model.predict(X_train.iloc[val_idx], num_iteration=best_it)
        rmse = float(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))

        fold_scores.append(rmse)
        best_iterations.append(best_it)
        print(f"Fold {fold} RMSE: {rmse:.4f}")

    print(f"Mean 5-fold time-series CV RMSE: {np.mean(fold_scores):.4f}")
else:
    baseline_preds = np.full_like(y_train, y_train.mean(), dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_train, baseline_preds)))
    fold_scores = [rmse]
    best_iterations = [num_boost_round]
    print(f"Mean 5-fold time-series CV RMSE: {rmse:.4f}")

final_rounds = int(np.clip(np.mean(best_iterations), 100, num_boost_round))

dtrain_full = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_preds = final_model.predict(X_test)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=int),
        "Appliances": test_preds.astype(float),
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
