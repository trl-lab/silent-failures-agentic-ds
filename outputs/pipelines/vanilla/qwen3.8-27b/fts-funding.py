import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/test.parquet"
TARGET = "amountUSD"
RANDOM_STATE = 42

DROP_COLS = [
    "id",
    "refCode",
    "createdAt",
    "updatedAt",
]

DATE_COLS = [
    "date",
    "firstReportedDate",
    "decisionDate",
]

TEXT_COLS = [
    "description",
]

LIST_COLS = [
    "destGlobalClusters",
    "keywords",
    "srcLocations",
    "destLocations",
]

LGB_N_ESTIMATORS = 2000
XGB_N_ESTIMATORS = 1500
CB_ITERATIONS = 2000


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(name: str) -> str:
        name = str(name)
        name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "feature"
        return name

    new_cols = []
    seen = {}
    for col in df.columns:
        base = clean(col)
        if base in seen:
            seen[base] += 1
            new = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            new = base
        new_cols.append(new)

    return df.rename(columns=dict(zip(df.columns, new_cols)))


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col not in df.columns:
            continue

        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year.astype("float32")
        df[f"{col}_month"] = dt.dt.month.astype("float32")
        df[f"{col}_day"] = dt.dt.day.astype("float32")
        df[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
        df[f"{col}_dayofyear"] = dt.dt.dayofyear.astype("float32")
        df[f"{col}_quarter"] = dt.dt.quarter.astype("float32")
        df = df.drop(columns=[col])

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str)
        df[f"{col}_len"] = s.str.len().astype("float32")
        df[f"{col}_words"] = s.str.split().str.len().astype("float32")
        df[f"{col}_digits"] = s.str.count(r"\d").astype("float32")
        df[f"{col}_numbers"] = s.str.count(r"\d+").astype("float32")
        df[f"{col}_upper"] = s.str.count(r"[A-Z]").astype("float32")
        df[f"{col}_lower"] = s.str.count(r"[a-z]").astype("float32")
        df[f"{col}_punct"] = s.str.count(r"[^\w\s]").astype("float32")
        df[f"{col}_dates"] = s.str.count(r"\d{4}-\d{2}-\d{2}").astype("float32")

        keywords = [
            "emergency",
            "standard",
            "flash",
            "addendum",
            "refugee",
            "earthquake",
            "conflict",
            "shelter",
            "wash",
            "health",
            "education",
            "food",
            "protection",
            "cash",
            "nfi",
            "rrp",
            "hnrf",
            "crrf",
            "project code",
            "allocation type",
            "project start date",
            "project end date",
        ]

        for kw in keywords:
            safe_kw = kw.replace(" ", "_")
            df[f"{col}_has_{safe_kw}"] = (
                s.str.contains(kw, case=False, regex=False).astype("float32")
            )

        df = df.drop(columns=[col])

    return df


def add_list_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in LIST_COLS:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str).str.strip()
        s = s.replace({"None": "", "none": "", "nan": "", "NaN": ""})

        df[f"{col}_n_items"] = s.str.split(",").str.len().astype("float32")
        df[f"{col}_len"] = s.str.len().astype("float32")

    return df


def encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = [col for col in train.columns if train[col].dtype == object]

    for col in cat_cols:
        if col not in test.columns:
            train = train.drop(columns=[col])
            continue

        train[col] = train[col].fillna("missing").astype(str).str.strip()
        test[col] = test[col].fillna("missing").astype(str).str.strip()

        train[col] = train[col].replace(
            {"": "missing", "None": "missing", "none": "missing", "nan": "missing", "NaN": "missing"}
        )
        test[col] = test[col].replace(
            {"": "missing", "None": "missing", "none": "missing", "nan": "missing", "NaN": "missing"}
        )

        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq).fillna(0.0).astype("float32")
        test[f"{col}_freq"] = test[col].map(freq).fillna(0.0).astype("float32")

        if train[col].nunique() <= 50:
            train_dummies = pd.get_dummies(train[col], prefix=col, dtype="float32")
            test_dummies = pd.get_dummies(test[col], prefix=col, dtype="float32")

            all_dummy_cols = train_dummies.columns.union(test_dummies.columns)
            train_dummies = train_dummies.reindex(columns=all_dummy_cols, fill_value=0.0)
            test_dummies = test_dummies.reindex(columns=all_dummy_cols, fill_value=0.0)

            train = pd.concat([train, train_dummies], axis=1)
            test = pd.concat([test, test_dummies], axis=1)

        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

    return train, test


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns], errors="ignore")
    test = test.drop(columns=[c for c in DROP_COLS if c in test.columns], errors="ignore")

    train = add_date_features(train)
    test = add_date_features(test)

    train = add_text_features(train)
    test = add_text_features(test)

    train = add_list_features(train)
    test = add_list_features(test)

    train, test = encode_categoricals(train, test)

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    train = sanitize_columns(train)
    test = sanitize_columns(test)

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    feature_cols = [c for c in train.columns if train[c].dtype != object]
    feature_cols = [
        c for c in feature_cols
        if train[c].notna().any() or test[c].notna().any()
    ]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X_train, X_test


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df[TARGET] = train_df[TARGET].astype("float64")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET])

    y = train_df[TARGET]
    train_df = train_df.drop(columns=[TARGET])

    X_train, X_test = prepare_features(train_df, test_df)

    if X_train.shape[1] == 0:
        predictions = np.full(len(X_test), float(y.mean()), dtype="float64")
        pd.DataFrame({"id": np.arange(len(predictions), dtype=int), "amountUSD": predictions}).to_csv(
            "predictions.csv",
            index=False,
        )
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    lgb_model = LGBMRegressor(
        n_estimators=LGB_N_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    xgb_model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    cb_model = CatBoostRegressor(
        iterations=CB_ITERATIONS,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        loss_function="RMSE",
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    val_preds = {
        "lgb": lgb_model.predict(X_val),
        "xgb": xgb_model.predict(X_val),
        "cb": cb_model.predict(X_val),
    }

    val_rmses = {
        name: float(np.sqrt(mean_squared_error(y_val, pred)))
        for name, pred in val_preds.items()
    }

    weights = {name: (1.0 / max(rmse, 1e-12)) for name, rmse in val_rmses.items()}
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    lgb_iters = int(getattr(lgb_model, "best_iteration_", 0))
    if lgb_iters <= 0:
        lgb_iters = LGB_N_ESTIMATORS

    cb_iters = int(cb_model.get_best_iteration())
    if cb_iters <= 0:
        cb_iters = CB_ITERATIONS
    else:
        cb_iters += 1

    lgb_final = LGBMRegressor(
        n_estimators=lgb_iters,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_final.fit(X_train, y)

    xgb_final = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )
    xgb_final.fit(X_train, y, verbose=False)

    cb_final = CatBoostRegressor(
        iterations=cb_iters,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        loss_function="RMSE",
    )
    cb_final.fit(X_train, y, verbose=0)

    test_preds = np.column_stack(
        [
            lgb_final.predict(X_test),
            xgb_final.predict(X_test),
            cb_final.predict(X_test),
        ]
    )

    weight_vec = np.array(
        [weights["lgb"], weights["xgb"], weights["cb"]],
        dtype="float64",
    )
    final_preds = test_preds @ weight_vec
    final_preds = np.clip(final_preds, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_preds), dtype=int),
            "amountUSD": final_preds.astype("float64"),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()