import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/museums/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/museums/data/test.parquet"
TARGET = "REVENUE15"
RANDOM_STATE = 42

KEEP_HIGH_CARDINALITY = {
    "LONGITUDE",
    "LATITUDE",
    "ADZIP5",
    "PHZIP5",
    "GZIP5",
}

FORCE_CATEGORICAL = {
    "DISCIPLINE",
    "CO_LOC_RES",
    "NTEEC",
    "DUNS",
    "NAICS",
    "ADSTATE",
    "PHSTATE",
    "IRS_STATE15",
    "INSTST",
    "GSTATE",
    "SOURCE",
    "LOCALE4",
    "AAMREG",
    "BEAREG",
    "FIPSST",
    "FIPSCO",
    "TAXPER15",
    "RULEDATE15",
    "BMF15_F",
    "IMLSAD_F",
    "PFND_F",
    "FCT3P_F",
    "USER_F",
    "UNI_F",
    "OTHER_F",
}


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    y = train[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    train = train.loc[y.notna()].copy()
    y = y.loc[train.index]

    X = train.drop(columns=[TARGET]).copy()
    X_test = test.copy()

    n_train = len(X)
    high_card_threshold = max(500, int(0.25 * n_train))

    feature_cols = []
    for col in X.columns:
        if col == TARGET:
            continue
        if X[col].isna().all():
            continue
        nunique = X[col].nunique(dropna=True)
        if nunique <= 1:
            continue
        if nunique > high_card_threshold and col not in KEEP_HIGH_CARDINALITY:
            continue
        feature_cols.append(col)

    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = np.nan

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if col in FORCE_CATEGORICAL:
            categorical_cols.append(col)
            continue

        converted = pd.to_numeric(X[col], errors="coerce")
        non_null_fraction = converted.notna().mean()
        nunique = converted.nunique(dropna=True)

        if non_null_fraction > 0.5 and nunique > 1:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype(float)
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_cols:
        train_vals = X[col].where(X[col].notna(), "__MISSING__").astype(str)
        codes, uniques = pd.factorize(train_vals, sort=False)
        X[col] = codes.astype(np.int32)

        test_vals = X_test[col].where(X_test[col].notna(), "__MISSING__").astype(str)
        mapping = {str(u): i for i, u in enumerate(uniques)}
        X_test[col] = test_vals.map(mapping).fillna(-1).astype(np.int32)

    X = X[feature_cols]
    X_test = X_test.reindex(columns=feature_cols)

    return X, y, X_test, numeric_cols, categorical_cols


def best_iteration(model, default_iterations):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration", None)

    if best_iter is None or best_iter <= 0:
        return default_iterations

    return int(best_iter) + 1


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, numeric_cols, categorical_cols = prepare_features(train_df, test_df)

    if len(X) == 0:
        predictions = np.zeros(len(X_test), dtype=float)
    else:
        if len(X) < 50:
            lgb_final = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_final.fit(X, y, categorical_feature=categorical_cols)

            cat_final = CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=0,
                cat_features=categorical_cols,
            )
            cat_final.fit(X, y, verbose=0)

            predictions = np.mean(
                [
                    lgb_final.predict(X_test),
                    cat_final.predict(X_test),
                ],
                axis=0,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )

            val_predictions = []

            lgb_model = LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.04,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )

            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                categorical_feature=categorical_cols,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            val_predictions.append(lgb_model.predict(X_val))

            cat_model = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.04,
                depth=7,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=0,
                cat_features=categorical_cols,
            )

            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=0,
            )
            val_predictions.append(cat_model.predict(X_val))

            val_pred = np.mean(val_predictions, axis=0)
            val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))

            lgb_best_iter = best_iteration(lgb_model, 1500)
            cat_best_iter = best_iteration(cat_model, 1500)

            lgb_final = LGBMRegressor(
                n_estimators=lgb_best_iter,
                learning_rate=0.04,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_final.fit(X, y, categorical_feature=categorical_cols)

            cat_final = CatBoostRegressor(
                iterations=cat_best_iter,
                learning_rate=0.04,
                depth=7,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=0,
                cat_features=categorical_cols,
            )
            cat_final.fit(X, y, verbose=0)

            predictions = np.mean(
                [
                    lgb_final.predict(X_test),
                    cat_final.predict(X_test),
                ],
                axis=0,
            )

        predictions = np.asarray(predictions, dtype=float)
        predictions = np.where(np.isfinite(predictions), predictions, 0.0)
        predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "REVENUE15": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()