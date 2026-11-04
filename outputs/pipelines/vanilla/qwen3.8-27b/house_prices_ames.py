import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/house_prices_ames/data/test.parquet"
TARGET = "Sale_Price"


def add_features(df):
    df = df.copy()

    def num(col, default=0.0):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return default

    df["Total_SF"] = num("Total_Bsmt_SF") + num("Gr_Liv_Area")
    df["Total_Bath"] = (
        num("Full_Bath")
        + 0.5 * num("Half_Bath")
        + num("Bsmt_Full_Bath")
        + 0.5 * num("Bsmt_Half_Bath")
    )
    df["Age"] = num("Year_Sold") - num("Year_Built")
    df["Remod_Age"] = num("Year_Sold") - num("Year_Remod_Add")
    df["Total_Porch"] = (
        num("Open_Porch_SF")
        + num("Enclosed_Porch")
        + num("Three_season_porch")
        + num("Screen_Porch")
    )
    return df


def prepare_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = train_df.dropna(subset=[TARGET])
    train_df = train_df[train_df[TARGET] > 0]

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    feature_cols = [
        col for col in train_df.columns
        if col != TARGET and col in test_df.columns
    ]
    feature_cols = [col for col in feature_cols if train_df[col].notna().any()]
    feature_cols = [
        col for col in feature_cols
        if train_df[col].nunique(dropna=True) > 1
    ]

    X = train_df[feature_cols].copy()
    y = train_df[TARGET].copy()
    X_test = test_df[feature_cols].copy()

    num_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
    cat_cols = [col for col in feature_cols if col not in num_cols]

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        X[col] = X[col].astype("object").fillna("missing").astype(str)
        X_test[col] = X_test[col].astype("object").fillna("missing").astype(str)

    return X, y, X_test, num_cols, cat_cols


def make_tree_features(X, num_cols, cat_cols, encoder=None):
    X = X.copy()
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if not cat_cols:
        return X[num_cols].values.astype(np.float32)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        cat_enc = encoder.fit_transform(X[cat_cols])
    else:
        cat_enc = encoder.transform(X[cat_cols])

    cat_enc = np.asarray(cat_enc.todense() if hasattr(cat_enc, "todense") else cat_enc, dtype=np.float32)
    num_enc = X[num_cols].values.astype(np.float32)
    return np.hstack([num_enc, cat_enc])


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_lgbm():
    return LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_xgb():
    return XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def make_catboost():
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X, y, X_test, num_cols, cat_cols = prepare_data(train_df, test_df)
    y_log = np.log1p(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    encoder = None
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(X_train[cat_cols])

    X_train_tree = make_tree_features(X_train, num_cols, cat_cols, encoder)
    X_val_tree = make_tree_features(X_val, num_cols, cat_cols, encoder)

    val_predictions = {}

    try:
        model = make_lgbm()
        model.fit(X_train_tree, y_train_log)
        val_predictions["lgbm"] = np.expm1(model.predict(X_val_tree))
    except Exception:
        pass

    try:
        model = make_xgb()
        model.fit(X_train_tree, y_train_log)
        val_predictions["xgb"] = np.expm1(model.predict(X_val_tree))
    except Exception:
        pass

    try:
        model = make_catboost()
        model.fit(
            X_train,
            y_train_log,
            cat_features=cat_cols,
            eval_set=(X_val, y_val_log),
            early_stopping_rounds=100,
            verbose=0,
        )
        val_predictions["catboost"] = np.expm1(model.predict(X_val))
    except Exception:
        pass

    if not val_predictions:
        final_pred = np.full(len(X_test), float(y.mean()), dtype=float)
    else:
        rmses = {name: rmse(y_val, pred) for name, pred in val_predictions.items()}
        finite = {name: value for name, value in rmses.items() if np.isfinite(value) and value > 0}

        if finite:
            inv = {name: 1.0 / value for name, value in finite.items()}
            total = sum(inv.values())
            weights = {name: value / total for name, value in inv.items()}
        else:
            weights = {name: 1.0 / len(val_predictions) for name in val_predictions}

        encoder_full = None
        if cat_cols:
            encoder_full = OneHotEncoder(handle_unknown="ignore")
            encoder_full.fit(X[cat_cols])

        X_full_tree = make_tree_features(X, num_cols, cat_cols, encoder_full)
        X_test_tree = make_tree_features(X_test, num_cols, cat_cols, encoder_full)

        test_predictions = {}

        for name in val_predictions:
            try:
                if name == "lgbm":
                    model = make_lgbm()
                    model.fit(X_full_tree, y_log)
                    test_predictions[name] = np.expm1(model.predict(X_test_tree))
                elif name == "xgb":
                    model = make_xgb()
                    model.fit(X_full_tree, y_log)
                    test_predictions[name] = np.expm1(model.predict(X_test_tree))
                elif name == "catboost":
                    model = make_catboost()
                    model.fit(X, y_log, cat_features=cat_cols, verbose=0)
                    test_predictions[name] = np.expm1(model.predict(X_test))
            except Exception:
                pass

        if not test_predictions:
            final_pred = np.full(len(X_test), float(y.mean()), dtype=float)
        else:
            available = {name: weights.get(name, 0.0) for name in test_predictions}
            total_weight = sum(available.values())
            if total_weight <= 0:
                available = {name: 1.0 / len(test_predictions) for name in test_predictions}
                total_weight = 1.0

            final_pred = np.zeros(len(X_test), dtype=float)
            for name, pred in test_predictions.items():
                final_pred += (available[name] / total_weight) * pred

    final_pred = np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)
    final_pred = np.clip(final_pred, 0.0, None)

    pd.DataFrame({
        "id": np.arange(len(final_pred), dtype=int),
        "Sale_Price": final_pred,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()