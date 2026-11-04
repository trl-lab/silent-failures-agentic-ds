import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/bike-sharing-demand/data/test.parquet"
TARGET = "count"
LEAKY_COLS = ["casual", "registered", "id"]


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 0.0, None)
    y_pred = np.clip(y_pred, 0.0, None)
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def add_time_features(df):
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")

        df["year"] = dt.dt.year.astype(float)
        df["month"] = dt.dt.month.astype(float)
        df["day"] = dt.dt.day.astype(float)
        df["hour"] = dt.dt.hour.astype(float)
        df["dayofweek"] = dt.dt.dayofweek.astype(float)
        df["dayofyear"] = dt.dt.dayofyear.astype(float)

        try:
            df["weekofyear"] = dt.dt.isocalendar().week.astype(float)
        except Exception:
            df["weekofyear"] = np.nan

        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)

        hour = dt.dt.hour.astype(float)
        dow = dt.dt.dayofweek.astype(float)
        month = dt.dt.month.astype(float)

        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    drop_cols = ["datetime", TARGET, "casual", "registered", "id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols)

    class Preprocessor:
        def fit(self, X):
            self.medians = {}
            for col in numeric_cols:
                if col in X.columns:
                    values = X[col].dropna()
                    try:
                        values = values.astype(float)
                    except Exception:
                        values = pd.Series(dtype=float)
                    self.medians[col] = float(values.median()) if len(values) else 0.0
                else:
                    self.medians[col] = 0.0

            self.cat_maps = {}
            for col in categorical_cols:
                if col in X.columns:
                    values = X[col].dropna().astype(str)
                else:
                    values = pd.Series(dtype=str)

                if len(values) == 0:
                    self.cat_maps[col] = {}
                else:
                    categories = sorted(values.unique())
                    self.cat_maps[col] = {cat: i for i, cat in enumerate(categories)}

            return self

        def transform(self, X):
            out = []

            for col in numeric_cols:
                if col in X.columns:
                    try:
                        series = X[col].astype(float)
                    except Exception:
                        series = pd.Series(np.nan, index=X.index)
                else:
                    series = pd.Series(np.nan, index=X.index)

                out.append(series.fillna(self.medians.get(col, 0.0)).to_numpy(dtype=float))

            for col in categorical_cols:
                mapping = self.cat_maps.get(col, {})
                if col in X.columns:
                    series = X[col].astype(str)
                else:
                    series = pd.Series("missing", index=X.index)

                out.append(series.map(mapping).fillna(-1).astype(int).to_numpy())

            if not out:
                return np.zeros((len(X), 1), dtype=float)

            return np.column_stack(out)

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    return Preprocessor()


def make_models():
    return {
        "lgbm": lgb.LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def fit_with_validation(name, model, X_train, y_train, X_val, y_val):
    if name == "lgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    elif name == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "cat":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    else:
        model.fit(X_train, y_train)

    return model


def best_iteration(name, model, default):
    try:
        if name == "lgbm":
            value = getattr(model, "best_iteration_", None)
        elif name == "xgb":
            value = getattr(model, "best_iteration", None)
            if value is not None:
                value += 1
        elif name == "cat":
            value = model.get_best_iteration()
            if value is not None:
                value += 1
        else:
            value = None

        if value is None:
            return default

        return max(100, int(value))
    except Exception:
        return default


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    if len(train_df) == 0:
        pd.DataFrame({
            "id": np.arange(len(test_df), dtype=int),
            "count": np.zeros(len(test_df), dtype=int),
        }).to_csv("predictions.csv", index=False)
        return

    y_train = train_df[TARGET].astype(float)
    y_train = y_train.fillna(0.0).clip(lower=0.0)

    train_df = train_df.drop(columns=[TARGET], errors="ignore")

    for col in LEAKY_COLS:
        train_df = train_df.drop(columns=[col], errors="ignore")
        test_df = test_df.drop(columns=[col], errors="ignore")

    train_dt = pd.to_datetime(train_df["datetime"], errors="coerce") if "datetime" in train_df.columns else None

    train_feat = add_time_features(train_df)
    test_feat = add_time_features(test_df)

    feature_cols = [c for c in train_feat.columns if not train_feat[c].isna().all()]

    if not feature_cols:
        train_feat["const"] = 1.0
        test_feat["const"] = 1.0
        feature_cols = ["const"]

    test_feat = test_feat.reindex(columns=feature_cols)

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_feat[c]) or pd.api.types.is_bool_dtype(train_feat[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X_train_full = train_feat[feature_cols]
    X_test = test_feat[feature_cols]
    y_log_train = np.log1p(y_train)

    n = len(X_train_full)

    if n <= 5:
        train_idx = np.arange(n)
        val_idx = np.arange(n)
    elif train_dt is not None and train_dt.notna().sum() >= 10:
        valid_idx = np.where(train_dt.notna().to_numpy())[0]
        order = valid_idx[np.argsort(train_dt.to_numpy()[valid_idx], kind="mergesort")]
        n_val = max(1, int(len(order) * 0.2))
        val_idx = order[-n_val:]
        train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
    else:
        train_idx, val_idx = train_test_split(
            np.arange(n),
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

    if len(train_idx) == 0:
        train_idx = np.arange(n)
        val_idx = np.arange(n)

    if len(val_idx) == 0:
        val_idx = np.array([n - 1])
        train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)

    X_train_split = X_train_full.iloc[train_idx]
    X_val_split = X_train_full.iloc[val_idx]
    y_train_split = y_log_train.iloc[train_idx]
    y_val_split = y_log_train.iloc[val_idx]
    y_val_true = y_train.iloc[val_idx]

    val_preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    X_train_split_t = val_preprocessor.fit_transform(X_train_split)
    X_val_split_t = val_preprocessor.transform(X_val_split)

    base_models = make_models()
    fitted_models = {}
    val_scores = {}

    for name, model in base_models.items():
        try:
            model = fit_with_validation(name, model, X_train_split_t, y_train_split, X_val_split_t, y_val_split)
            pred_log = model.predict(X_val_split_t)
            pred = np.expm1(pred_log)
            score = rmsle(y_val_true, pred)

            if not np.isfinite(score):
                score = np.inf

            fitted_models[name] = model
            val_scores[name] = score
        except Exception:
            continue

    if not fitted_models:
        try:
            fallback = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            fallback.fit(X_train_split_t, y_train_split)
            fitted_models["fallback"] = fallback
            val_scores["fallback"] = rmsle(y_val_true, np.expm1(fallback.predict(X_val_split_t)))
        except Exception:
            fitted_models["constant"] = None
            val_scores["constant"] = 0.0

    top_models = sorted(val_scores, key=lambda k: val_scores[k])[:2]

    final_preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    X_train_full_t = final_preprocessor.fit_transform(X_train_full)
    X_test_t = final_preprocessor.transform(X_test)

    final_models = []

    for name in top_models:
        if name == "constant":
            continue

        try:
            if name in base_models:
                model = base_models[name]
            elif name == "fallback":
                model = lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                continue

            iters = best_iteration(name, fitted_models[name], default=1000)

            if name == "cat":
                model.set_params(iterations=iters)
            else:
                model.set_params(n_estimators=iters)

            model.fit(X_train_full_t, y_log_train)
            final_models.append(model)
        except Exception:
            continue

    if not final_models:
        test_pred = np.full(len(X_test), float(np.expm1(y_log_train.mean())), dtype=float)
    else:
        preds = []
        for model in final_models:
            try:
                pred_log = model.predict(X_test_t)
                preds.append(np.expm1(pred_log))
            except Exception:
                continue

        if preds:
            test_pred = np.mean(preds, axis=0)
        else:
            test_pred = np.full(len(X_test), float(np.expm1(y_log_train.mean())), dtype=float)

    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, None)
    test_pred = np.round(test_pred).astype(int)

    pd.DataFrame({
        "id": np.arange(len(test_pred), dtype=int),
        "count": test_pred,
    }).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()