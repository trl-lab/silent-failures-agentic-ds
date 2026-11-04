import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TARGET = "Gross Issued Amount (US$)"
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/miga-issued-projects/data/test.parquet"
DROP_COLS = ["Project ID", "Project Name", "Project Links"]
DATE_COLS = ["Project Start Date", "Project Expiration Date"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Project Start Date" in df.columns:
        df["start_year"] = df["Project Start Date"].dt.year
        df["start_month"] = df["Project Start Date"].dt.month
        df["start_quarter"] = df["Project Start Date"].dt.quarter

    if "Project Expiration Date" in df.columns:
        df["expiration_year"] = df["Project Expiration Date"].dt.year
        df["expiration_month"] = df["Project Expiration Date"].dt.month

    if "Project Start Date" in df.columns and "Project Expiration Date" in df.columns:
        df["duration_years"] = (
            df["Project Expiration Date"] - df["Project Start Date"]
        ).dt.days / 365.25

    if "Issued FY" in df.columns:
        if "start_year" in df.columns:
            df["start_year_minus_issued_fy"] = df["start_year"] - df["Issued FY"]
        if "expiration_year" in df.columns:
            df["expiration_year_minus_issued_fy"] = (
                df["expiration_year"] - df["Issued FY"]
            )

    df = df.drop(columns=DATE_COLS, errors="ignore")
    return df


def categorical_frame(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    out = df[cat_cols].astype(object).copy()
    for col in cat_cols:
        out[col] = out[col].where(pd.notnull(out[col]), "missing")
    return out


def fit_encoder(df: pd.DataFrame, cat_cols: list) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(categorical_frame(df, cat_cols))
    return encoder


def transform_features(
    df: pd.DataFrame,
    feature_cols: list,
    cat_cols: list,
    encoder,
) -> pd.DataFrame:
    df = df[feature_cols].copy()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    if num_cols:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    else:
        num_df = pd.DataFrame(index=df.index)

    if cat_cols and encoder is not None:
        cat_array = encoder.transform(categorical_frame(df, cat_cols))
        if hasattr(cat_array, "toarray"):
            cat_array = cat_array.toarray()
        cat_df = pd.DataFrame(
            cat_array,
            columns=encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )
        out = pd.concat([num_df, cat_df], axis=1)
    else:
        out = num_df

    out = out.loc[:, ~out.columns.duplicated()]
    if out.shape[1] == 0:
        out = pd.DataFrame({"const": np.ones(len(out))}, index=out.index)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.astype(float)


def make_model(name: str):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    if name == "catboost":
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_with_early_stopping(model, name: str, X_tr, y_tr, X_val, y_val):
    if name == "lgbm":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    elif name == "xgboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
    elif name == "catboost":
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def best_iteration(model, name: str, default: int = 1000) -> int:
    if name == "lgbm":
        return max(1, int(getattr(model, "best_iteration_", default)))
    if name == "xgboost":
        bi = getattr(model, "best_iteration", None)
        if bi is None or bi < 0:
            return default
        return max(1, int(bi) + 1)
    if name == "catboost":
        bi = getattr(model, "best_iteration_", None)
        if bi is None or bi < 0:
            return default
        return max(1, int(bi) + 1)
    return default


def inverse_transform_target(pred: np.ndarray, use_log: bool, min_y: float) -> np.ndarray:
    if use_log:
        pred = np.expm1(pred)
    if min_y >= 0:
        pred = np.maximum(pred, 0.0)
    return pred


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_clean = clean_dataframe(train_df)
    test_clean = clean_dataframe(test_df)

    y = train_clean[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    X_train_full = train_clean.drop(columns=[TARGET])
    X_test_full = test_clean.drop(columns=[TARGET], errors="ignore")

    valid = y.notna()
    X_train_full = X_train_full.loc[valid]
    y = y.loc[valid]

    if len(y) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                TARGET: np.zeros(len(test_df), dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    common_cols = [c for c in X_train_full.columns if c in X_test_full.columns]
    X_train_full = X_train_full[common_cols]
    X_test_full = X_test_full[common_cols]

    X_train_full = X_train_full.loc[:, X_train_full.notna().any()]
    X_test_full = X_test_full.loc[:, X_train_full.columns]

    if X_train_full.shape[1] == 0:
        X_train_full["const"] = 1.0
        X_test_full["const"] = 1.0

    cat_cols = [
        c
        for c in X_train_full.columns
        if X_train_full[c].dtype == object or str(X_train_full[c].dtype) == "category"
    ]
    feature_cols = list(X_train_full.columns)
    cat_cols = [c for c in cat_cols if c in feature_cols]

    if len(y) >= 20:
        X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
            X_train_full,
            y,
            test_size=0.2,
            random_state=42,
        )
    else:
        X_tr_raw = X_train_full
        X_val_raw = X_train_full
        y_tr = y
        y_val = y

    encoder_tr = fit_encoder(X_tr_raw, cat_cols) if cat_cols else None
    X_tr = transform_features(X_tr_raw, feature_cols, cat_cols, encoder_tr)
    X_val = transform_features(X_val_raw, feature_cols, cat_cols, encoder_tr)
    X_test = transform_features(X_test_full, feature_cols, cat_cols, encoder_tr)

    min_y = float(y.min())
    use_log_options = [False]
    if min_y >= 0:
        use_log_options.append(True)

    best_rmse = np.inf
    best_name = "lgbm"
    best_use_log = False
    best_iter = 1000

    for name in ["lgbm", "xgboost", "catboost"]:
        for use_log in use_log_options:
            y_tr_t = np.log1p(y_tr) if use_log else y_tr.to_numpy(dtype=float)
            y_val_t = np.log1p(y_val) if use_log else y_val.to_numpy(dtype=float)

            model = make_model(name)
            try:
                model = fit_with_early_stopping(
                    model, name, X_tr, y_tr_t, X_val, y_val_t
                )
                pred_val_t = model.predict(X_val)
                pred_val = inverse_transform_target(
                    np.asarray(pred_val_t, dtype=float), use_log, min_y
                )
                rmse = root_mean_squared_error(
                    y_val.to_numpy(dtype=float), pred_val
                )
            except Exception:
                continue

            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_use_log = use_log
                best_iter = best_iteration(model, name, default=1000)

    encoder_full = fit_encoder(X_train_full, cat_cols) if cat_cols else None
    X_train_final = transform_features(
        X_train_full, feature_cols, cat_cols, encoder_full
    )
    X_test_final = transform_features(
        X_test_full, feature_cols, cat_cols, encoder_full
    )

    final_model = make_model(best_name)
    if best_name == "lgbm":
        final_model.set_params(n_estimators=best_iter)
    elif best_name == "xgboost":
        final_model.set_params(n_estimators=best_iter)
    elif best_name == "catboost":
        final_model.set_params(iterations=best_iter)

    y_full_t = np.log1p(y) if best_use_log else y.to_numpy(dtype=float)

    try:
        final_model.fit(X_train_final, y_full_t)
        pred_test_t = final_model.predict(X_test_final)
    except Exception:
        try:
            fallback = LGBMRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            fallback.fit(X_train_final, y.to_numpy(dtype=float))
            pred_test_t = fallback.predict(X_test_final)
            best_use_log = False
        except Exception:
            pred_test_t = np.full(len(X_test_final), float(y.median()))

    pred_test = inverse_transform_target(
        np.asarray(pred_test_t, dtype=float), best_use_log, min_y
    )
    pred_test = np.nan_to_num(
        pred_test,
        nan=float(y.median()),
        posinf=float(y.max()),
        neginf=float(y.min()),
    )
    if min_y >= 0:
        pred_test = np.maximum(pred_test, 0.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(pred_test), dtype=int),
            TARGET: pred_test.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()