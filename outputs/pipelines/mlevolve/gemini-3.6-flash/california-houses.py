import gc
import os
import re
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")


def process_and_engineer_features(train_path, test_path, target_col):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Filter invalid target values in training set
    df_train = df_train[
        df_train[target_col].notna() & (df_train[target_col] > 0)
    ].reset_index(drop=True)

    y_train_raw = df_train[target_col].values.astype(np.float64)
    y_train_log = np.log1p(y_train_raw)

    if "id" in df_test.columns:
        test_ids = df_test["id"].values
    else:
        test_ids = np.arange(len(df_test), dtype=int)

    drop_cols = [target_col, "id"] if "id" in df_train.columns else [target_col]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    n_train = len(df_train)
    df_full = pd.concat(
        [df_train[feature_cols], df_test[feature_cols]], axis=0
    ).reset_index(drop=True)

    # Ensure unique and clean column names without special characters
    clean_cols = []
    seen = set()
    for col in df_full.columns:
        c = re.sub(r'[\[\]<>{}:,"]', "_", str(col))
        if c in seen:
            idx = 1
            while f"{c}_{idx}" in seen:
                idx += 1
            c = f"{c}_{idx}"
        seen.add(c)
        clean_cols.append(c)
    df_full.columns = clean_cols

    num_cols = []
    cat_cols = []
    text_cols = []

    for col in df_full.columns:
        if pd.api.types.is_numeric_dtype(df_full[col]):
            num_cols.append(col)
        else:
            sample_non_null = df_full[col].dropna().astype(str)
            if len(sample_non_null) > 0 and sample_non_null.str.len().mean() > 40:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    engineered_data = {}

    # 1. Process Numerical Features & Log Transformations
    for c in num_cols:
        vals = pd.to_numeric(df_full[c], errors="coerce").values.astype(np.float32)
        engineered_data[c] = vals
        engineered_data[f"{c}_isna"] = np.isnan(vals).astype(np.float32)

        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals) > 0 and np.min(valid_vals) >= 0:
            if np.nanstd(vals) > 2.0 * (np.nanmean(vals) + 1e-5):
                engineered_data[f"{c}_log"] = np.log1p(np.maximum(0, vals))

    # Domain interaction features
    bath_cols = [c for c in num_cols if "bath" in c.lower()]
    bed_cols = [c for c in num_cols if "bed" in c.lower()]
    lot_cols = [c for c in num_cols if "lot" in c.lower() or "sqft" in c.lower()]
    price_cols = [c for c in num_cols if "price" in c.lower() or "tax" in c.lower()]

    if bath_cols and bed_cols:
        b = np.nan_to_num(engineered_data[bath_cols[0]], nan=0.0)
        bd = np.nan_to_num(engineered_data[bed_cols[0]], nan=0.0)
        engineered_data["bath_bed_ratio"] = b / (bd + 1.0)

    if price_cols and bed_cols:
        p = np.nan_to_num(engineered_data[price_cols[0]], nan=0.0)
        bd = np.nan_to_num(engineered_data[bed_cols[0]], nan=0.0)
        engineered_data["price_bed_ratio"] = p / (bd + 1.0)

    if lot_cols and bed_cols:
        lt = np.nan_to_num(engineered_data[lot_cols[0]], nan=0.0)
        bd = np.nan_to_num(engineered_data[bed_cols[0]], nan=0.0)
        engineered_data["lot_bed_ratio"] = lt / (bd + 1.0)

    # 2. Process Categorical Features
    for c in cat_cols:
        s_train = df_full[c].iloc[:n_train].astype(str).fillna("missing")
        s_full = df_full[c].astype(str).fillna("missing")
        freq_map = s_train.value_counts().to_dict()
        engineered_data[f"{c}_freq"] = s_full.map(freq_map).fillna(0).values.astype(np.float32)
        codes, _ = pd.factorize(s_full)
        engineered_data[f"{c}_code"] = codes.astype(np.float32)

    # 3. Text Features via TF-IDF & TruncatedSVD (Fit on Train, Transform Full)
    if text_cols:
        combined_text = df_full[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        combined_text_train = combined_text.iloc[:n_train]
        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_mat_train = tfidf.fit_transform(combined_text_train)
        tfidf_mat_full = tfidf.transform(combined_text)
        n_comp = min(16, tfidf_mat_train.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(tfidf_mat_train)
        text_emb = svd.transform(tfidf_mat_full).astype(np.float32)
        for i in range(n_comp):
            engineered_data[f"text_svd_{i}"] = text_emb[:, i]

    X_df = pd.DataFrame(engineered_data)
    X_full = X_df.values.astype(np.float32)

    X_train = X_full[:n_train]
    X_test = X_full[n_train:]

    return X_train, X_test, y_train_raw, y_train_log, test_ids


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)

    target_col = "Total interior livable area"

    # 1. Feature Engineering
    X_train, X_test, y_train_raw, y_train_log, test_ids = process_and_engineer_features(
        train_path, test_path, target_col
    )

    n_train = len(X_train)
    n_test = len(X_test)

    # Clean array values for tree models
    X_train_tree = np.nan_to_num(X_train, nan=-999.0, posinf=1e6, neginf=-1e6)
    X_test_tree = np.nan_to_num(X_test, nan=-999.0, posinf=1e6, neginf=-1e6)

    # 2. Stratified Cross-Validation & Out-of-Fold Tracking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)
    oof_cb = np.zeros(n_train)
    oof_ridge = np.zeros(n_train)

    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_cb = np.zeros(n_test)
    test_ridge = np.zeros(n_test)

    # Model Hyperparameters
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 1200,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1200,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    cb_params = {
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr_t, y_tr_log = X_train_tree[train_idx], y_train_log[train_idx]
        X_va_t, y_va_log = X_train_tree[val_idx], y_train_log[val_idx]

        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr_t,
            y_tr_log,
            eval_set=[(X_va_t, y_va_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va_t)
        test_lgb += model_lgb.predict(X_test_tree) / kf.n_splits

        # XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr_t, y_tr_log, eval_set=[(X_va_t, y_va_log)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_va_t)
        test_xgb += model_xgb.predict(X_test_tree) / kf.n_splits

        # CatBoost
        model_cb = cb.CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_tr_t,
            y_tr_log,
            eval_set=(X_va_t, y_va_log),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb[val_idx] = model_cb.predict(X_va_t)
        test_cb += model_cb.predict(X_test_tree) / kf.n_splits

        # Ridge Regression
        imputer = SimpleImputer(strategy="median", fill_value=0.0)
        scaler = StandardScaler()

        X_tr_imp = np.nan_to_num(
            imputer.fit_transform(X_train[train_idx]),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        X_va_imp = np.nan_to_num(
            imputer.transform(X_train[val_idx]),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        X_te_imp = np.nan_to_num(
            imputer.transform(X_test), nan=0.0, posinf=1e6, neginf=-1e6
        )

        X_tr_lin = scaler.fit_transform(X_tr_imp)
        X_va_lin = scaler.transform(X_va_imp)
        X_te_lin = scaler.transform(X_te_imp)

        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_tr_lin, y_tr_log)
        oof_ridge[val_idx] = model_ridge.predict(X_va_lin)
        test_ridge += model_ridge.predict(X_te_lin) / kf.n_splits

        fold_blend_log = (
            0.35 * oof_lgb[val_idx]
            + 0.35 * oof_xgb[val_idx]
            + 0.25 * oof_cb[val_idx]
            + 0.05 * oof_ridge[val_idx]
        )
        fold_blend_raw = np.maximum(1.0, np.expm1(np.clip(fold_blend_log, 0, 15)))
        fold_rmse = np.sqrt(np.mean((y_train_raw[val_idx] - fold_blend_raw) ** 2))
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

        gc.collect()

    # 3. Convert OOF & Test Predictions to Raw Scale
    oof_raw_lgb = np.maximum(1.0, np.expm1(np.clip(oof_lgb, 0, 15)))
    oof_raw_xgb = np.maximum(1.0, np.expm1(np.clip(oof_xgb, 0, 15)))
    oof_raw_cb = np.maximum(1.0, np.expm1(np.clip(oof_cb, 0, 15)))
    oof_raw_ridge = np.maximum(1.0, np.expm1(np.clip(oof_ridge, 0, 15)))

    test_raw_lgb = np.maximum(1.0, np.expm1(np.clip(test_lgb, 0, 15)))
    test_raw_xgb = np.maximum(1.0, np.expm1(np.clip(test_xgb, 0, 15)))
    test_raw_cb = np.maximum(1.0, np.expm1(np.clip(test_cb, 0, 15)))
    test_raw_ridge = np.maximum(1.0, np.expm1(np.clip(test_ridge, 0, 15)))

    OOF_matrix = np.column_stack([oof_raw_lgb, oof_raw_xgb, oof_raw_cb, oof_raw_ridge])
    TEST_matrix = np.column_stack(
        [test_raw_lgb, test_raw_xgb, test_raw_cb, test_raw_ridge]
    )

    # 4. SciPy SLSQP Weight Optimization on Raw RMSE Scale
    def ensemble_loss(weights):
        w = np.maximum(0, weights)
        sum_w = np.sum(w)
        if sum_w > 0:
            w = w / sum_w
        pred = OOF_matrix @ w
        return np.sqrt(np.mean((y_train_raw - pred) ** 2))

    init_weights = np.array([0.35, 0.35, 0.25, 0.05])
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        res = minimize(
            ensemble_loss,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success and np.sum(np.maximum(0, res.x)) > 0:
            optimal_weights = np.maximum(0, res.x)
            optimal_weights /= np.sum(optimal_weights)
        else:
            optimal_weights = init_weights
    except Exception:
        optimal_weights = init_weights

    # 5. Final Out-of-Fold Evaluation & Submission
    final_oof_preds = np.maximum(1.0, OOF_matrix @ optimal_weights)
    final_score = np.sqrt(np.mean((y_train_raw - final_oof_preds) ** 2))

    final_test_preds = np.maximum(1.0, TEST_matrix @ optimal_weights)

    submission_df = pd.DataFrame({"id": test_ids, target_col: final_test_preds})
    submission_df.to_csv(os.path.join(sub_dir, "submission.csv"), index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()