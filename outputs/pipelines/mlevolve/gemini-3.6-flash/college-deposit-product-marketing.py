import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


def run_pipeline():
    # 1. Load Parquet Datasets
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Input parquet files not found in ./input directory.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "TOTAL_ACTIVE_ACCOUNTS_"
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' missing from train dataframe.")

    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    test_ids = test_df["id"].values.astype(int)

    # Separate Target
    y_raw = train_df[target_col].values
    y_log = np.log1p(np.maximum(0, y_raw))

    train_df_features = train_df.drop(columns=[target_col])
    n_train = len(train_df_features)

    # 2. Separate Column Types
    non_feature_cols = ["id"]
    candidate_cols = [c for c in train_df_features.columns if c not in non_feature_cols]

    cat_cols = []
    num_cols = []

    for c in candidate_cols:
        if (
            train_df_features[c].dtype == "object"
            or str(train_df_features[c].dtype) == "category"
        ):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # 3. Feature Engineering Pipeline
    df_all = pd.concat([train_df_features, test_df], axis=0, ignore_index=True)
    engineered_df = pd.DataFrame({"id": df_all["id"].values})

    # Process Numerical Features
    for col in num_cols:
        if df_all[col].isnull().sum() > 0:
            engineered_df[f"{col}_isna"] = df_all[col].isnull().astype(int)

        train_median = train_df_features[col].median()
        val_filled = df_all[col].fillna(train_median)
        engineered_df[col] = val_filled

        if (val_filled >= 0).all() and val_filled.max() > 10:
            engineered_df[f"{col}_log1p"] = np.log1p(val_filled)

    # Process Categorical Features & Frequency Encoding
    for col in cat_cols:
        s = df_all[col].astype(str).fillna("Missing")
        engineered_df[col] = s

        freq_map = train_df_features[col].astype(str).value_counts(normalize=True).to_dict()
        engineered_df[f"{col}_freq"] = df_all[col].astype(str).map(freq_map).fillna(0)

    # Quantile Rank Features for Numerical Columns
    for col in num_cols:
        tr_rank = train_df_features[col].rank(pct=True).fillna(0.5)
        te_rank = test_df[col].rank(pct=True).fillna(0.5)
        engineered_df[f"{col}_qrank"] = pd.concat([tr_rank, te_rank], ignore_index=True).values

    # Numeric Ratio and Difference Interactions
    for i in range(len(num_cols)):
        col1 = num_cols[i]
        for j in range(i + 1, len(num_cols)):
            col2 = num_cols[j]
            engineered_df[f"{col1}_diff_{col2}"] = engineered_df[col1] - engineered_df[col2]
            engineered_df[f"{col1}_ratio_{col2}"] = engineered_df[col1] / (engineered_df[col2].abs() + 1e-5)

    # Categorical Aggregations across all categorical columns
    for cat in cat_cols:
        for num in num_cols:
            grp_stats = train_df_features.groupby(cat)[num].agg(["mean", "std", "min", "max", "median"]).reset_index()
            grp_stats.columns = [
                cat,
                f"{num}_by_{cat}_mean",
                f"{num}_by_{cat}_std",
                f"{num}_by_{cat}_min",
                f"{num}_by_{cat}_max",
                f"{num}_by_{cat}_median",
            ]

            merged = df_all[[cat]].merge(grp_stats, on=cat, how="left")
            for stat in ["mean", "std", "min", "max", "median"]:
                col_name = f"{num}_by_{cat}_{stat}"
                engineered_df[col_name] = merged[col_name].fillna(0).values

            engineered_df[f"{num}_diff_{cat}_mean"] = (
                engineered_df[num] - engineered_df[f"{num}_by_{cat}_mean"]
            )
            engineered_df[f"{num}_ratio_{cat}_mean"] = engineered_df[num] / (
                engineered_df[f"{num}_by_{cat}_mean"].abs() + 1e-5
            )

    # Out-of-Fold Target Encoding across all categorical columns
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices = np.arange(n_train)

    for cat in cat_cols:
        oof_te = np.zeros(n_train)
        cat_series_train = train_df_features[cat].astype(str).fillna("Missing")

        for tr_idx, val_idx in kf.split(train_indices):
            tr_cat = cat_series_train.iloc[tr_idx]
            tr_y = y_log[tr_idx]

            mapping = (
                pd.Series(tr_y, index=tr_cat.index)
                .groupby(tr_cat)
                .mean()
                .to_dict()
            )
            global_mean = tr_y.mean()

            val_cat = cat_series_train.iloc[val_idx]
            oof_te[val_idx] = val_cat.map(mapping).fillna(global_mean).values

        engineered_df.loc[: n_train - 1, f"{cat}_te"] = oof_te

        full_mapping = (
            pd.Series(y_log, index=cat_series_train.index)
            .groupby(cat_series_train)
            .mean()
            .to_dict()
        )
        global_mean_full = y_log.mean()
        test_cat = test_df[cat].astype(str).fillna("Missing")
        engineered_df.loc[n_train:, f"{cat}_te"] = (
            test_cat.map(full_mapping).fillna(global_mean_full).values
        )

    # One-Hot Encoding for Low-Cardinality Categoricals
    final_cat_cols = [c for c in cat_cols if df_all[c].nunique() < 20]
    if final_cat_cols:
        engineered_df = pd.get_dummies(
            engineered_df, columns=final_cat_cols, drop_first=True
        )

    remaining_str_cols = [
        c for c in engineered_df.columns if engineered_df[c].dtype == "object"
    ]
    engineered_df = engineered_df.drop(columns=remaining_str_cols)
    engineered_df = engineered_df.loc[:, ~engineered_df.columns.duplicated()]
    engineered_df = engineered_df.fillna(0)

    # Split into Train and Test Arrays
    X_train_full = engineered_df.iloc[:n_train].copy()
    X_test = engineered_df.iloc[n_train:].copy()

    feature_cols = [c for c in X_train_full.columns if c != "id"]

    for col in feature_cols:
        X_train_full[col] = pd.to_numeric(X_train_full[col], errors="coerce").fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)

    X_tr_np = np.nan_to_num(
        X_train_full[feature_cols].values.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    X_te_np = np.nan_to_num(
        X_test[feature_cols].values.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # 4. Cross-Validation Split & Modeling Setup
    n_samples = len(X_tr_np)
    n_test = len(X_te_np)
    n_folds = 5
    max_y_log = float(y_log.max())

    folds = np.zeros(n_samples, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(kf.split(X_tr_np)):
        folds[val_idx] = fold_idx

    model_names = ["lgb_log", "lgb_huber_log", "xgb_log", "lgb_raw", "xgb_raw"]
    if HAS_CATBOOST:
        model_names.extend(["cb_log", "cb_raw"])
    else:
        model_names.extend(["et_log", "et_raw"])

    oof_preds = {name: np.zeros(n_samples) for name in model_names}
    test_preds = {name: np.zeros(n_test) for name in model_names}

    # 5. Training Loop
    for fold in range(n_folds):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr, y_tr_log, y_tr_raw = X_tr_np[tr_idx], y_log[tr_idx], y_raw[tr_idx]
        X_va, y_va_log, y_va_raw = X_tr_np[va_idx], y_log[va_idx], y_raw[va_idx]

        # 1) LightGBM MSE (Log Target)
        model_lgb1 = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb1.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        p_va_1 = model_lgb1.predict(X_va)
        p_te_1 = model_lgb1.predict(X_te_np)
        oof_preds["lgb_log"][va_idx] = np.maximum(
            0, np.expm1(np.clip(p_va_1, 0, max_y_log * 1.1))
        )
        test_preds["lgb_log"] += (
            np.maximum(0, np.expm1(np.clip(p_te_1, 0, max_y_log * 1.1))) / n_folds
        )

        # 2) LightGBM Huber (Log Target)
        model_lgb2 = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.9,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=100 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb2.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        p_va_2 = model_lgb2.predict(X_va)
        p_te_2 = model_lgb2.predict(X_te_np)
        oof_preds["lgb_huber_log"][va_idx] = np.maximum(
            0, np.expm1(np.clip(p_va_2, 0, max_y_log * 1.1))
        )
        test_preds["lgb_huber_log"] += (
            np.maximum(0, np.expm1(np.clip(p_te_2, 0, max_y_log * 1.1))) / n_folds
        )

        # 3) XGBoost MSE (Log Target)
        model_xgb1 = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            eval_metric="rmse",
        )
        model_xgb1.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
        p_va_3 = model_xgb1.predict(X_va)
        p_te_3 = model_xgb1.predict(X_te_np)
        oof_preds["xgb_log"][va_idx] = np.maximum(
            0, np.expm1(np.clip(p_va_3, 0, max_y_log * 1.1))
        )
        test_preds["xgb_log"] += (
            np.maximum(0, np.expm1(np.clip(p_te_3, 0, max_y_log * 1.1))) / n_folds
        )

        # 4) LightGBM MSE (Raw Target)
        model_lgb_raw = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=200 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb_raw.fit(
            X_tr,
            y_tr_raw,
            eval_set=[(X_va, y_va_raw)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        p_va_raw_1 = model_lgb_raw.predict(X_va)
        p_te_raw_1 = model_lgb_raw.predict(X_te_np)
        oof_preds["lgb_raw"][va_idx] = np.maximum(0, p_va_raw_1)
        test_preds["lgb_raw"] += np.maximum(0, p_te_raw_1) / n_folds

        # 5) XGBoost MSE (Raw Target)
        model_xgb_raw = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=200 + fold,
            n_jobs=-1,
            eval_metric="rmse",
        )
        model_xgb_raw.fit(X_tr, y_tr_raw, eval_set=[(X_va, y_va_raw)], verbose=False)
        p_va_raw_2 = model_xgb_raw.predict(X_va)
        p_te_raw_2 = model_xgb_raw.predict(X_te_np)
        oof_preds["xgb_raw"][va_idx] = np.maximum(0, p_va_raw_2)
        test_preds["xgb_raw"] += np.maximum(0, p_te_raw_2) / n_folds

        # 6) CatBoost / ExtraTrees (Log & Raw Targets)
        if HAS_CATBOOST:
            # CatBoost Log
            model_cb_log = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                early_stopping_rounds=50,
                random_seed=42 + fold,
                verbose=0,
            )
            model_cb_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
            p_va_4 = model_cb_log.predict(X_va)
            p_te_4 = model_cb_log.predict(X_te_np)
            oof_preds["cb_log"][va_idx] = np.maximum(
                0, np.expm1(np.clip(p_va_4, 0, max_y_log * 1.1))
            )
            test_preds["cb_log"] += (
                np.maximum(0, np.expm1(np.clip(p_te_4, 0, max_y_log * 1.1))) / n_folds
            )

            # CatBoost Raw
            model_cb_raw = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                early_stopping_rounds=50,
                random_seed=200 + fold,
                verbose=0,
            )
            model_cb_raw.fit(X_tr, y_tr_raw, eval_set=(X_va, y_va_raw), verbose=False)
            p_va_raw_3 = model_cb_raw.predict(X_va)
            p_te_raw_3 = model_cb_raw.predict(X_te_np)
            oof_preds["cb_raw"][va_idx] = np.maximum(0, p_va_raw_3)
            test_preds["cb_raw"] += np.maximum(0, p_te_raw_3) / n_folds
        else:
            # ExtraTrees Log
            model_et_log = ExtraTreesRegressor(
                n_estimators=200, max_depth=12, random_state=42 + fold, n_jobs=-1
            )
            model_et_log.fit(X_tr, y_tr_log)
            p_va_4 = model_et_log.predict(X_va)
            p_te_4 = model_et_log.predict(X_te_np)
            oof_preds["et_log"][va_idx] = np.maximum(
                0, np.expm1(np.clip(p_va_4, 0, max_y_log * 1.1))
            )
            test_preds["et_log"] += (
                np.maximum(0, np.expm1(np.clip(p_te_4, 0, max_y_log * 1.1))) / n_folds
            )

            # ExtraTrees Raw
            model_et_raw = ExtraTreesRegressor(
                n_estimators=200, max_depth=12, random_state=200 + fold, n_jobs=-1
            )
            model_et_raw.fit(X_tr, y_tr_raw)
            p_va_raw_3 = model_et_raw.predict(X_va)
            p_te_raw_3 = model_et_raw.predict(X_te_np)
            oof_preds["et_raw"][va_idx] = np.maximum(0, p_va_raw_3)
            test_preds["et_raw"] += np.maximum(0, p_te_raw_3) / n_folds

    # 6. Ensemble Blending Optimization
    oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
    test_matrix = np.column_stack([test_preds[m] for m in model_names])

    def raw_rmse_objective(weights):
        w_norm = weights / np.maximum(1e-8, np.sum(weights))
        blend_pred = np.maximum(0, np.dot(oof_matrix, w_norm))
        return np.sqrt(mean_squared_error(y_raw, blend_pred))

    initial_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]

    opt_res = minimize(
        raw_rmse_objective,
        initial_w,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 300},
    )

    opt_weights = opt_res.x / np.maximum(1e-8, np.sum(opt_res.x))

    final_oof_preds = np.maximum(0, np.dot(oof_matrix, opt_weights))
    final_test_preds = np.maximum(0, np.dot(test_matrix, opt_weights))

    final_rmse = float(np.sqrt(mean_squared_error(y_raw, final_oof_preds)))

    # 7. Output Final Predictions
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {"id": test_ids, "TOTAL_ACTIVE_ACCOUNTS_": final_test_preds}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    run_pipeline()