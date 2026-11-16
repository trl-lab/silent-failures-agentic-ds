import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def main():
    # Ensure working and submission output directories exist
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Standardize ID column
    id_col = None
    for candidate in ["ID", "id", "Id"]:
        if candidate in df_train.columns:
            id_col = candidate
            break

    target_col = "TARGET"

    if id_col is not None:
        train_ids = df_train[id_col].values
        test_ids = df_test[id_col].values
        df_train_features = df_train.drop(columns=[id_col, target_col])
        df_test_features = df_test.drop(columns=[id_col])
    else:
        train_ids = np.arange(len(df_train))
        test_ids = np.arange(len(df_test))
        df_train_features = df_train.drop(columns=[target_col])
        df_test_features = df_test.copy()

    y_train_full = df_train[target_col].values

    # Stratified Train / Validation Split FIRST to strictly avoid data leakage
    X_tr_raw, X_va_raw, y_tr, y_va, ids_tr, ids_va = train_test_split(
        df_train_features,
        y_train_full,
        train_ids,
        test_size=0.20,
        stratify=y_train_full,
        random_state=42,
    )

    X_te_raw = df_test_features.copy()

    # 2. Sentinel Value Cleaning (-999999 in var3 replaced by mode in train)
    if "var3" in X_tr_raw.columns:
        mode_var3 = X_tr_raw["var3"].replace(-999999, np.nan).mode()[0]
        X_tr_raw["var3"] = X_tr_raw["var3"].replace(-999999, mode_var3)
        X_va_raw["var3"] = X_va_raw["var3"].replace(-999999, mode_var3)
        X_te_raw["var3"] = X_te_raw["var3"].replace(-999999, mode_var3)

    # 3. Remove Zero-Variance (Constant) Features on Training Set ONLY
    stds = X_tr_raw.std(axis=0)
    non_constant_cols = stds[stds > 0].index.tolist()

    X_tr = X_tr_raw[non_constant_cols].copy()
    X_va = X_va_raw[non_constant_cols].copy()
    X_te = X_te_raw[non_constant_cols].copy()

    # 4. Remove Duplicate Features on Training Set ONLY
    sample_rows = min(10000, len(X_tr))
    X_tr_sample = X_tr.iloc[:sample_rows].T
    dup_mask = X_tr_sample.duplicated()
    unique_cols = X_tr_sample.index[~dup_mask].tolist()

    X_tr = X_tr[unique_cols]
    X_va = X_va[unique_cols]
    X_te = X_te[unique_cols]

    # Helper function for engineered features
    def generate_engineered_features(df):
        fe_df = pd.DataFrame(index=df.index)

        # A. Sparsity / Zero Count Statistics
        fe_df["num_zeros"] = (df == 0).sum(axis=1)
        fe_df["num_pos"] = (df > 0).sum(axis=1)
        fe_df["num_neg"] = (df < 0).sum(axis=1)

        # B. Global Numerical Aggregations
        num_vals = df.values
        fe_df["row_mean"] = np.nanmean(num_vals, axis=1)
        fe_df["row_std"] = np.nanstd(num_vals, axis=1)
        fe_df["row_sum"] = np.nansum(num_vals, axis=1)
        fe_df["row_max"] = np.nanmax(num_vals, axis=1)
        fe_df["row_min"] = np.nanmin(num_vals, axis=1)

        # C. Prefix Group Aggregations
        prefixes = ["ind_", "saldo_", "num_", "imp_", "delta_"]
        cols = df.columns
        for prefix in prefixes:
            group_cols = [c for c in cols if c.startswith(prefix)]
            if group_cols:
                sub_vals = df[group_cols].values
                fe_df[f"{prefix}sum"] = np.nansum(sub_vals, axis=1)
                fe_df[f"{prefix}mean"] = np.nanmean(sub_vals, axis=1)
                fe_df[f"{prefix}zeros"] = (sub_vals == 0).sum(axis=1)
                fe_df[f"{prefix}max"] = np.nanmax(sub_vals, axis=1)

        # D. Time Horizon Differences & Ratios (_ult1 vs _ult3)
        ult1_cols = [c for c in cols if c.endswith("_ult1")]
        for col1 in ult1_cols:
            col3 = col1.replace("_ult1", "_ult3")
            if col3 in cols:
                v1 = df[col1].values
                v3 = df[col3].values
                fe_df[f"{col1}_diff_ult3"] = v1 - v3
                fe_df[f"{col1}_ratio_ult3"] = v1 / (np.abs(v3) + 1e-5)

        # E. Age Interaction Features (var15 is Age)
        if "var15" in cols:
            age = df["var15"].values
            fe_df["var15_is_young"] = (age < 23).astype(np.int32)
            fe_df["var15_is_senior"] = (age > 65).astype(np.int32)
            if "saldo_sum" in fe_df.columns:
                fe_df["saldo_per_age"] = fe_df["saldo_sum"] / (age + 1e-5)

        return fe_df

    # Generate engineered features for each split
    fe_tr = generate_engineered_features(X_tr)
    fe_va = generate_engineered_features(X_va)
    fe_te = generate_engineered_features(X_te)

    # Frequency encoding for discrete columns (computed on train ONLY)
    discrete_cols = [c for c in X_tr.columns if X_tr[c].nunique() < 50]
    for col in discrete_cols:
        freq_map = X_tr[col].value_counts(normalize=True).to_dict()
        fe_tr[f"{col}_freq"] = X_tr[col].map(freq_map).fillna(0)
        fe_va[f"{col}_freq"] = X_va[col].map(freq_map).fillna(0)
        fe_te[f"{col}_freq"] = X_te[col].map(freq_map).fillna(0)

    # Combine original cleaned features with engineered features
    X_tr_final = pd.concat([X_tr, fe_tr], axis=1)
    X_va_final = pd.concat([X_va, fe_va], axis=1)
    X_te_final = pd.concat([X_te, fe_te], axis=1)

    # Convert to float32/int32 to optimize computational efficiency
    for c in X_tr_final.columns:
        if X_tr_final[c].dtype == np.float64:
            X_tr_final[c] = X_tr_final[c].astype(np.float32)
            X_va_final[c] = X_va_final[c].astype(np.float32)
            X_te_final[c] = X_te_final[c].astype(np.float32)
        elif X_tr_final[c].dtype == np.int64:
            X_tr_final[c] = X_tr_final[c].astype(np.int32)
            X_va_final[c] = X_va_final[c].astype(np.int32)
            X_te_final[c] = X_te_final[c].astype(np.int32)

    # Save intermediate processed feature datasets
    X_tr_final.to_parquet("./working/train_fe.parquet")
    X_va_final.to_parquet("./working/val_fe.parquet")
    X_te_final.to_parquet("./working/test_fe.parquet")

    # 5. Model Paradigm 1: LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        objective="binary",
        n_jobs=-1,
    )

    lgb_model.fit(
        X_tr_final,
        y_tr,
        eval_set=[(X_va_final, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    lgb_val_preds = lgb_model.predict_proba(X_va_final)[:, 1]
    lgb_test_preds = lgb_model.predict_proba(X_te_final)[:, 1]

    # 6. Model Paradigm 2: XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1.0,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=100,
        tree_method="hist",
        n_jobs=-1,
    )

    xgb_model.fit(X_tr_final, y_tr, eval_set=[(X_va_final, y_va)], verbose=False)

    xgb_val_preds = xgb_model.predict_proba(X_va_final)[:, 1]
    xgb_test_preds = xgb_model.predict_proba(X_te_final)[:, 1]

    # 7. Model Paradigm 3: CatBoost Classifier (if installed)
    has_cat = False
    if HAS_CATBOOST:
        try:
            cat_model = CatBoostClassifier(
                iterations=3000,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=5,
                random_seed=42,
                eval_metric="AUC",
                early_stopping_rounds=100,
                verbose=False,
                thread_count=-1,
            )

            cat_model.fit(X_tr_final, y_tr, eval_set=(X_va_final, y_va), verbose=False)

            cat_val_preds = cat_model.predict_proba(X_va_final)[:, 1]
            cat_test_preds = cat_model.predict_proba(X_te_final)[:, 1]
            has_cat = True
        except Exception:
            has_cat = False

    # 8. Metric-Aligned Rank Percentile Ensembling
    lgb_val_rank = pd.Series(lgb_val_preds).rank(pct=True).values
    xgb_val_rank = pd.Series(xgb_val_preds).rank(pct=True).values
    lgb_test_rank = pd.Series(lgb_test_preds).rank(pct=True).values
    xgb_test_rank = pd.Series(xgb_test_preds).rank(pct=True).values

    if has_cat:
        cat_val_rank = pd.Series(cat_val_preds).rank(pct=True).values
        cat_test_rank = pd.Series(cat_test_preds).rank(pct=True).values

        val_ensemble = 0.40 * lgb_val_rank + 0.35 * xgb_val_rank + 0.25 * cat_val_rank
        test_ensemble = (
            0.40 * lgb_test_rank + 0.35 * xgb_test_rank + 0.25 * cat_test_rank
        )
    else:
        val_ensemble = 0.55 * lgb_val_rank + 0.45 * xgb_val_rank
        test_ensemble = 0.55 * lgb_test_rank + 0.45 * xgb_test_rank

    final_val_auc = roc_auc_score(y_va, val_ensemble)

    # 9. Output Predictions to Official Submission Destination
    submission = pd.DataFrame({"id": test_ids, "TARGET": test_ensemble})
    submission.to_csv("./submission/submission.csv", index=False)

    # Print exact metric as required
    print(f"Final Validation Score: {final_val_auc}")


if __name__ == "__main__":
    main()
