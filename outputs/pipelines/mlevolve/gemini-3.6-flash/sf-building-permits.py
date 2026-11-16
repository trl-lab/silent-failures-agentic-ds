import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw dataset from ./input...")
    train_full = pd.read_parquet("./input/train.parquet")
    test_full = pd.read_parquet("./input/test.parquet")

    # Extract test ids
    test_ids = test_full["id"] if "id" in test_full.columns else test_full.index

    # Target separation
    target_col = "time_to_approve"
    y_full = train_full[target_col].values
    train_df = train_full.drop(columns=[target_col]).copy()
    test_df = test_full.copy()

    # Feature Engineering Pipeline Function
    def extract_features(df):
        df_feat = df.copy()

        # 1. Datetime feature extraction
        date_cols = [c for c in df_feat.columns if "date" in c.lower()]
        for col in date_cols:
            df_feat[col] = pd.to_datetime(df_feat[col], errors="coerce")

        # Pairwise temporal differences (in days)
        for i in range(len(date_cols)):
            for j in range(i + 1, len(date_cols)):
                d1, d2 = date_cols[i], date_cols[j]
                df_feat[f"diff_{d1}_{d2}"] = (
                    df_feat[d2] - df_feat[d1]
                ).dt.total_seconds() / 86400.0

        for col in date_cols:
            df_feat[f"{col}_year"] = df_feat[col].dt.year
            df_feat[f"{col}_month"] = df_feat[col].dt.month
            df_feat[f"{col}_day"] = df_feat[col].dt.day
            df_feat[f"{col}_dayofweek"] = df_feat[col].dt.dayofweek
            df_feat[f"{col}_quarter"] = df_feat[col].dt.quarter

            # Cyclic encodings
            df_feat[f"{col}_month_sin"] = np.sin(
                2 * np.pi * df_feat[f"{col}_month"] / 12
            )
            df_feat[f"{col}_month_cos"] = np.cos(
                2 * np.pi * df_feat[f"{col}_month"] / 12
            )
            df_feat[f"{col}_dow_sin"] = np.sin(
                2 * np.pi * df_feat[f"{col}_dayofweek"] / 7
            )
            df_feat[f"{col}_dow_cos"] = np.cos(
                2 * np.pi * df_feat[f"{col}_dayofweek"] / 7
            )

            df_feat.drop(columns=[col], inplace=True)

        # 2. Construction Domain Feature Interactions
        col_map = {c.lower().replace(" ", "_"): c for c in df_feat.columns}

        prop_stories = col_map.get("number_of_proposed_stories", None)
        exist_stories = col_map.get("number_of_existing_stories", None)
        if prop_stories and exist_stories:
            df_feat["delta_stories"] = df_feat[prop_stories] - df_feat[exist_stories]

        prop_units = col_map.get("proposed_units", None)
        exist_units = col_map.get("existing_units", None)
        if prop_units and exist_units:
            df_feat["delta_units"] = df_feat[prop_units] - df_feat[exist_units]

        est_cost = col_map.get("estimated_cost", None)
        rev_cost = col_map.get("revised_cost", None)
        if est_cost and rev_cost:
            df_feat["cost_diff"] = df_feat[rev_cost] - df_feat[est_cost]
            df_feat["cost_ratio"] = df_feat[rev_cost] / (df_feat[est_cost] + 1.0)

        if est_cost and prop_stories:
            df_feat["cost_per_story"] = df_feat[est_cost] / (
                df_feat[prop_stories] + 1.0
            )
        if est_cost and prop_units:
            df_feat["cost_per_unit"] = df_feat[est_cost] / (df_feat[prop_units] + 1.0)

        # 3. Log-transform skewed continuous numerical features
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if df_feat[col].skew() > 1.5:
                df_feat[f"{col}_log1p"] = np.log1p(
                    np.maximum(0, df_feat[col].fillna(0))
                )

        return df_feat

    print("Extracting domain and temporal features...")
    X_train_fe = extract_features(train_df)
    X_test_fe = extract_features(test_df)

    # Missing value imputation using train medians
    num_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
    train_medians = X_train_fe[num_cols].median()
    X_train_fe[num_cols] = X_train_fe[num_cols].fillna(train_medians)
    X_test_fe[num_cols] = X_test_fe[num_cols].fillna(train_medians)

    # Categorical Frequency and Label Encoding
    cat_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train_fe[col] = X_train_fe[col].astype(str).fillna("Missing")
        X_test_fe[col] = X_test_fe[col].astype(str).fillna("Missing")

        # Frequency Encoding computed strictly on train set
        freq_map = X_train_fe[col].value_counts(normalize=True).to_dict()
        X_train_fe[f"{col}_freq"] = X_train_fe[col].map(freq_map).fillna(0.0)
        X_test_fe[f"{col}_freq"] = X_test_fe[col].map(freq_map).fillna(0.0)

        # Label Encoding
        le = LabelEncoder()
        le.fit(list(set(X_train_fe[col]).union(set(X_test_fe[col]))))
        X_train_fe[col] = le.transform(X_train_fe[col])
        X_test_fe[col] = le.transform(X_test_fe[col])

    # Align columns
    feature_names = [c for c in X_train_fe.columns if c != "id"]
    X_full = X_train_fe[feature_names].reset_index(drop=True)
    X_test = X_test_fe[feature_names].reset_index(drop=True)

    print(f"Feature matrix ready: Train={X_full.shape}, Test={X_test.shape}")

    # Target log-transformation for robust training against right-skewness
    y_full_log = np.log1p(np.maximum(0, y_full))
    n_samples = len(X_full)

    # 5-Fold Cross Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cb = np.zeros(n_samples)

    test_lgb_folds = np.zeros(len(X_test))
    test_xgb_folds = np.zeros(len(X_test))
    test_cb_folds = np.zeros(len(X_test))

    print("Starting 5-Fold Cross-Validation training ensemble...")

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_full, y_full_log)):
        X_trn, y_trn_log = X_full.iloc[trn_idx], y_full_log[trn_idx]
        X_v, y_v_log = X_full.iloc[val_idx], y_full_log[val_idx]

        # LightGBM Model
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_trn,
            y_trn_log,
            eval_set=[(X_v, y_v_log)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_lgb[val_idx] = lgb_model.predict(X_v)
        test_lgb_folds += lgb_model.predict(X_test) / kf.n_splits

        # XGBoost Model
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=50,
            eval_metric="rmse",
            n_jobs=-1,
        )
        xgb_model.fit(X_trn, y_trn_log, eval_set=[(X_v, y_v_log)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_v)
        test_xgb_folds += xgb_model.predict(X_test) / kf.n_splits

        # CatBoost Model
        cb_model = cb.CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=0,
        )
        cb_model.fit(X_trn, y_trn_log, eval_set=(X_v, y_v_log), verbose=False)
        oof_cb[val_idx] = cb_model.predict(X_v)
        test_cb_folds += cb_model.predict(X_test) / kf.n_splits

        print(f"Fold {fold + 1}/5 completed.")

    # Constrained Ensemble Weight Optimization on OOF predictions
    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred_log = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb
        pred_orig = np.clip(np.expm1(pred_log), 0, None)
        return np.sqrt(mean_squared_error(y_full, pred_orig))

    initial_weights = [0.333, 0.333, 0.334]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

    res = minimize(
        blend_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x / np.sum(res.x)

    print(
        f"Optimized OOF Ensemble Weights: LightGBM={opt_weights[0]:.4f}, XGBoost={opt_weights[1]:.4f}, CatBoost={opt_weights[2]:.4f}"
    )

    # Compute Final Out-Of-Fold Validation Score
    oof_final_log = (
        opt_weights[0] * oof_lgb + opt_weights[1] * oof_xgb + opt_weights[2] * oof_cb
    )
    oof_final_orig = np.clip(np.expm1(oof_final_log), 0, None)
    val_score = np.sqrt(mean_squared_error(y_full, oof_final_orig))

    # Test Prediction Inference
    test_final_log = (
        opt_weights[0] * test_lgb_folds
        + opt_weights[1] * test_xgb_folds
        + opt_weights[2] * test_cb_folds
    )
    test_final_orig = np.clip(np.expm1(test_final_log), 0, None)

    # Save Submission CSV
    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame({"id": test_ids, "time_to_approve": test_final_orig})
    sub_df.to_csv(submission_path, index=False)
    print(f"Saved final submission to {submission_path} with shape {sub_df.shape}")

    # Output Final Validation Score as required
    print(f"Final Validation Score: {val_score}")


if __name__ == "__main__":
    main()
