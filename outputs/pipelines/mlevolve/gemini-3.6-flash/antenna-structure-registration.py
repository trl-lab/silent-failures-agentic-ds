import json
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")


def compute_rmse(y_true, y_pred):
    """Computes exact Root Mean Squared Error metric."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # 1. Setup paths and directories
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Load raw datasets
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Strucht"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    y_train = train_df[target_col].values.astype(np.float64)

    # Ensure integer id column
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    test_ids = test_df["id"].values

    n_train = len(train_df)
    X_train_raw = train_df.drop(columns=[target_col])
    X_test_raw = test_df.copy()

    combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

    # 2. Feature Engineering
    features_df = pd.DataFrame(index=combined.index)
    features_df["id"] = combined["id"].values

    lat_col = next((c for c in combined.columns if "lat" in c.lower()), None)
    lon_col = next(
        (c for c in combined.columns if "lon" in c.lower() or "long" in c.lower()), None
    )
    elev_col = next(
        (c for c in combined.columns if "elev" in c.lower() or "ground" in c.lower()),
        None,
    )

    # Spatial Geodetic & ECEF 3D Coordinates
    if lat_col and lon_col:
        lat_vals = combined[lat_col].fillna(combined[lat_col].median()).values
        lon_vals = combined[lon_col].fillna(combined[lon_col].median()).values

        lat_rad = np.radians(lat_vals)
        lon_rad = np.radians(lon_vals)

        R = 6371.0
        features_df["ecef_x"] = R * np.cos(lat_rad) * np.cos(lon_rad)
        features_df["ecef_y"] = R * np.cos(lat_rad) * np.sin(lon_rad)
        features_df["ecef_z"] = R * np.sin(lat_rad)

        features_df["lat_raw"] = combined[lat_col]
        features_df["lon_raw"] = combined[lon_col]

        for angle in [15, 30, 45, 60]:
            rad = np.radians(angle)
            features_df[f"rot_lat_{angle}"] = lat_vals * np.cos(
                rad
            ) - lon_vals * np.sin(rad)
            features_df[f"rot_lon_{angle}"] = lat_vals * np.sin(
                rad
            ) + lon_vals * np.cos(rad)

        features_df["grid_005_lat"] = np.round(lat_vals / 0.05) * 0.05
        features_df["grid_005_lon"] = np.round(lon_vals / 0.05) * 0.05
        features_df["grid_01_lat"] = np.round(lat_vals / 0.1) * 0.1
        features_df["grid_01_lon"] = np.round(lon_vals / 0.1) * 0.1
        features_df["grid_05_lat"] = np.round(lat_vals / 0.5) * 0.5
        features_df["grid_05_lon"] = np.round(lon_vals / 0.5) * 0.5

        conus_lat, conus_lon = 39.8283, -98.5795
        features_df["dist_conus_center"] = np.sqrt(
            (lat_vals - conus_lat) ** 2 + (lon_vals - conus_lon) ** 2
        )

    # Elevation Features
    if elev_col:
        features_df["elev_raw"] = combined[elev_col]
        features_df["elev_log"] = np.log1p(np.maximum(0, combined[elev_col].fillna(0)))
        features_df["elev_is_missing"] = combined[elev_col].isna().astype(int)

        elev_series = combined[elev_col].fillna(combined[elev_col].median())
        if lat_col and lon_col:
            for grid_prefix in ["grid_01", "grid_05"]:
                lat_g = features_df[f"{grid_prefix}_lat"]
                lon_g = features_df[f"{grid_prefix}_lon"]
                grid_group = pd.Series(zip(lat_g, lon_g), index=features_df.index)
                g_mean = elev_series.groupby(grid_group).transform("mean")
                g_min = elev_series.groupby(grid_group).transform("min")
                g_max = elev_series.groupby(grid_group).transform("max")

                features_df[f"elev_rel_{grid_prefix}_mean"] = elev_series - g_mean
                features_df[f"elev_rel_{grid_prefix}_min"] = elev_series - g_min
                features_df[f"elev_rel_{grid_prefix}_max"] = elev_series - g_max

    # Keyword Extraction
    string_cols = [
        c
        for c in combined.columns
        if combined[c].dtype == "object" or combined[c].dtype == "string"
    ]
    keywords = [
        "tower",
        "pole",
        "mast",
        "monopole",
        "building",
        "water",
        "tank",
        "silo",
        "stack",
        "sign",
        "lattice",
        "guyed",
    ]

    for c in string_cols:
        col_str = combined[c].astype(str).str.lower()
        for kw in keywords:
            features_df[f"kw_{c}_{kw}"] = col_str.str.contains(kw, regex=False).astype(
                int
            )

    # TF-IDF Vectorization and TruncatedSVD Text Features (Word & Character N-Gram)
    if string_cols:
        combined_text = (
            combined[string_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        # Word-level TF-IDF
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        tfidf_mat = tfidf.fit_transform(combined_text)
        n_svd_comp = min(15, max(1, tfidf_mat.shape[1] - 1))
        if tfidf_mat.shape[1] > 1:
            svd = TruncatedSVD(n_components=n_svd_comp, random_state=42)
            svd_feats = svd.fit_transform(tfidf_mat)
            for comp_idx in range(n_svd_comp):
                features_df[f"tfidf_svd_{comp_idx}"] = svd_feats[:, comp_idx]

        # Character n-gram (3-5) TF-IDF
        char_tfidf = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=5000, min_df=2
        )
        char_tfidf_mat = char_tfidf.fit_transform(combined_text)
        n_char_svd_comp = min(15, max(1, char_tfidf_mat.shape[1] - 1))
        if char_tfidf_mat.shape[1] > 1:
            char_svd = TruncatedSVD(n_components=n_char_svd_comp, random_state=42)
            char_svd_feats = char_svd.fit_transform(char_tfidf_mat)
            for comp_idx in range(n_char_svd_comp):
                features_df[f"char_tfidf_svd_{comp_idx}"] = char_svd_feats[:, comp_idx]

    # Categorical Frequency Encoding
    cat_cols = [c for c in string_cols if combined[c].nunique() < 1000]
    for c in cat_cols:
        freq = combined[c].value_counts(normalize=True).to_dict()
        features_df[f"freq_{c}"] = combined[c].map(freq).fillna(0)
        features_df[c] = combined[c].astype(str)

    # Remaining Numeric Features
    num_cols = [
        c
        for c in combined.columns
        if c not in string_cols
        and c != "id"
        and c != lat_col
        and c != lon_col
        and c != elev_col
    ]
    for c in num_cols:
        features_df[c] = combined[c]
        features_df[f"{c}_is_missing"] = combined[c].isna().astype(int)

    # Separate train and test sets
    X_train = features_df.iloc[:n_train].copy()
    X_test = features_df.iloc[n_train:].copy()
    X_train[target_col] = y_train

    # Leak-free OOF Target Encodings
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_train["fold"] = -1

    high_card_cat_targets = [c for c in cat_cols if combined[c].nunique() >= 5]
    if lat_col and lon_col:
        high_card_cat_targets.extend(["grid_01_lat", "grid_05_lat"])

    global_mean = float(y_train.mean())
    global_std = float(y_train.std())

    for col in high_card_cat_targets:
        oof_col_mean = f"oof_target_mean_{col}"
        oof_col_std = f"oof_target_std_{col}"

        X_train[oof_col_mean] = global_mean
        X_train[oof_col_std] = global_std
        X_test[oof_col_mean] = global_mean
        X_test[oof_col_std] = global_std

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_train.iloc[val_idx, X_train.columns.get_loc("fold")] = fold
            tr_sub = X_train.iloc[train_idx]

            mean_map = tr_sub.groupby(col)[target_col].mean().to_dict()
            std_map = tr_sub.groupby(col)[target_col].std().to_dict()

            val_series = X_train.iloc[val_idx][col]
            X_train.iloc[val_idx, X_train.columns.get_loc(oof_col_mean)] = (
                val_series.map(mean_map).fillna(global_mean).values
            )
            X_train.iloc[val_idx, X_train.columns.get_loc(oof_col_std)] = (
                val_series.map(std_map).fillna(global_std).values
            )

        full_mean_map = X_train.groupby(col)[target_col].mean().to_dict()
        full_std_map = X_train.groupby(col)[target_col].std().to_dict()

        X_test[oof_col_mean] = X_test[col].map(full_mean_map).fillna(global_mean).values
        X_test[oof_col_std] = X_test[col].map(full_std_map).fillna(global_std).values

    # Out-of-Fold Spatial K-Nearest Neighbors (KNN) Target and Elevation Statistics
    if "ecef_x" in X_train.columns and "ecef_y" in X_train.columns and "ecef_z" in X_train.columns:
        k_list = [5, 15, 50]
        knn_coords_tr = X_train[["ecef_x", "ecef_y", "ecef_z"]].values
        knn_coords_te = X_test[["ecef_x", "ecef_y", "ecef_z"]].values
        elev_tr_vals = (
            combined[elev_col].iloc[:n_train].fillna(combined[elev_col].median()).values
            if elev_col
            else None
        )
        elev_te_vals = (
            combined[elev_col].iloc[n_train:].fillna(combined[elev_col].median()).values
            if elev_col
            else None
        )

        for k_val in k_list:
            for stat in ["mean", "std", "min", "max"]:
                X_train[f"knn_{k_val}_target_{stat}"] = global_mean if stat == "mean" else 0.0
                X_test[f"knn_{k_val}_target_{stat}"] = global_mean if stat == "mean" else 0.0
            if elev_col:
                for stat in ["mean", "min", "max"]:
                    X_train[f"knn_{k_val}_elev_diff_{stat}"] = 0.0
                    X_test[f"knn_{k_val}_elev_diff_{stat}"] = 0.0

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            nn_model = NearestNeighbors(n_neighbors=max(k_list), algorithm="kd_tree")
            nn_model.fit(knn_coords_tr[train_idx])

            distances, indices = nn_model.kneighbors(knn_coords_tr[val_idx])

            for k_val in k_list:
                k_indices = indices[:, :k_val]
                k_targets = y_train[train_idx[k_indices]]

                X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_target_mean")] = np.mean(k_targets, axis=1)
                X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_target_std")] = np.std(k_targets, axis=1)
                X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_target_min")] = np.min(k_targets, axis=1)
                X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_target_max")] = np.max(k_targets, axis=1)

                if elev_col:
                    val_elevs = elev_tr_vals[val_idx]
                    k_elevs = elev_tr_vals[train_idx[k_indices]]
                    X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_elev_diff_mean")] = val_elevs - np.mean(k_elevs, axis=1)
                    X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_elev_diff_min")] = val_elevs - np.min(k_elevs, axis=1)
                    X_train.iloc[val_idx, X_train.columns.get_loc(f"knn_{k_val}_elev_diff_max")] = val_elevs - np.max(k_elevs, axis=1)

        # Full fit for test set
        nn_full = NearestNeighbors(n_neighbors=max(k_list), algorithm="kd_tree")
        nn_full.fit(knn_coords_tr)
        distances_te, indices_te = nn_full.kneighbors(knn_coords_te)

        for k_val in k_list:
            k_indices_te = indices_te[:, :k_val]
            k_targets_te = y_train[k_indices_te]

            X_test[f"knn_{k_val}_target_mean"] = np.mean(k_targets_te, axis=1)
            X_test[f"knn_{k_val}_target_std"] = np.std(k_targets_te, axis=1)
            X_test[f"knn_{k_val}_target_min"] = np.min(k_targets_te, axis=1)
            X_test[f"knn_{k_val}_target_max"] = np.max(k_targets_te, axis=1)

            if elev_col:
                k_elevs_te = elev_tr_vals[k_indices_te]
                X_test[f"knn_{k_val}_elev_diff_mean"] = elev_te_vals - np.mean(k_elevs_te, axis=1)
                X_test[f"knn_{k_val}_elev_diff_min"] = elev_te_vals - np.min(k_elevs_te, axis=1)
                X_test[f"knn_{k_val}_elev_diff_max"] = elev_te_vals - np.max(k_elevs_te, axis=1)

    # Encode categorical columns to ordinal numeric values for tree models
    feature_cols = [c for c in X_train.columns if c not in {"id", target_col, "fold"}]

    object_cols = [
        c
        for c in feature_cols
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category"
    ]
    for col in object_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[col] = encoder.fit_transform(X_train[[col]])
        X_test[col] = encoder.transform(X_test[[col]])

    X_train_full = X_train[feature_cols].copy()
    X_test_full = X_test[feature_cols].copy()
    folds = X_train["fold"].values
    n_folds = int(folds.max() + 1)

    # 3. Model Hyperparameters
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    # 4. K-Fold Training & Cross-Validation
    oof_lgb = np.zeros(len(X_train), dtype=np.float64)
    oof_cb = np.zeros(len(X_train), dtype=np.float64)
    oof_xgb = np.zeros(len(X_train), dtype=np.float64)

    test_preds_lgb = np.zeros(len(X_test), dtype=np.float64)
    test_preds_cb = np.zeros(len(X_test), dtype=np.float64)
    test_preds_xgb = np.zeros(len(X_test), dtype=np.float64)

    for fold in range(n_folds):
        trn_mask = folds != fold
        val_mask = folds == fold

        X_tr, y_tr = X_train_full.loc[trn_mask], y_train[trn_mask]
        X_va, y_va = X_train_full.loc[val_mask], y_train[val_mask]

        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        oof_lgb[val_mask] = model_lgb.predict(X_va)
        test_preds_lgb += model_lgb.predict(X_test_full) / n_folds

        # CatBoost
        model_cb = CatBoostRegressor(**cb_params, early_stopping_rounds=50)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_mask] = model_cb.predict(X_va)
        test_preds_cb += model_cb.predict(X_test_full) / n_folds

        # XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_mask] = model_xgb.predict(X_va)
        test_preds_xgb += model_xgb.predict(X_test_full) / n_folds

        print(
            f"Fold {fold+1}/{n_folds} | LGB RMSE: {compute_rmse(y_va, oof_lgb[val_mask]):.4f} | CB RMSE: {compute_rmse(y_va, oof_cb[val_mask]):.4f} | XGB RMSE: {compute_rmse(y_va, oof_xgb[val_mask]):.4f}"
        )

    # 5. Out-of-Fold Constrained Optimization (SLSQP Blending)
    oof_models = [oof_lgb, oof_cb, oof_xgb]
    test_models = [test_preds_lgb, test_preds_cb, test_preds_xgb]

    def blend_loss(weights):
        blend = np.zeros_like(y_train)
        for w, oof in zip(weights, oof_models):
            blend += w * oof
        blend = np.clip(blend, a_min=0.0, a_max=None)
        return compute_rmse(y_train, blend)

    init_weights = [1.0 / len(oof_models)] * len(oof_models)
    bounds = [(0.0, 1.0) for _ in range(len(oof_models))]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

    opt_res = minimize(
        blend_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = opt_res.x

    final_oof = np.zeros_like(y_train)
    for w, oof in zip(opt_weights, oof_models):
        final_oof += w * oof
    final_oof = np.clip(final_oof, a_min=0.0, a_max=None)

    final_val_score = compute_rmse(y_train, final_oof)

    # 6. Generate Test Submission File
    final_test_preds = np.zeros(len(X_test), dtype=np.float64)
    for w, t_pred in zip(opt_weights, test_models):
        final_test_preds += w * t_pred
    final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

    submission_df = pd.DataFrame(
        {"id": test_ids.astype(int), "Strucht": final_test_preds}
    )
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()