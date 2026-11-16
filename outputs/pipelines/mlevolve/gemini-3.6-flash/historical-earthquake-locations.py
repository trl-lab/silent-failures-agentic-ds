import json
import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Raw Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 2. Identify Target Column
    target_col = "Damage Description (total)"
    for col in train_df.columns:
        if col.strip().lower() == "damage description (total)":
            target_col = col
            break

    print(f"Target column identified: '{target_col}'")

    # Clean target and encode
    train_df = train_df.dropna(subset=[target_col]).copy()
    label_encoder = LabelEncoder()
    train_df["target"] = label_encoder.fit_transform(train_df[target_col].astype(str))

    y = train_df["target"].values
    train_raw = train_df.drop(columns=[target_col, "target"])
    test_raw = test_df.copy()

    # 3. Dynamic Column Detection
    def find_column(df, keywords):
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None

    lat_col = find_column(train_raw, ["latitude", "lat"])
    lon_col = find_column(train_raw, ["longitude", "lon", "long"])
    depth_col = find_column(train_raw, ["depth", "focal"])
    mag_col = find_column(train_raw, ["magnitude", "mag", "primary magnitude"])
    year_col = find_column(train_raw, ["year"])

    # Build Domain Physics & Geospatial Features
    def build_domain_features(df):
        feat = pd.DataFrame(index=df.index)

        # 3D Geographic Coordinates
        if lat_col and lon_col:
            lat_rad = np.radians(pd.to_numeric(df[lat_col], errors="coerce").fillna(0))
            lon_rad = np.radians(pd.to_numeric(df[lon_col], errors="coerce").fillna(0))
            feat["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            feat["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            feat["coord_z"] = np.sin(lat_rad)
            feat["lat_raw"] = pd.to_numeric(df[lat_col], errors="coerce").fillna(0)
            feat["lon_raw"] = pd.to_numeric(df[lon_col], errors="coerce").fillna(0)

        # Physical Energy & Attenuation
        m = (
            pd.to_numeric(df[mag_col], errors="coerce").fillna(0)
            if mag_col
            else pd.Series(0, index=df.index)
        )
        d = (
            pd.to_numeric(df[depth_col], errors="coerce").fillna(0)
            if depth_col
            else pd.Series(0, index=df.index)
        )

        feat["mag"] = m
        feat["depth"] = d
        feat["log_seismic_energy"] = 1.5 * m
        feat["seismic_energy_proxy"] = 10.0 ** (1.5 * np.clip(m, 0, 10))
        feat["hypocenter_dist_proxy"] = np.sqrt(np.maximum(0, d**2 + 100.0))
        feat["energy_attenuation_proxy"] = feat["seismic_energy_proxy"] / (d**2 + 100.0)
        feat["mag_depth_ratio"] = m / (d + 1.0)
        feat["mag_depth_product"] = m * d
        feat["pga_proxy"] = (10.0 ** (0.5 * m)) / ((d**2 + 100.0) ** 0.75)
        feat["pga_proxy_log"] = 0.5 * m - 0.75 * np.log10(d**2 + 100.0)

        # Temporal Features
        if year_col:
            y_val = pd.to_numeric(df[year_col], errors="coerce").fillna(1970)
            feat["year"] = y_val
            feat["is_modern_era"] = (y_val >= 1970).astype(float)
            feat["century"] = (y_val // 100).astype(float)
            feat["decade"] = (y_val // 10).astype(float)

        # Indicator for missing values
        for col in [lat_col, lon_col, depth_col, mag_col]:
            if col and col in df.columns:
                feat[f"{col}_isna"] = df[col].isna().astype(float)

        return feat

    X_train_phys = build_domain_features(train_raw)
    X_test_phys = build_domain_features(test_raw)

    # 4. Spatial Clustering (Fitted strictly on Train)
    spatial_cols = [
        c
        for c in ["coord_x", "coord_y", "coord_z", "depth"]
        if c in X_train_phys.columns
    ]
    if len(spatial_cols) >= 3:
        scaler_geo = StandardScaler()
        train_geo_scaled = scaler_geo.fit_transform(
            X_train_phys[spatial_cols].fillna(0)
        )
        test_geo_scaled = scaler_geo.transform(X_test_phys[spatial_cols].fillna(0))

        kmeans = KMeans(n_clusters=16, random_state=42, n_init=10)
        X_train_phys["spatial_cluster"] = kmeans.fit_predict(train_geo_scaled)
        X_test_phys["spatial_cluster"] = kmeans.predict(test_geo_scaled)

        train_dists = kmeans.transform(train_geo_scaled)
        test_dists = kmeans.transform(test_geo_scaled)
        for k in range(min(5, train_dists.shape[1])):
            X_train_phys[f"dist_cluster_{k}"] = train_dists[:, k]
            X_test_phys[f"dist_cluster_{k}"] = test_dists[:, k]

        # Spatial cluster-level aggregated stats (difference from cluster mean)
        cluster_stats = X_train_phys.groupby("spatial_cluster")[["mag", "depth"]].mean()
        cluster_mag_mean = cluster_stats["mag"].to_dict()
        cluster_depth_mean = cluster_stats["depth"].to_dict()

        X_train_phys["cluster_mean_mag"] = X_train_phys["spatial_cluster"].map(cluster_mag_mean)
        X_train_phys["cluster_mean_depth"] = X_train_phys["spatial_cluster"].map(cluster_depth_mean)
        X_train_phys["mag_diff_cluster_mean"] = X_train_phys["mag"] - X_train_phys["cluster_mean_mag"]
        X_train_phys["depth_diff_cluster_mean"] = X_train_phys["depth"] - X_train_phys["cluster_mean_depth"]

        X_test_phys["cluster_mean_mag"] = X_test_phys["spatial_cluster"].map(cluster_mag_mean)
        X_test_phys["cluster_mean_depth"] = X_test_phys["spatial_cluster"].map(cluster_depth_mean)
        X_test_phys["mag_diff_cluster_mean"] = X_test_phys["mag"] - X_test_phys["cluster_mean_mag"]
        X_test_phys["depth_diff_cluster_mean"] = X_test_phys["depth"] - X_test_phys["cluster_mean_depth"]

    # 5. Process Categorical & Numerical Features
    cat_cols = train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = train_raw.select_dtypes(include=[np.number]).columns.tolist()

    num_medians = train_raw[num_cols].median().to_dict() if len(num_cols) > 0 else {}

    X_train_num = pd.DataFrame(index=train_raw.index)
    X_test_num = pd.DataFrame(index=test_raw.index)

    for c in num_cols:
        med = num_medians.get(c, 0)
        X_train_num[f"{c}_clean"] = pd.to_numeric(train_raw[c], errors="coerce").fillna(
            med
        )
        X_test_num[f"{c}_clean"] = pd.to_numeric(test_raw[c], errors="coerce").fillna(
            med
        )

    X_train_cat = pd.DataFrame(index=train_raw.index)
    X_test_cat = pd.DataFrame(index=test_raw.index)

    for c in cat_cols:
        freq_map = train_raw[c].astype(str).value_counts(normalize=True).to_dict()
        X_train_cat[f"{c}_freq"] = train_raw[c].astype(str).map(freq_map).fillna(0.0)
        X_test_cat[f"{c}_freq"] = test_raw[c].astype(str).map(freq_map).fillna(0.0)

        le_cat = LabelEncoder()
        combined = pd.concat(
            [train_raw[c].astype(str), test_raw[c].astype(str)], axis=0
        )
        le_cat.fit(combined)
        X_train_cat[f"{c}_encoded"] = le_cat.transform(train_raw[c].astype(str))
        X_test_cat[f"{c}_encoded"] = le_cat.transform(test_raw[c].astype(str))

    # Combine Feature Sets
    X_train_full = pd.concat([X_train_phys, X_train_num, X_train_cat], axis=1)
    X_test_full = pd.concat([X_test_phys, X_test_num, X_test_cat], axis=1)

    X_train_full = X_train_full.loc[:, ~X_train_full.columns.duplicated()]
    X_test_full = X_test_full.loc[:, ~X_test_full.columns.duplicated()]

    train_std = X_train_full.std(numeric_only=True)
    feature_cols = train_std[train_std > 1e-6].index.tolist()

    X = X_train_full[feature_cols].copy().fillna(0).astype(np.float32)
    X_test = X_test_full[feature_cols].copy().fillna(0).astype(np.float32)

    # 6. Cross-Validation Setup
    n_classes = len(np.unique(y))
    n_samples = len(X)
    n_test = len(X_test)
    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_xgb = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_cb = np.zeros((n_samples, n_classes), dtype=np.float64)

    test_lgb = np.zeros((n_test, n_classes), dtype=np.float64)
    test_xgb = np.zeros((n_test, n_classes), dtype=np.float64)
    test_cb = np.zeros((n_test, n_classes), dtype=np.float64)

    print(
        f"Training multi-model ensemble on {n_samples} samples across {n_splits} folds..."
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_va_prob = lgb_model.predict_proba(X_va)
        lgb_test_prob = lgb_model.predict_proba(X_test)

        # Model 2: XGBoost
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_va_prob = xgb_model.predict_proba(X_va)
        xgb_test_prob = xgb_model.predict_proba(X_test)

        # Model 3: CatBoost
        cb_model = cb.CatBoostClassifier(
            loss_function="MultiClass",
            iterations=1000,
            learning_rate=0.04,
            depth=5,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        cb_va_prob = cb_model.predict_proba(X_va)
        cb_test_prob = cb_model.predict_proba(X_test)

        oof_lgb[val_idx] = lgb_va_prob
        oof_xgb[val_idx] = xgb_va_prob
        oof_cb[val_idx] = cb_va_prob

        test_lgb += lgb_test_prob / n_splits
        test_xgb += xgb_test_prob / n_splits
        test_cb += cb_test_prob / n_splits

        fold_va_prob = (lgb_va_prob + xgb_va_prob + cb_va_prob) / 3.0
        fold_acc = accuracy_score(y_va, np.argmax(fold_va_prob, axis=1))
        print(f"Fold {fold + 1}/{n_splits} Accuracy: {fold_acc:.5f}")

    # 7. Validation Evaluation & Post-Processing Optimization
    def objective(params):
        w_lgb, w_xgb, w_cb = params[0], params[1], params[2]
        class_weights = params[3:]

        m_weights = np.maximum(0, np.array([w_lgb, w_xgb, w_cb]))
        s = np.sum(m_weights)
        if s > 0:
            m_weights /= s
        else:
            m_weights = np.array([1/3, 1/3, 1/3])

        blended_oof = (
            m_weights[0] * oof_lgb +
            m_weights[1] * oof_xgb +
            m_weights[2] * oof_cb
        )
        scaled_oof = blended_oof * np.maximum(1e-5, class_weights)
        preds = np.argmax(scaled_oof, axis=1)
        return -accuracy_score(y, preds)

    x0 = np.array([1/3, 1/3, 1/3] + [1.0] * n_classes)

    best_res = None
    best_score = float("inf")

    for seed in range(5):
        np.random.seed(seed)
        init_x = x0.copy()
        if seed > 0:
            init_x[:3] += np.random.uniform(-0.1, 0.1, size=3)
            init_x[3:] += np.random.uniform(-0.2, 0.2, size=n_classes)

        res = minimize(
            objective, init_x, method="Nelder-Mead", options={"maxiter": 2000}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_res = res

    opt_params = best_res.x
    opt_w_lgb, opt_w_xgb, opt_w_cb = opt_params[0], opt_params[1], opt_params[2]
    opt_m_weights = np.maximum(0, np.array([opt_w_lgb, opt_w_xgb, opt_w_cb]))
    if np.sum(opt_m_weights) > 0:
        opt_m_weights /= np.sum(opt_m_weights)
    else:
        opt_m_weights = np.array([1/3, 1/3, 1/3])

    opt_class_weights = np.maximum(1e-5, opt_params[3:])

    print(
        f"Optimized Model Weights: LightGBM={opt_m_weights[0]:.4f}, XGBoost={opt_m_weights[1]:.4f}, CatBoost={opt_m_weights[2]:.4f}"
    )
    print(f"Optimized Class Multipliers: {opt_class_weights.round(4)}")

    oof_blended = (
        opt_m_weights[0] * oof_lgb +
        opt_m_weights[1] * oof_xgb +
        opt_m_weights[2] * oof_cb
    )
    oof_preds = np.argmax(oof_blended * opt_class_weights, axis=1)
    final_score = accuracy_score(y, oof_preds)

    test_blended = (
        opt_m_weights[0] * test_lgb +
        opt_m_weights[1] * test_xgb +
        opt_m_weights[2] * test_cb
    )
    test_preds = np.argmax(test_blended * opt_class_weights, axis=1)
    test_labels = label_encoder.inverse_transform(test_preds)

    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_labels)), "Damage Description (total)": test_labels}
    )

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
