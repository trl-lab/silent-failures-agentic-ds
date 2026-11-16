import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================================================================
# 1. Feature Engineering
# =========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific feature engineering to Steel Plates Faults dataset.
    Extracts geometric dimensions, aspect ratios, luminosity statistics,
    and physical material interactions.
    """
    df = df.copy()

    # Map OpenML V1-V27 attributes to domain feature names if present
    col_map = {
        "V1": "X_Minimum",
        "V2": "X_Maximum",
        "V3": "Y_Minimum",
        "V4": "Y_Maximum",
        "V5": "Pixels_Areas",
        "V6": "X_Perimeter",
        "V7": "Y_Perimeter",
        "V8": "Sum_of_Luminosity",
        "V9": "Minimum_of_Luminosity",
        "V10": "Maximum_of_Luminosity",
        "V11": "Length_of_Conveyer",
        "V12": "TypeOfSteel_A300",
        "V13": "TypeOfSteel_A400",
        "V14": "Steel_Plate_Thickness",
        "V15": "Edges_Index",
        "V16": "Empty_Index",
        "V17": "Square_Index",
        "V18": "Outside_X_Index",
        "V19": "Edges_X_Index",
        "V20": "Edges_Y_Index",
        "V21": "Outside_Global_Index",
        "V22": "LogOfAreas",
        "V23": "Log_X_Index",
        "V24": "Log_Y_Index",
        "V25": "Orientation_Index",
        "V26": "Luminosity_Index",
        "V27": "SigmoidOfAreas",
    }
    rename_dict = {}
    for col in df.columns:
        if col in col_map:
            rename_dict[col] = col_map[col]
        elif col.upper() in col_map:
            rename_dict[col] = col_map[col.upper()]
    if rename_dict:
        df = df.rename(columns=rename_dict)

    eps = 1e-6

    # 1. Geometric & Bounding Box Dimensions
    if "X_Maximum" in df.columns and "X_Minimum" in df.columns:
        df["X_Length"] = df["X_Maximum"] - df["X_Minimum"]
        df["X_Center"] = (df["X_Minimum"] + df["X_Maximum"]) / 2.0
    if "Y_Maximum" in df.columns and "Y_Minimum" in df.columns:
        df["Y_Length"] = df["Y_Maximum"] - df["Y_Minimum"]
        df["Y_Center"] = (df["Y_Minimum"] + df["Y_Maximum"]) / 2.0
    if "X_Length" in df.columns and "Y_Length" in df.columns:
        df["BBox_Area"] = df["X_Length"] * df["Y_Length"]
        df["Aspect_Ratio"] = (df["X_Length"] + eps) / (df["Y_Length"] + eps)
        df["Elongation_Index"] = (df["X_Length"] - df["Y_Length"]) / (
            df["X_Length"] + df["Y_Length"] + eps
        )

    # 2. Shape, Density, & Topological features
    if "Pixels_Areas" in df.columns and "BBox_Area" in df.columns:
        df["BBox_Density"] = df["Pixels_Areas"] / (df["BBox_Area"] + eps)
        df["Log_Area_Diff"] = np.log1p(np.maximum(0, df["BBox_Area"])) - np.log1p(
            np.maximum(0, df["Pixels_Areas"])
        )
    if "X_Perimeter" in df.columns and "Y_Perimeter" in df.columns:
        df["Perimeter_Ratio"] = (df["X_Perimeter"] + eps) / (df["Y_Perimeter"] + eps)
        df["Total_Perimeter"] = df["X_Perimeter"] + df["Y_Perimeter"]
        if "Pixels_Areas" in df.columns:
            df["Perimeter_to_Area"] = df["Total_Perimeter"] / (
                df["Pixels_Areas"] + eps
            )
            df["Compactness"] = (df["Total_Perimeter"] ** 2) / (
                df["Pixels_Areas"] + eps
            )
            # Circularity / Form Factor (4 * pi * Area / Perimeter^2)
            df["Form_Factor"] = (4.0 * np.pi * df["Pixels_Areas"]) / (
                (df["Total_Perimeter"] ** 2) + eps
            )
            # Fractal dimension proxy
            df["Fractal_Dim_Proxy"] = np.log(df["Total_Perimeter"] + 1.0) / (
                np.log(df["Pixels_Areas"] + 1.0) + eps
            )

    # Convexity residuals
    if "BBox_Density" in df.columns and "Empty_Index" in df.columns:
        df["Convexity_Diff"] = df["BBox_Density"] - df["Empty_Index"]
        df["Convexity_Ratio"] = (df["BBox_Density"] + eps) / (df["Empty_Index"] + eps)

    # 3. Luminosity Statistics & Dynamic Contrast
    if "Maximum_of_Luminosity" in df.columns and "Minimum_of_Luminosity" in df.columns:
        df["Luminosity_Range"] = (
            df["Maximum_of_Luminosity"] - df["Minimum_of_Luminosity"]
        )
        df["Luminosity_Dynamic_Contrast"] = (
            df["Maximum_of_Luminosity"] - df["Minimum_of_Luminosity"]
        ) / (df["Maximum_of_Luminosity"] + df["Minimum_of_Luminosity"] + eps)
    if "Sum_of_Luminosity" in df.columns and "Pixels_Areas" in df.columns:
        df["Luminosity_Mean_Per_Pixel"] = df["Sum_of_Luminosity"] / (
            df["Pixels_Areas"] + eps
        )
        if "Luminosity_Range" in df.columns:
            df["Luminosity_Spread_Ratio"] = df["Luminosity_Range"] / (
                df["Luminosity_Mean_Per_Pixel"] + eps
            )

    # 4. Conveyer & Spatial Position Relatives
    if "Length_of_Conveyer" in df.columns:
        if "X_Center" in df.columns:
            df["X_Relative_Conveyer"] = df["X_Center"] / (
                df["Length_of_Conveyer"] + eps
            )
        if "X_Length" in df.columns:
            df["X_Length_Relative_Conveyer"] = df["X_Length"] / (
                df["Length_of_Conveyer"] + eps
            )

    # 5. Steel Type & Thickness Interactions
    if "Steel_Plate_Thickness" in df.columns:
        if "Pixels_Areas" in df.columns:
            df["Thickness_to_Area"] = df["Steel_Plate_Thickness"] / (
                df["Pixels_Areas"] + eps
            )
        if "TypeOfSteel_A300" in df.columns and "TypeOfSteel_A400" in df.columns:
            df["Steel_Type_Composite"] = (
                df["TypeOfSteel_A300"] * 1 + df["TypeOfSteel_A400"] * 2
            )
            df["Thickness_x_A300"] = (
                df["Steel_Plate_Thickness"] * df["TypeOfSteel_A300"]
            )
            df["Thickness_x_A400"] = (
                df["Steel_Plate_Thickness"] * df["TypeOfSteel_A400"]
            )

    # 6. Non-linear & Edge Multipliers
    if "Edges_X_Index" in df.columns and "Edges_Y_Index" in df.columns:
        df["Edge_Interaction"] = df["Edges_X_Index"] * df["Edges_Y_Index"]
        df["Edges_Ratio"] = (df["Edges_X_Index"] + eps) / (df["Edges_Y_Index"] + eps)

    return df


# =========================================================================
# 2. Diverse Model Family Builders (GBDTs, ExtraTrees, RandomForest)
# =========================================================================
def build_lgbm_classifier(
    num_classes: int = 7,
    seed: int = 42,
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=15,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb_classifier(
    num_classes: int = 7,
    seed: int = 42,
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def build_catboost_classifier(
    num_classes: int = 7,
    seed: int = 42,
    iterations: int = 1500,
    learning_rate: float = 0.03,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


def build_extratrees_classifier(
    seed: int = 42,
    n_estimators: int = 500,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def build_randomforest_classifier(
    seed: int = 42,
    n_estimators: int = 500,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


# =========================================================================
# 4. Main Training and Evaluation Pipeline
# =========================================================================
def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Load Data
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "target" if "target" in train_df.columns else [c for c in train_df.columns if c not in test_df.columns][0]
    y_raw = train_df[target_col].copy()
    X_train_raw = train_df.drop(columns=[target_col]).copy()
    X_test_raw = test_df.copy()

    # Encode categorical target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    # 2. Apply Domain Feature Engineering
    X_train_feat = engineer_features(X_train_raw)
    X_test_feat = engineer_features(X_test_raw)

    # Handle numerical stability (inf/nan imputation strictly using train medians)
    X_train_feat = X_train_feat.replace([np.inf, -np.inf], np.nan)
    X_test_feat = X_test_feat.replace([np.inf, -np.inf], np.nan)
    train_medians = X_train_feat.median()
    X_train_feat = X_train_feat.fillna(train_medians)
    X_test_feat = X_test_feat.fillna(train_medians)

    X = X_train_feat.values
    X_test_base = X_test_feat.values
    n_samples = len(X)
    n_test = len(X_test_base)
    n_splits = 10
    seeds = [42, 100, 2024]

    oof_lgb = np.zeros((n_samples, num_classes))
    oof_xgb = np.zeros((n_samples, num_classes))
    oof_cat = np.zeros((n_samples, num_classes))
    oof_et = np.zeros((n_samples, num_classes))
    oof_rf = np.zeros((n_samples, num_classes))

    test_lgb = np.zeros((n_test, num_classes))
    test_xgb = np.zeros((n_test, num_classes))
    test_cat = np.zeros((n_test, num_classes))
    test_et = np.zeros((n_test, num_classes))
    test_rf = np.zeros((n_test, num_classes))

    # 4. Multi-Seed 10-Fold Stratified Cross-Validation
    for seed_idx, seed_val in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_val)
        seed_oof_lgb = np.zeros((n_samples, num_classes))
        seed_oof_xgb = np.zeros((n_samples, num_classes))
        seed_oof_cat = np.zeros((n_samples, num_classes))
        seed_oof_et = np.zeros((n_samples, num_classes))
        seed_oof_rf = np.zeros((n_samples, num_classes))

        seed_test_lgb = np.zeros((n_test, num_classes))
        seed_test_xgb = np.zeros((n_test, num_classes))
        seed_test_cat = np.zeros((n_test, num_classes))
        seed_test_et = np.zeros((n_test, num_classes))
        seed_test_rf = np.zeros((n_test, num_classes))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr_raw, y_tr = X[train_idx], y[train_idx]
            X_va_raw, y_va = X[val_idx], y[val_idx]
            X_te_raw = X_test_base.copy()

            # Fold-isolated KMeans cluster distance features
            km_scaler = StandardScaler()
            X_tr_scaled = km_scaler.fit_transform(X_tr_raw)
            X_va_scaled = km_scaler.transform(X_va_raw)
            X_te_scaled = km_scaler.transform(X_te_raw)

            kmeans = KMeans(n_clusters=num_classes, random_state=seed_val + fold, n_init=10)
            tr_km_dist = kmeans.fit_transform(X_tr_scaled)
            va_km_dist = kmeans.transform(X_va_scaled)
            te_km_dist = kmeans.transform(X_te_scaled)

            X_tr = np.hstack([X_tr_raw, tr_km_dist])
            X_va = np.hstack([X_va_raw, va_km_dist])
            X_test = np.hstack([X_te_raw, te_km_dist])

            # A. LightGBM
            lgb_clf = build_lgbm_classifier(
                num_classes=num_classes, seed=seed_val + fold, n_estimators=1500
            )
            lgb_clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            seed_oof_lgb[val_idx] = lgb_clf.predict_proba(X_va)
            seed_test_lgb += lgb_clf.predict_proba(X_test) / n_splits

            # B. XGBoost
            xgb_clf = build_xgb_classifier(
                num_classes=num_classes, seed=seed_val + fold, n_estimators=1500
            )
            xgb_clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            seed_oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)
            seed_test_xgb += xgb_clf.predict_proba(X_test) / n_splits

            # C. CatBoost
            cat_clf = build_catboost_classifier(
                num_classes=num_classes, seed=seed_val + fold, iterations=1500
            )
            cat_clf.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
            )
            seed_oof_cat[val_idx] = cat_clf.predict_proba(X_va)
            seed_test_cat += cat_clf.predict_proba(X_test) / n_splits

            # D. ExtraTrees
            et_clf = build_extratrees_classifier(
                seed=seed_val + fold, n_estimators=500
            )
            et_clf.fit(X_tr, y_tr)
            seed_oof_et[val_idx] = et_clf.predict_proba(X_va)
            seed_test_et += et_clf.predict_proba(X_test) / n_splits

            # E. RandomForest
            rf_clf = build_randomforest_classifier(
                seed=seed_val + fold, n_estimators=500
            )
            rf_clf.fit(X_tr, y_tr)
            seed_oof_rf[val_idx] = rf_clf.predict_proba(X_va)
            seed_test_rf += rf_clf.predict_proba(X_test) / n_splits

        # Accumulate across seeds
        weight_seed = 1.0 / len(seeds)
        oof_lgb += seed_oof_lgb * weight_seed
        oof_xgb += seed_oof_xgb * weight_seed
        oof_cat += seed_oof_cat * weight_seed
        oof_et += seed_oof_et * weight_seed
        oof_rf += seed_oof_rf * weight_seed

        test_lgb += seed_test_lgb * weight_seed
        test_xgb += seed_test_xgb * weight_seed
        test_cat += seed_test_cat * weight_seed
        test_et += seed_test_et * weight_seed
        test_rf += seed_test_rf * weight_seed

    models_oof = [oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf]
    models_test = [test_lgb, test_xgb, test_cat, test_et, test_rf]
    model_names = ["LightGBM", "XGBoost", "CatBoost", "ExtraTrees", "RandomForest"]

    for name, oof_m in zip(model_names, models_oof):
        acc = accuracy_score(y, np.argmax(oof_m, axis=1))
        print(f"{name} 10-fold Multi-Seed OOF Accuracy: {acc:.5f}")

    # 5. Constrained SLSQP Optimization for Ensemble Blending Weights
    def loss_func(weights):
        w_norm = np.array(weights) / np.sum(weights)
        blend_probs = sum(w * m for w, m in zip(w_norm, models_oof))
        blend_probs = np.clip(blend_probs, 1e-15, 1.0 - 1e-15)
        blend_probs = blend_probs / blend_probs.sum(axis=1, keepdims=True)
        return log_loss(y, blend_probs)

    init_weights = [1.0 / len(models_oof)] * len(models_oof)
    bounds = [(0.0, 1.0) for _ in range(len(models_oof))]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

    opt_res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    opt_weights = opt_res.x / np.sum(opt_res.x)
    print(f"Optimal SLSQP Weights: {dict(zip(model_names, np.round(opt_weights, 4)))}")

    oof_blend = sum(w * m for w, m in zip(opt_weights, models_oof))
    final_score = accuracy_score(y, np.argmax(oof_blend, axis=1))

    # 6. Test Predictions & Submission File Generation
    test_blend = sum(w * m for w, m in zip(opt_weights, models_test))
    test_pred_indices = np.argmax(test_blend, axis=1)
    test_pred_labels = label_encoder.inverse_transform(test_pred_indices)

    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_pred_labels)), "target": test_pred_labels}
    )
    sub_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()