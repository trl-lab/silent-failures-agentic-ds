import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
y = train_df[target_col].copy()
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)


# 2. Enhanced Feature Engineering
def add_features(df):
    df_feat = df.copy()
    eps = 1e-5

    r_col = next((c for c in df.columns if "rawred" in c), None)
    g_col = next((c for c in df.columns if "rawgreen" in c), None)
    b_col = next((c for c in df.columns if "rawblue" in c), None)

    if r_col and g_col and b_col:
        df_feat["rgb_sum"] = df_feat[r_col] + df_feat[g_col] + df_feat[b_col]
        df_feat["r_ratio"] = df_feat[r_col] / (df_feat["rgb_sum"] + eps)
        df_feat["g_ratio"] = df_feat[g_col] / (df_feat["rgb_sum"] + eps)
        df_feat["b_ratio"] = df_feat[b_col] / (df_feat["rgb_sum"] + eps)
        df_feat["r_g_diff"] = df_feat[r_col] - df_feat[g_col]
        df_feat["r_b_diff"] = df_feat[r_col] - df_feat[b_col]
        df_feat["g_b_diff"] = df_feat[g_col] - df_feat[b_col]

        rgb_vals = df_feat[[r_col, g_col, b_col]]
        df_feat["rgb_std"] = rgb_vals.std(axis=1)
        df_feat["rgb_max"] = rgb_vals.max(axis=1)
        df_feat["rgb_min"] = rgb_vals.min(axis=1)
        df_feat["rgb_range"] = df_feat["rgb_max"] - df_feat["rgb_min"]

    exr_col = next((c for c in df.columns if "exred" in c), None)
    exg_col = next((c for c in df.columns if "exgreen" in c), None)
    exb_col = next((c for c in df.columns if "exblue" in c), None)

    if exr_col and exg_col and exb_col:
        df_feat["ex_rg_diff"] = df_feat[exr_col] - df_feat[exg_col]
        df_feat["ex_rb_diff"] = df_feat[exr_col] - df_feat[exb_col]
        df_feat["ex_gb_diff"] = df_feat[exg_col] - df_feat[exb_col]

    vmean = next(
        (c for c in df.columns if "vedge-mean" in c or "vedge_mean" in c), None
    )
    vsd = next(
        (c for c in df.columns if ("vedge" in c or "vegde" in c) and "sd" in c), None
    )
    hmean = next(
        (c for c in df.columns if "hedge-mean" in c or "hedge_mean" in c), None
    )
    hsd = next((c for c in df.columns if "hedge" in c and "sd" in c), None)

    if vmean and vsd:
        df_feat["vedge_snr"] = df_feat[vmean] / (df_feat[vsd] + eps)
    if hmean and hsd:
        df_feat["hedge_snr"] = df_feat[hmean] / (df_feat[hsd] + eps)
    if vmean and hmean:
        df_feat["edge_ratio"] = df_feat[vmean] / (df_feat[hmean] + eps)
        df_feat["total_edge_mean"] = df_feat[vmean] + df_feat[hmean]
        df_feat["edge_magnitude"] = np.sqrt(df_feat[vmean] ** 2 + df_feat[hmean] ** 2)
        df_feat["edge_angle"] = np.arctan2(df_feat[vmean], df_feat[hmean])
    if vsd and hsd:
        df_feat["total_edge_sd"] = df_feat[vsd] + df_feat[hsd]
        df_feat["edge_sd_magnitude"] = np.sqrt(df_feat[vsd] ** 2 + df_feat[hsd] ** 2)

    sld5 = next((c for c in df.columns if "density-5" in c or "density_5" in c), None)
    sld2 = next((c for c in df.columns if "density-2" in c or "density_2" in c), None)
    if sld5 and sld2:
        df_feat["line_density_sum"] = df_feat[sld5] + df_feat[sld2]
        df_feat["line_density_diff"] = df_feat[sld5] - df_feat[sld2]
        df_feat["line_density_ratio"] = df_feat[sld5] / (df_feat[sld2] + eps)

    intensity_col = next((c for c in df.columns if "intensity" in c), None)
    val_col = next((c for c in df.columns if "value" in c), None)
    sat_col = next((c for c in df.columns if "saturat" in c), None)
    hue_col = next((c for c in df.columns if "hue" in c), None)

    if intensity_col and val_col:
        df_feat["intensity_val_diff"] = df_feat[intensity_col] - df_feat[val_col]
    if sat_col and val_col:
        df_feat["chroma"] = df_feat[sat_col] * df_feat[val_col]
    if hue_col:
        df_feat["hue_sin"] = np.sin(df_feat[hue_col])
        df_feat["hue_cos"] = np.cos(df_feat[hue_col])

    return df_feat


X_feat = add_features(X)
X_test_feat = add_features(X_test)

X_feat.columns = [c.replace("-", "_") for c in X_feat.columns]
X_test_feat.columns = [c.replace("-", "_") for c in X_test_feat.columns]

# 3. Multi-Model Stratified K-Fold Training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
num_classes = len(le.classes_)
num_models = 7

oof_model_preds = [np.zeros((len(X_feat), num_classes)) for _ in range(num_models)]
test_model_preds = [
    np.zeros((len(X_test_feat), num_classes)) for _ in range(num_models)
]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y_encoded)):
    X_train, y_train = X_feat.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X_feat.iloc[val_idx], y_encoded[val_idx]

    models = [
        # 0: LightGBM
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=22,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        ),
        # 1: CatBoost
        CatBoostClassifier(
            iterations=450,
            learning_rate=0.03,
            depth=5,
            random_seed=42 + fold,
            verbose=0,
        ),
        # 2: XGBoost
        xgb.XGBClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="mlogloss",
        ),
        # 3: ExtraTrees
        ExtraTreesClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=2,
            random_state=42 + fold,
        ),
        # 4: RandomForest
        RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=2,
            random_state=42 + fold,
        ),
        # 5: HistGradientBoosting
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=22,
            random_state=42 + fold,
        ),
        # 6: Distance-weighted KNN with Feature Scaling
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        ),
    ]

    for m_idx, model in enumerate(models):
        model.fit(X_train, y_train)
        oof_model_preds[m_idx][val_idx] = model.predict_proba(X_val)
        test_model_preds[m_idx] += model.predict_proba(X_test_feat) / n_splits


# 4. Optimal Ensemble Weight Finding via SLSQP
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_oof = np.zeros_like(oof_model_preds[0])
    for i, w in enumerate(weights):
        blend_oof += w * oof_model_preds[i]
    blend_oof = np.clip(blend_oof, 1e-15, 1.0 - 1e-15)
    return log_loss(y_encoded, blend_oof)


init_weights = [1.0 / num_models] * num_models
bounds = [(0.0, 1.0)] * num_models
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
optimal_weights = res.x / np.sum(res.x)

# 5. Out-of-fold Evaluation
final_oof_probs = np.zeros_like(oof_model_preds[0])
final_test_probs = np.zeros_like(test_model_preds[0])

for i, w in enumerate(optimal_weights):
    final_oof_probs += w * oof_model_preds[i]
    final_test_probs += w * test_model_preds[i]

oof_class_preds = np.argmax(final_oof_probs, axis=1)
cv_accuracy = accuracy_score(y_encoded, oof_class_preds)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# 6. Save Submission
final_test_labels = le.inverse_transform(np.argmax(final_test_probs, axis=1))

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "class": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
