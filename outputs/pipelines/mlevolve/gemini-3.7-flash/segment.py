import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. Reproducibility & Directories Setup
# ---------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ---------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df.columns = [re.sub(r"[-\s]+", "_", c).lower() for c in train_df.columns]
test_df.columns = [re.sub(r"[-\s]+", "_", c).lower() for c in test_df.columns]

target_col = "class"
y_raw = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

constant_cols = [col for col in X_train_raw.columns if X_train_raw[col].nunique() <= 1]
if constant_cols:
    X_train_raw = X_train_raw.drop(columns=constant_cols)
    X_test_raw = X_test_raw.drop(columns=constant_cols, errors="ignore")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)


def find_col(df: pd.DataFrame, candidates: list):
    norm_map = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        cand_norm = re.sub(r"[^a-z0-9]", "", cand.lower())
        if cand_norm in norm_map:
            return df[norm_map[cand_norm]].astype(float)
    return None


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    eps = 1e-6

    r = find_col(feat, ["rawred_mean", "rawred", "raw_red_mean", "rawredmean"])
    g = find_col(feat, ["rawgreen_mean", "rawgreen", "raw_green_mean", "rawgreenmean"])
    b = find_col(feat, ["rawblue_mean", "rawblue", "raw_blue_mean", "rawbluemean"])

    intensity = find_col(feat, ["intensity_mean", "intensity", "intensity_avg", "intensitymean"])
    if intensity is None and r is not None and g is not None and b is not None:
        intensity = (r + g + b) / 3.0

    if r is not None and g is not None and b is not None:
        rgb_sum = r + g + b + eps
        feat["r_norm"] = r / rgb_sum
        feat["g_norm"] = g / rgb_sum
        feat["b_norm"] = b / rgb_sum

        feat["rg_opponent"] = r - g
        feat["yb_opponent"] = (r + g) * 0.5 - b
        feat["rb_diff"] = r - b
        feat["gb_diff"] = g - b

        feat["opp_o1"] = (r - g) / np.sqrt(2.0)
        feat["opp_o2"] = (r + g - 2.0 * b) / np.sqrt(6.0)
        feat["opp_o3"] = (r + g + b) / np.sqrt(3.0)
        feat["opp_angle"] = np.arctan2(feat["opp_o1"], feat["opp_o2"] + eps)

        feat["greenness_idx"] = (2.0 * g - r - b) / (rgb_sum)
        feat["redness_idx"] = (2.0 * r - g - b) / (rgb_sum)
        feat["blueness_idx"] = (2.0 * b - r - g) / (rgb_sum)

        r_val, g_val, b_val = r.values, g.values, b.values
        c_max = np.maximum(np.maximum(r_val, g_val), b_val)
        c_min = np.minimum(np.minimum(r_val, g_val), b_val)
        delta = c_max - c_min

        h_arr = np.zeros_like(c_max)
        mask_r = (c_max == r_val) & (delta > eps)
        mask_g = (c_max == g_val) & (~mask_r) & (delta > eps)
        mask_b = (c_max == b_val) & (~mask_r) & (~mask_g) & (delta > eps)

        h_arr[mask_r] = 60.0 * (((g_val[mask_r] - b_val[mask_r]) / (delta[mask_r] + eps)) % 6)
        h_arr[mask_g] = 60.0 * (((b_val[mask_g] - r_val[mask_g]) / (delta[mask_g] + eps)) + 2)
        h_arr[mask_b] = 60.0 * (((r_val[mask_b] - g_val[mask_b]) / (delta[mask_b] + eps)) + 4)

        feat["hsv_hue_sin"] = np.sin(np.radians(h_arr))
        feat["hsv_hue_cos"] = np.cos(np.radians(h_arr))
        feat["hsv_saturation"] = delta / (c_max + eps)
        feat["hsv_value"] = c_max

        rgb_stack = np.stack([r.values, g.values, b.values], axis=1)
        feat["rgb_max"] = c_max
        feat["rgb_min"] = c_min
        feat["chroma"] = delta
        feat["rgb_std"] = np.std(rgb_stack, axis=1)
        if intensity is not None:
            feat["chroma_to_intensity"] = feat["chroma"] / (intensity + eps)
            feat["sat_to_intensity"] = feat["hsv_saturation"] / (intensity + eps)

    if intensity is not None:
        exred = find_col(feat, ["exred_mean", "exred", "excess_red", "exredmean"])
        exgreen = find_col(feat, ["exgreen_mean", "exgreen", "excess_green", "exgreenmean"])
        exblue = find_col(feat, ["exblue_mean", "exblue", "excess_blue", "exbluemean"])
        if exred is not None:
            feat["exred_norm"] = exred / (intensity + eps)
        if exgreen is not None:
            feat["exgreen_norm"] = exgreen / (intensity + eps)
        if exblue is not None:
            feat["exblue_norm"] = exblue / (intensity + eps)

        val_mean = find_col(feat, ["value_mean", "value", "valuemean"])
        if val_mean is not None:
            feat["val_intensity_diff"] = val_mean - intensity

    v_mean = find_col(feat, ["vedge_mean", "vedge", "v_edge_mean", "vedgemean"])
    h_mean = find_col(feat, ["hedge_mean", "hedge", "h_edge_mean", "hedgemean"])
    v_sd = find_col(feat, ["vegde_sd", "vedge_sd", "vegde_std", "vedge_std", "vegdesd", "vedgesd"])
    h_sd = find_col(feat, ["hedge_sd", "hedge_std", "hedgesd"])

    if v_mean is not None and h_mean is not None:
        feat["edge_magnitude_mean"] = np.sqrt(v_mean**2 + h_mean**2)
        feat["edge_total_mean"] = v_mean + h_mean
        feat["edge_anisotropy"] = (v_mean - h_mean) / (v_mean + h_mean + eps)
        feat["edge_ratio_vh"] = (v_mean + eps) / (h_mean + eps)
        feat["edge_angle"] = np.arctan2(v_mean, h_mean + eps)
        feat["edge_gradient_energy"] = v_mean**2 + h_mean**2

    if v_sd is not None and h_sd is not None:
        feat["edge_total_sd"] = v_sd + h_sd
        feat["edge_sd_magnitude"] = np.sqrt(v_sd**2 + h_sd**2)
        feat["edge_sd_energy"] = v_sd**2 + h_sd**2
        if v_mean is not None and h_mean is not None:
            feat["vedge_cov"] = v_sd / (v_mean + eps)
            feat["hedge_cov"] = h_sd / (h_mean + eps)

    sld5 = find_col(feat, ["short_line_density_5", "shortline5", "short_line_5", "shortlinedensity5"])
    sld2 = find_col(feat, ["short_line_density_2", "shortline2", "short_line_2", "shortlinedensity2"])
    if sld5 is not None and sld2 is not None:
        feat["sld_total"] = sld5 + sld2
        feat["sld_ratio"] = (sld2 + eps) / (sld5 + eps)
        feat["sld_product"] = sld5 * sld2
        if v_mean is not None and h_mean is not None:
            feat["edge_to_density_ratio"] = (v_mean + h_mean) / (feat["sld_total"] + eps)
        if intensity is not None:
            feat["sld_intensity_prod"] = feat["sld_total"] * intensity
        if "chroma" in feat.columns:
            feat["sld_chroma_prod"] = feat["sld_total"] * feat["chroma"]

    if intensity is not None and "edge_magnitude_mean" in feat.columns:
        feat["edge_intensity_ratio"] = feat["edge_magnitude_mean"] / (intensity + eps)

    raw_num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    for c in raw_num_cols:
        if (feat[c] >= 0).all() and feat[c].max() > 10:
            feat[f"{c}_log1p"] = np.log1p(feat[c])

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feat


X_train_eng = engineer_domain_features(X_train_raw)
X_test_eng = engineer_domain_features(X_test_raw)
X_test_eng = X_test_eng[list(X_train_eng.columns)]

# ---------------------------------------------------------
# 3. Stratified 10-Fold Cross-Validation & Multi-Model Ensemble
# ---------------------------------------------------------
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

model_names = ["cat", "lgbm", "xgb", "et", "rf", "svc", "knn", "lr"]
oof_model_preds = {name: np.zeros((len(train_df), num_classes), dtype=np.float64) for name in model_names}
test_model_preds = {name: np.zeros((len(test_df), num_classes), dtype=np.float64) for name in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_eng, y_encoded)):
    X_tr_fold = X_train_eng.iloc[train_idx].values
    y_tr_fold = y_encoded[train_idx]
    X_val_fold = X_train_eng.iloc[val_idx].values
    y_val_fold = y_encoded[val_idx]
    X_test_fold = X_test_eng.values

    # Preprocessing: Dual-scaling (Standard + Quantile) + PCA strictly fit on training fold
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_fold)
    X_val_std = scaler.transform(X_val_fold)
    X_test_std = scaler.transform(X_test_fold)

    n_q = min(100, len(X_tr_fold))
    qt = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=RANDOM_SEED)
    X_tr_qt = qt.fit_transform(X_tr_fold)
    X_val_qt = qt.transform(X_val_fold)
    X_test_qt = qt.transform(X_test_fold)

    n_pca = min(6, X_tr_std.shape[1])
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    pca_tr = pca.fit_transform(X_tr_std)
    pca_val = pca.transform(X_val_std)
    pca_test = pca.transform(X_test_std)

    X_tr_final = np.hstack([X_tr_std, X_tr_qt, pca_tr])
    X_val_final = np.hstack([X_val_std, X_val_qt, pca_val])
    X_test_final = np.hstack([X_test_std, X_test_qt, pca_test])

    models = {
        "cat": CatBoostClassifier(
            iterations=550,
            learning_rate=0.03,
            depth=5,
            random_seed=RANDOM_SEED + fold,
            verbose=False,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=450,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED + fold,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=450,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        ),
        "svc": SVC(
            C=5.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED + fold,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
        "lr": LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_tr_final, y_tr_fold)
        val_prob = model.predict_proba(X_val_final)
        test_prob = model.predict_proba(X_test_final)

        oof_model_preds[name][val_idx] = val_prob
        test_model_preds[name] += test_prob / n_splits

    fold_probs_avg = np.mean([oof_model_preds[name][val_idx] for name in model_names], axis=0)
    fold_acc = accuracy_score(y_val_fold, np.argmax(fold_probs_avg, axis=1))
    print(f"Fold {fold + 1:02d}/{n_splits:02d} - Mean Ensemble Val Accuracy: {fold_acc:.4f}")

# ---------------------------------------------------------
# 4. Out-of-Fold Accuracy-Oriented Weight Optimization & Test Inference
# ---------------------------------------------------------
single_accuracies = {}
for name in model_names:
    acc = accuracy_score(y_encoded, np.argmax(oof_model_preds[name], axis=1))
    single_accuracies[name] = acc
    print(f"Individual Model OOF Accuracy ({name}): {acc:.4f}")

best_single_name = max(single_accuracies, key=single_accuracies.get)
best_single_acc = single_accuracies[best_single_name]
print(f"Top Single Model: {best_single_name} ({best_single_acc:.4f})")

def compute_blend_oof(weights):
    w = np.array(weights, dtype=np.float64)
    w = np.maximum(w, 0)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones_like(w) / len(w)
    blend_oof = np.zeros((len(train_df), num_classes), dtype=np.float64)
    for idx, name in enumerate(model_names):
        blend_oof += w[idx] * oof_model_preds[name]
    return blend_oof, w

def accuracy_objective(weights):
    blend_oof, w = compute_blend_oof(weights)
    pred = np.argmax(blend_oof, axis=1)
    acc = accuracy_score(y_encoded, pred)
    # Primary goal is accuracy, with tiny log_loss tie-breaker for continuous smoothness
    clipped_oof = np.clip(blend_oof / (np.sum(blend_oof, axis=1, keepdims=True) + 1e-15), 1e-15, 1.0 - 1e-15)
    loss = -acc + 1e-4 * log_loss(y_encoded, clipped_oof)
    return loss

# Multi-start Nelder-Mead optimization across diverse initialization points
candidate_inits = [
    [1.0 / len(model_names)] * len(model_names),
    [1.0 if name == best_single_name else 0.0 for name in model_names],
    [single_accuracies[name] ** 4 for name in model_names],
]

for _ in range(25):
    candidate_inits.append(np.random.dirichlet(np.ones(len(model_names))).tolist())

best_opt_score = float("inf")
best_weights = None

for init_w in candidate_inits:
    res = minimize(
        accuracy_objective,
        init_w,
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-5},
    )
    if res.fun < best_opt_score:
        best_opt_score = res.fun
        best_weights = res.x

_, opt_weights = compute_blend_oof(best_weights)

oof_final = np.zeros((len(train_df), num_classes), dtype=np.float64)
test_final = np.zeros((len(test_df), num_classes), dtype=np.float64)
for idx, name in enumerate(model_names):
    oof_final += opt_weights[idx] * oof_model_preds[name]
    test_final += opt_weights[idx] * test_model_preds[name]

final_score = accuracy_score(y_encoded, np.argmax(oof_final, axis=1))

# Guarantee ensemble accuracy >= best single model
if final_score < best_single_acc:
    print(f"Blending did not outperform single best ({best_single_name}). Falling back to top model.")
    oof_final = oof_model_preds[best_single_name]
    test_final = test_model_preds[best_single_name]
    final_score = best_single_acc

print("Optimized Ensemble Weights:")
for name, w in zip(model_names, opt_weights):
    print(f"  {name}: {w:.4f}")

test_preds_encoded = np.argmax(test_final, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_preds_encoded)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "class": test_pred_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == len(
    X_test_eng
), f"Submission row count mismatch: {len(submission_df)} vs {len(X_test_eng)}"
assert (
    submission_df.isna().sum().sum() == 0
), "Submission contains unexpected null values!"

print(f"Final Validation Score: {final_score}")