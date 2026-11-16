import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load raw data
print("Loading datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and feature columns
target_col = "Class" if "Class" in train_df.columns else "target"
y_raw = train_df[target_col].values
X_train_df = train_df.drop(columns=[target_col], errors="ignore")

if "id" in X_train_df.columns:
    X_train_df = X_train_df.drop(columns=["id"])

test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
X_test_df = test_df.drop(columns=["id", target_col], errors="ignore")

feature_names = list(X_train_df.columns)
X_test_df = X_test_df[feature_names]

X_raw = X_train_df.values
X_test_vals = X_test_df.values

# Convert targets to 0/1 binary format regardless of original encoding (e.g. {-1, 1}, {1, 2}, or {0, 1})
unique_targets = np.array(sorted(np.unique(y_raw)))
target_map = {val: i for i, val in enumerate(unique_targets)}
inverse_target_map = {i: val for i, val in enumerate(unique_targets)}
y = np.array([target_map[val] for val in y_raw], dtype=np.int64)

# 2. Stratified K-Fold Cross-Validation Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds_prob = np.zeros(len(X_test_vals))

print(f"Starting {n_splits}-fold cross-validation ensemble training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
    X_tr_raw, y_tr_f = X_raw[train_idx], y[train_idx]
    X_va_raw, y_va_f = X_raw[val_idx], y[val_idx]
    X_te_raw = X_test_vals.copy()

    # Feature Selection strictly inside fold
    top_k = 20
    et_selector = ExtraTreesClassifier(n_estimators=300, random_state=42 + fold, n_jobs=-1)
    et_selector.fit(X_tr_raw, y_tr_f)
    selected_indices = np.argsort(et_selector.feature_importances_)[::-1][:top_k]

    X_tr_f = X_tr_raw[:, selected_indices]
    X_va_f = X_va_raw[:, selected_indices]
    X_test_sub = X_te_raw[:, selected_indices]

    scaler = StandardScaler()
    X_tr_f_scaled = scaler.fit_transform(X_tr_f)
    X_va_f_scaled = scaler.transform(X_va_f)
    X_te_f_scaled = scaler.transform(X_test_sub)

    # Spatial KNN features
    knn = NearestNeighbors(n_neighbors=10, metric="minkowski", p=2, n_jobs=-1)
    knn.fit(X_tr_f_scaled)

    def add_knn_feats(X_in):
        dists, _ = knn.kneighbors(X_in)
        return np.hstack([X_in, dists[:, :5], dists.mean(axis=1, keepdims=True)])

    X_tr_f_ext = add_knn_feats(X_tr_f_scaled)
    X_va_f_ext = add_knn_feats(X_va_f_scaled)
    X_te_f_ext = add_knn_feats(X_te_f_scaled)

    # Diverse model family for Madelon dataset
    m_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=15, random_state=42 + fold, n_jobs=-1
    )
    m_rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, random_state=42 + fold, n_jobs=-1
    )
    m_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1,
    )
    m_xgb = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        eval_metric="logloss",
    )
    m_svc = SVC(
        probability=True, C=5.0, kernel="rbf", gamma="scale", random_state=42 + fold
    )
    m_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        early_stopping=True,
        random_state=42 + fold,
    )

    models = [m_et, m_rf, m_lgb, m_xgb, m_svc, m_mlp]

    fold_val_probs = np.zeros(len(val_idx))
    fold_test_probs = np.zeros(len(X_test_vals))

    for m in models:
        m.fit(X_tr_f_ext, y_tr_f)
        fold_val_probs += m.predict_proba(X_va_f_ext)[:, 1] / len(models)
        fold_test_probs += m.predict_proba(X_te_f_ext)[:, 1] / len(models)

    oof_preds[val_idx] = fold_val_probs
    test_preds_prob += fold_test_probs / n_splits

    fold_acc = accuracy_score(y_va_f, (fold_val_probs >= 0.5).astype(int))
    print(f"Fold {fold + 1}/{n_splits} Validation Accuracy: {fold_acc:.4f}")

# Overall Out-Of-Fold Evaluation
oof_class_preds = (oof_preds >= 0.5).astype(int)
final_val_acc = accuracy_score(y, oof_class_preds)

# 4. Generate Final Test Predictions & Submission File
test_preds_binary = (test_preds_prob >= 0.5).astype(int)
test_preds_final = np.array([inverse_target_map[val] for val in test_preds_binary])

submission_df = pd.DataFrame({"id": test_ids, "Class": test_preds_final})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved successfully to {submission_path}")

print(f"Final Validation Score: {final_val_acc}")