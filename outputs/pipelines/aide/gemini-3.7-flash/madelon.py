import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].values
y = train_df[target_col].values
X_test = test_df[feature_cols].values

classes = np.unique(y)
n_classes = len(classes)

# 2. Stratified 5-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float64)
test_prob_list = []

# Define diverse set of k-NN configurations (multi-scale neighborhood sizes and metrics)
knn_configs = [
    {"n_neighbors": k, "weights": "distance", "metric": m}
    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for m in ["euclidean", "manhattan"]
]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Robust multi-seed ExtraTrees selector to isolate the exact 20 informative features
    importances = np.zeros(X.shape[1], dtype=np.float64)
    for seed_offset in [0, 100, 200]:
        selector_tree = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=42 + fold * 10 + seed_offset,
            n_jobs=-1,
        )
        selector_tree.fit(X_train, y_train)
        importances += selector_tree.feature_importances_

    top_feature_indices = np.argsort(importances)[::-1][:20]

    X_train_selected = X_train[:, top_feature_indices]
    X_val_selected = X_val[:, top_feature_indices]
    X_test_selected = X_test[:, top_feature_indices]

    # Standardize the selected 20 informative features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Recover the true 5-dimensional orthogonal hypercube geometry using PCA
    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    fold_val_probs = np.zeros((len(val_idx), n_classes), dtype=np.float64)
    fold_test_probs = np.zeros((len(test_df), n_classes), dtype=np.float64)

    total_models = 0

    # 1. Train multi-scale k-NN on 5D PCA orthogonal space (de-skewed hypercube manifold)
    for cfg in knn_configs:
        knn = KNeighborsClassifier(
            n_neighbors=cfg["n_neighbors"],
            weights=cfg["weights"],
            metric=cfg["metric"],
            n_jobs=-1,
        )
        knn.fit(X_train_pca, y_train)
        fold_val_probs += 1.5 * knn.predict_proba(X_val_pca)
        fold_test_probs += 1.5 * knn.predict_proba(X_test_pca)
        total_models += 1.5

    # 2. Train multi-scale k-NN on standardized 20D feature space for complementary representation
    for cfg in knn_configs:
        knn = KNeighborsClassifier(
            n_neighbors=cfg["n_neighbors"],
            weights=cfg["weights"],
            metric=cfg["metric"],
            n_jobs=-1,
        )
        knn.fit(X_train_scaled, y_train)
        fold_val_probs += 1.0 * knn.predict_proba(X_val_scaled)
        fold_test_probs += 1.0 * knn.predict_proba(X_test_scaled)
        total_models += 1.0

    fold_val_probs /= total_models
    fold_test_probs /= total_models

    oof_probs[val_idx] = fold_val_probs
    test_prob_list.append(fold_test_probs)

# 3. Validation Evaluation
oof_preds = classes[np.argmax(oof_probs, axis=1)]
cv_accuracy = accuracy_score(y, oof_preds)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# 4. Generate Final Test Predictions
mean_test_probs = np.mean(test_prob_list, axis=0)
test_preds = classes[np.argmax(mean_test_probs, axis=1)]

# 5. Save Submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
