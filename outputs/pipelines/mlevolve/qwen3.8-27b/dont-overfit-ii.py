import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['target'].value_counts()}")

# Separate target and id
y = train["target"].values
X_all = train.drop(columns=["target", "id"])
X_test = test.drop(columns=["id"])

feature_cols = X_all.columns.tolist()
print(f"Number of features: {len(feature_cols)}")

# ============================================================
# 2. TRAIN/VALIDATION SPLIT (before any feature engineering)
# ============================================================
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train split: {X_train_raw.shape[0]} samples")
print(f"Val split: {X_val_raw.shape[0]} samples")
print(f"Train target balance: {y_train.mean():.3f}")
print(f"Val target balance: {y_val.mean():.3f}")

# ============================================================
# 3. FEATURE ENGINEERING: L1-based Selection + PCA
# ============================================================

# 3a. Remove near-constant features (std < 0.01)
train_std = X_train_raw.std()
constant_mask = train_std < 0.01
print(f"Near-constant features removed: {constant_mask.sum()}")
active_cols = [c for c in feature_cols if not constant_mask[c]]

X_train_active = X_train_raw[active_cols].values
X_val_active = X_val_raw[active_cols].values
X_test_active = X_test[active_cols].values

# 3b. L1-regularized Logistic Regression for feature importance (fit on train only)
l1_model = LogisticRegression(
    penalty="l1", solver="liblinear", C=0.1, max_iter=10000, random_state=42
)
l1_model.fit(X_train_active, y_train)
l1_coefs = np.abs(l1_model.coef_[0])

# Select top 30 features by L1 coefficient magnitude
top_30_idx = np.argsort(l1_coefs)[::-1][:30]
selected_cols = [active_cols[i] for i in top_30_idx]
print(f"Features after L1 selection: {len(selected_cols)} (from {len(active_cols)})")
print(f"L1 coefs - max: {l1_coefs.max():.4f}, mean of selected: {l1_coefs[top_30_idx].mean():.4f}")

# 3c. PCA components (top 10) - fit on train only to avoid leakage
pca = PCA(n_components=10, random_state=42)
pca.fit(X_train_active)
pca_train = pca.transform(X_train_active)
pca_val = pca.transform(X_val_active)
pca_test = pca.transform(X_test_active)

# Combine L1-selected features with PCA components
X_train_l1 = X_train_active[:, top_30_idx]
X_val_l1 = X_val_active[:, top_30_idx]
X_test_l1 = X_test_active[:, top_30_idx]

X_train_final = np.hstack([X_train_l1, pca_train])
X_val_final = np.hstack([X_val_l1, pca_val])
X_test_final = np.hstack([X_test_l1, pca_test])

print(f"\nFinal feature dimensions: {X_train_final.shape[1]} (30 L1 + 10 PCA)")

# Test IDs for submission
test_ids = test["id"].values if "id" in test.columns else np.arange(len(test))

# ============================================================
# 4. LIGHTGBM ENSEMBLE MODEL DESIGN
# ============================================================
input_dim = X_train_final.shape[1]
print(f"Input dimension: {input_dim}")

# Define 5 model configurations with genuinely different architectures
model_configs = [
    # Model A: very regularized
    {"seed": 42, "colsample_bytree": 0.5, "num_leaves": 3, "max_depth": 2, "min_child_samples": 20},
    # Model B: moderate
    {"seed": 123, "colsample_bytree": 0.6, "num_leaves": 5, "max_depth": 3, "min_child_samples": 15},
    # Model C: least regularized
    {"seed": 2024, "colsample_bytree": 0.7, "num_leaves": 7, "max_depth": 4, "min_child_samples": 10},
    # Model D: strongest regularization
    {"seed": 7, "colsample_bytree": 0.8, "num_leaves": 3, "max_depth": 2, "min_child_samples": 25},
    # Model E: moderate
    {"seed": 999, "colsample_bytree": 0.9, "num_leaves": 5, "max_depth": 3, "min_child_samples": 20},
]

# Common hyperparameters with stronger regularization
common_params = {
    "reg_alpha": 5.0,
    "reg_lambda": 10.0,
    "feature_fraction": 0.6,
    "learning_rate": 0.03,
    "n_estimators": 1000,
    "verbose": -1,
}


# ============================================================
# 5. TRAINING WITH 5-FOLD CV
# ============================================================
def train_lgbm_cv(X, y, config, n_folds=5, random_state=42):
    """Train LightGBM with 5-fold CV, return CV AUC, best iterations, and OOF predictions."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_aucs = []
    best_iterations = []
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        params = {
            "objective": "binary",
            "metric": "auc",
            "seed": config["seed"],
            "colsample_bytree": config["colsample_bytree"],
            "num_leaves": config["num_leaves"],
            "max_depth": config["max_depth"],
            "min_child_samples": config["min_child_samples"],
            "reg_alpha": common_params["reg_alpha"],
            "reg_lambda": common_params["reg_lambda"],
            "feature_fraction": common_params["feature_fraction"],
            "learning_rate": common_params["learning_rate"],
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=common_params["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        val_pred = model.predict(X_vl, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        val_auc = roc_auc_score(y_vl, val_pred)
        cv_aucs.append(val_auc)
        best_iterations.append(model.best_iteration)

    return np.mean(cv_aucs), best_iterations, oof_preds


# ============================================================
# 6. MULTI-SEED LIGHTGBM ENSEMBLE TRAINING
# ============================================================
print("\n--- LightGBM Ensemble Training (5-fold CV) ---")
cv_results = []
best_iters = []
oof_preds_list = []

for i, config in enumerate(model_configs):
    cv_auc, iterations, oof_preds = train_lgbm_cv(
        X_train_final, y_train, config, n_folds=5, random_state=config["seed"]
    )
    cv_results.append(cv_auc)
    best_iters.append(iterations)
    oof_preds_list.append(oof_preds)
    print(
        f"  Model {i+1} (seed={config['seed']}, colsample={config['colsample_bytree']}, "
        f"leaves={config['num_leaves']}, depth={config['max_depth']}, "
        f"min_child={config['min_child_samples']}): "
        f"CV AUC = {cv_auc:.4f}, best_iters = {iterations}"
    )

print(f"\nIndividual model CV AUCs: {[f'{a:.4f}' for a in cv_results]}")
print(f"Mean CV AUC: {np.mean(cv_results):.4f}")


# ============================================================
# 7. TRAIN FINAL MODELS ON ALL DATA AND PREDICT
# ============================================================
# Train on all 200 samples for final test predictions
# Use the mean best iteration from CV as the number of boosting rounds
mean_best_iter = int(np.mean([np.mean(iters) for iters in best_iters]))
print(f"\nMean best iteration across models: {mean_best_iter}")

# Compute ensemble validation AUC from pooled OOF predictions
ensemble_oof = np.mean(oof_preds_list, axis=0)
val_auc = roc_auc_score(y_train, ensemble_oof)
print(f"Ensemble Validation AUC (pooled OOF): {val_auc:.4f}")

# Train final models on all data and predict test
test_preds_list = []
for i, config in enumerate(model_configs):
    params = {
        "objective": "binary",
        "metric": "auc",
        "seed": config["seed"],
        "colsample_bytree": config["colsample_bytree"],
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "min_child_samples": config["min_child_samples"],
        "reg_alpha": common_params["reg_alpha"],
        "reg_lambda": common_params["reg_lambda"],
        "feature_fraction": common_params["feature_fraction"],
        "learning_rate": common_params["learning_rate"],
        "verbose": -1,
    }

    dtrain_full = lgb.Dataset(X_train_final, label=y_train)
    model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=mean_best_iter,
    )

    test_pred = model.predict(X_test_final, num_iteration=model.best_iteration)
    test_preds_list.append(test_pred)
    print(f"  Model {i+1} trained on all data, test pred mean: {test_pred.mean():.4f}")

# Average all model predictions
test_preds = np.mean(test_preds_list, axis=0)
print(f"\nTest predictions shape: {test_preds.shape}")
print(f"Test pred range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "target": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")
print(submission.head())

# ============================================================
# FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")