import numpy as np
import pandas as pd
import os
import json
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Separate target and id
target = train["target"].values
train_ids = train["id"].values
test_ids = test["id"].values
train.drop(["id", "target"], axis=1, inplace=True)
test.drop(["id"], axis=1, inplace=True)

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (stratified)
# ============================================================
train_df, val_df = train_test_split(
    train, test_size=0.2, random_state=42, stratify=target
)
target_train = target[train_df.index]
target_val = target[val_df.index]

print(f"Train split: {train_df.shape}, Val split: {val_df.shape}")
print(
    f"Target balance - train: {target_train.mean():.4f}, val: {target_val.mean():.4f}"
)

# ============================================================
# 3. IDENTIFY FEATURE CATEGORIES
# ============================================================
bin_cols = [c for c in train.columns if c.startswith("bin_")]
nom_cols = [c for c in train.columns if c.startswith("nom_")]
ord_cols = [c for c in train.columns if c.startswith("ord_")]
other_cols = [c for c in train.columns if c not in bin_cols + nom_cols + ord_cols]

print(f"Binary: {bin_cols}")
print(f"Nominal: {nom_cols}")
print(f"Ordinal: {ord_cols}")
print(f"Other: {other_cols}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# --- 4a. Missing value indicators ---
all_feature_cols = bin_cols + nom_cols + ord_cols + other_cols
for col in all_feature_cols:
    train_df[f"{col}_missing"] = train_df[col].isna().astype(int)
    val_df[f"{col}_missing"] = val_df[col].isna().astype(int)
    test[f"{col}_missing"] = test[col].isna().astype(int)

# Total missing count per row
train_df["total_missing"] = train_df[[f"{c}_missing" for c in all_feature_cols]].sum(
    axis=1
)
val_df["total_missing"] = val_df[[f"{c}_missing" for c in all_feature_cols]].sum(axis=1)
test["total_missing"] = test[[f"{c}_missing" for c in all_feature_cols]].sum(axis=1)

# --- 4b. Binary features: fill NaN with -1, encode as float ---
for col in bin_cols:
    train_df[col] = train_df[col].fillna(-1)
    val_df[col] = val_df[col].fillna(-1)
    test[col] = test[col].fillna(-1)
    if train_df[col].dtype == object:
        mapping = {"T": 1, "F": 0, "Y": 1, "N": 0}
        train_df[col] = train_df[col].map(mapping).fillna(-1)
        val_df[col] = val_df[col].map(mapping).fillna(-1)
        test[col] = test[col].map(mapping).fillna(-1)
    train_df[col] = train_df[col].astype(float)
    val_df[col] = val_df[col].astype(float)
    test[col] = test[col].astype(float)

# --- 4c. Ordinal features: map to numeric ---
for col in ord_cols:
    if train_df[col].dtype == object:
        unique_vals = sorted(train_df[col].dropna().unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        train_df[col] = train_df[col].map(mapping).astype(float)
        val_df[col] = val_df[col].map(mapping).astype(float)
        test[col] = test[col].map(mapping).astype(float)
    else:
        # Already numeric (e.g., ord_0 is float64)
        train_df[col] = train_df[col].astype(float)
        val_df[col] = val_df[col].astype(float)
        test[col] = test[col].astype(float)


# --- 4d. Nominal features: K-fold Target Encoding + Frequency Encoding + Count Encoding ---
from sklearn.model_selection import KFold

def kfold_target_encode(train_col, train_target, val_col, test_col, smoothing=50, n_splits=5, random_state=42):
    """K-fold target encoding to reduce overfitting in the encoding itself."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = train_target.mean()

    train_enc = pd.Series(np.nan, index=train_col.index)

    for train_idx, val_idx in kf.split(train_col):
        # Fit encoding on the training fold
        fold_train_col = train_col.iloc[train_idx]
        fold_train_target = train_target[train_idx]

        agg = pd.DataFrame({"col": fold_train_col.values, "target": fold_train_target})
        stats = agg.groupby("col")["target"].agg(["mean", "count"])
        fold_global_mean = fold_train_target.mean()
        stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * fold_global_mean) / (
            stats["count"] + smoothing
        )

        # Apply to validation fold
        train_enc.iloc[val_idx] = train_col.iloc[val_idx].map(stats["smoothed"]).fillna(fold_global_mean)

    # Fill any remaining NaN (shouldn't happen but safety)
    train_enc = train_enc.fillna(global_mean)

    # For val and test, fit on ALL train data
    agg_full = pd.DataFrame({"col": train_col.values, "target": train_target})
    stats_full = agg_full.groupby("col")["target"].agg(["mean", "count"])
    stats_full["smoothed"] = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (
        stats_full["count"] + smoothing
    )

    val_enc = val_col.map(stats_full["smoothed"]).fillna(global_mean)
    test_enc = test_col.map(stats_full["smoothed"]).fillna(global_mean)

    return train_enc, val_enc, test_enc


def frequency_encode(train_col, val_col, test_col):
    """Frequency encoding fitted only on train."""
    freq = train_col.value_counts(normalize=True)
    train_enc = train_col.map(freq).fillna(0)
    val_enc = val_col.map(freq).fillna(0)
    test_enc = test_col.map(freq).fillna(0)
    return train_enc, val_enc, test_enc


def count_encode(train_col, val_col, test_col):
    """Count encoding (raw count of each category in train)."""
    counts = train_col.value_counts()
    train_enc = train_col.map(counts).fillna(0)
    val_enc = val_col.map(counts).fillna(0)
    test_enc = test_col.map(counts).fillna(0)
    return train_enc, val_enc, test_enc


# Apply K-fold target encoding + frequency + count to all nominal features
for col in nom_cols:
    te_train, te_val, te_test = kfold_target_encode(
        train_df[col], target_train, val_df[col], test[col], smoothing=50
    )
    train_df[f"{col}_te"] = te_train
    val_df[f"{col}_te"] = te_val
    test[f"{col}_te"] = te_test

    fe_train, fe_val, fe_test = frequency_encode(train_df[col], val_df[col], test[col])
    train_df[f"{col}_freq"] = fe_train
    val_df[f"{col}_freq"] = fe_val
    test[f"{col}_freq"] = fe_test

    ce_train, ce_val, ce_test = count_encode(train_df[col], val_df[col], test[col])
    train_df[f"{col}_count"] = ce_train
    val_df[f"{col}_count"] = ce_val
    test[f"{col}_count"] = ce_test

# Also apply K-fold target encoding to ordinal features
for col in ord_cols:
    te_train, te_val, te_test = kfold_target_encode(
        train_df[col], target_train, val_df[col], test[col], smoothing=50
    )
    train_df[f"{col}_te"] = te_train
    val_df[f"{col}_te"] = te_val
    test[f"{col}_te"] = te_test

# Add nom_5_x_nom_6 interaction (product of their target encodings)
train_df["nom_5_x_nom_6"] = train_df["nom_5_te"] * train_df["nom_6_te"]
val_df["nom_5_x_nom_6"] = val_df["nom_5_te"] * val_df["nom_6_te"]
test["nom_5_x_nom_6"] = test["nom_5_te"] * test["nom_6_te"]

# --- 4e. Pairwise interactions between low-cardinality features ---
low_card_features = []
for col in all_feature_cols:
    nunique = train_df[col].nunique()
    if nunique <= 30:
        low_card_features.append(col)

print(f"Low-cardinality features for interactions: {low_card_features}")

interaction_pairs = []
for i in range(len(low_card_features)):
    for j in range(i + 1, len(low_card_features)):
        c1, c2 = low_card_features[i], low_card_features[j]
        te1_train = train_df.get(f"{c1}_te", train_df[c1])
        te2_train = train_df.get(f"{c2}_te", train_df[c2])
        te1_val = val_df.get(f"{c1}_te", val_df[c1])
        te2_val = val_df.get(f"{c2}_te", val_df[c2])  # Fixed: was c1 instead of c2
        te1_test = test.get(f"{c1}_te", test[c1])
        te2_test = test.get(f"{c2}_te", test[c2])

        train_df[f"{c1}_x_{c2}"] = te1_train * te2_train
        val_df[f"{c1}_x_{c2}"] = te1_val * te2_val
        test[f"{c1}_x_{c2}"] = te1_test * te2_test
        interaction_pairs.append(f"{c1}_x_{c2}")

print(f"Created {len(interaction_pairs)} interaction features")

# --- 4f. Cyclical encoding for day and month ---
for col in ["day"]:
    if col in train_df.columns:
        train_df[f"{col}_sin"] = np.sin(2 * np.pi * train_df[col] / 7)
        train_df[f"{col}_cos"] = np.cos(2 * np.pi * train_df[col] / 7)
        val_df[f"{col}_sin"] = np.sin(2 * np.pi * val_df[col] / 7)
        val_df[f"{col}_cos"] = np.cos(2 * np.pi * val_df[col] / 7)
        test[f"{col}_sin"] = np.sin(2 * np.pi * test[col] / 7)
        test[f"{col}_cos"] = np.cos(2 * np.pi * test[col] / 7)

for col in ["month"]:
    if col in train_df.columns:
        train_df[f"{col}_sin"] = np.sin(2 * np.pi * train_df[col] / 12)
        train_df[f"{col}_cos"] = np.cos(2 * np.pi * train_df[col] / 12)
        val_df[f"{col}_sin"] = np.sin(2 * np.pi * val_df[col] / 12)
        val_df[f"{col}_cos"] = np.cos(2 * np.pi * val_df[col] / 12)
        test[f"{col}_sin"] = np.sin(2 * np.pi * test[col] / 12)
        test[f"{col}_cos"] = np.cos(2 * np.pi * test[col] / 12)

# ============================================================
# 5. FINALIZE FEATURE MATRIX
# ============================================================
cols_to_drop = [c for c in all_feature_cols if c in train_df.columns]
X_train = train_df.drop(columns=cols_to_drop)
X_val = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])
X_test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# Ensure all features are numeric
for col in X_train.columns:
    if X_train[col].dtype == object:
        X_train[col] = X_train[col].astype(float)
        X_val[col] = X_val[col].astype(float)
        X_test[col] = X_test[col].astype(float)

print(f"\nFinal feature matrix shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# ============================================================
# 6. SAVE TO DISK
# ============================================================
np.save("./working/X_train.npy", X_train.values.astype(np.float32))
np.save("./working/X_val.npy", X_val.values.astype(np.float32))
np.save("./working/X_test.npy", X_test.values.astype(np.float32))
np.save("./working/target_train.npy", target_train.astype(np.float32))
np.save("./working/target_val.npy", target_val.astype(np.float32))
np.save("./working/test_ids.npy", test_ids)

with open("./working/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

print(f"Saved {len(X_train.columns)} features")

# ============================================================
# 7. MODEL CONFIGURATION (Ensemble of 5 diverse models: 3 LGBM + 1 XGB + 1 CatBoost)
# ============================================================
lgbm_configs = [
    {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "max_bin": 255,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
    },
    {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "max_bin": 255,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 123,
    },
    {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 511,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.01,
        "max_bin": 255,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 777,
    },
]

xgb_config = {
    "max_depth": 8,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "learning_rate": 0.02,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 2024,
    "nthread": -1,
}

catboost_config = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "boosting_type": "Ordered",
    "eval_metric": "AUC",
    "verbose": 0,
    "random_seed": 31415,
}

print(f"\nEnsemble of 5 diverse models: 3 LightGBM + 1 XGBoost + 1 CatBoost")
for i, cfg in enumerate(lgbm_configs):
    print(
        f"  LGBM Model {i}: lr={cfg['learning_rate']}, leaves={cfg['num_leaves']}, "
        f"feat_frac={cfg['feature_fraction']}, seed={cfg['seed']}"
    )
print(f"  XGBoost: lr={xgb_config['learning_rate']}, depth={xgb_config['max_depth']}, seed={xgb_config['seed']}")
print(f"  CatBoost: lr={catboost_config['learning_rate']}, depth={catboost_config['depth']}, seed={catboost_config['random_seed']}")

# ============================================================
# 8. TRAIN ENSEMBLE WITH EARLY STOPPING
# ============================================================
print("\n--- Training Ensemble of 5 diverse models (3 LGBM + 1 XGB + 1 CatBoost) ---")
print(f"  Early stopping patience: 100 rounds")

models = []
best_iterations = []
val_aucs = []

# --- Train LightGBM models ---
train_data = lgb.Dataset(
    X_train.values.astype(np.float32),
    label=target_train,
    feature_name=list(X_train.columns),
    free_raw_data=False,
)
val_data = lgb.Dataset(
    X_val.values.astype(np.float32),
    label=target_val,
    feature_name=list(X_train.columns),
    reference=train_data,
    free_raw_data=False,
)

for i, params in enumerate(lgbm_configs):
    print(f"\n  Training LGBM model {i} (lr={params['learning_rate']}, leaves={params['num_leaves']}, seed={params['seed']})...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    models.append(("lgbm", model))
    best_iterations.append(model.best_iteration)
    val_aucs.append(model.best_score["valid"]["auc"])
    print(f"    Best iteration: {model.best_iteration}, Best val AUC: {model.best_score['valid']['auc']:.6f}")

# --- Train XGBoost model ---
print(f"\n  Training XGBoost (lr={xgb_config['learning_rate']}, depth={xgb_config['max_depth']}, seed={xgb_config['seed']})...")
xgb_model = xgb.XGBClassifier(
    max_depth=xgb_config["max_depth"],
    colsample_bytree=xgb_config["colsample_bytree"],
    subsample=xgb_config["subsample"],
    reg_alpha=xgb_config["reg_alpha"],
    reg_lambda=xgb_config["reg_lambda"],
    learning_rate=xgb_config["learning_rate"],
    tree_method=xgb_config["tree_method"],
    objective=xgb_config["objective"],
    eval_metric=xgb_config["eval_metric"],
    random_state=xgb_config["seed"],
    n_jobs=xgb_config["nthread"],
    early_stopping_rounds=100,
    n_estimators=5000,
)
xgb_model.fit(
    X_train.values.astype(np.float32),
    target_train,
    eval_set=[(X_val.values.astype(np.float32), target_val)],
    verbose=500,
)
models.append(("xgb", xgb_model))
best_iterations.append(xgb_model.best_iteration)
xgb_val_pred = xgb_model.predict_proba(X_val.values.astype(np.float32), iteration_range=(0, xgb_model.best_iteration + 1))[:, 1]
val_aucs.append(roc_auc_score(target_val, xgb_val_pred))
print(f"    Best iteration: {xgb_model.best_iteration}, Best val AUC: {val_aucs[-1]:.6f}")

# --- Train CatBoost model ---
print(f"\n  Training CatBoost (lr={catboost_config['learning_rate']}, depth={catboost_config['depth']}, seed={catboost_config['random_seed']})...")
cb_model = CatBoostClassifier(
    iterations=catboost_config["iterations"],
    learning_rate=catboost_config["learning_rate"],
    depth=catboost_config["depth"],
    l2_leaf_reg=catboost_config["l2_leaf_reg"],
    subsample=catboost_config["subsample"],
    boosting_type=catboost_config["boosting_type"],
    eval_metric=catboost_config["eval_metric"],
    verbose=catboost_config["verbose"],
    random_seed=catboost_config["random_seed"],
    early_stopping_rounds=100,
)
cb_model.fit(
    X_train.values.astype(np.float32),
    target_train,
    eval_set=(X_val.values.astype(np.float32), target_val),
    verbose=500,
)
models.append(("catboost", cb_model))
best_iterations.append(cb_model.get_best_iteration())
cb_val_pred = cb_model.predict_proba(X_val.values.astype(np.float32))[:, 1]
val_aucs.append(roc_auc_score(target_val, cb_val_pred))
print(f"    Best iteration: {cb_model.get_best_iteration()}, Best val AUC: {val_aucs[-1]:.6f}")

print(f"\n  Individual val AUCs: {[f'{a:.6f}' for a in val_aucs]}")
print(f"  Individual best iterations: {best_iterations}")

# ============================================================
# 9. VALIDATION PREDICTIONS (Ensemble Average)
# ============================================================
val_preds = np.zeros(len(X_val))
for i, (model_type, model) in enumerate(models):
    if model_type == "lgbm":
        val_preds += model.predict(X_val.values.astype(np.float32), num_iteration=best_iterations[i])
    elif model_type == "xgb":
        val_preds += model.predict_proba(X_val.values.astype(np.float32), iteration_range=(0, best_iterations[i] + 1))[:, 1]
    elif model_type == "catboost":
        val_preds += model.predict_proba(X_val.values.astype(np.float32))[:, 1]
val_preds /= len(models)
val_auc = roc_auc_score(target_val, val_preds)
print(f"\nEnsemble Validation ROC-AUC: {val_auc:.6f}")

# ============================================================
# 10. TEST PREDICTIONS (Retrain on full train+val, then average)
# ============================================================
print("\n--- Retraining on full data (train+val) for test predictions ---")

# Combine train and val for full-data retraining
X_full = np.vstack([X_train.values.astype(np.float32), X_val.values.astype(np.float32)])
target_full = np.concatenate([target_train, target_val])

full_data = lgb.Dataset(
    X_full,
    label=target_full,
    feature_name=list(X_train.columns),
    free_raw_data=False,
)

test_preds = np.zeros(len(X_test))
for i, (model_type, _) in enumerate(models):
    if model_type == "lgbm":
        print(f"  Retraining LGBM model {i} on full data (using {best_iterations[i]} rounds)...")
        full_model = lgb.train(
            lgbm_configs[i],
            full_data,
            num_boost_round=best_iterations[i],
            valid_sets=[full_data],
            valid_names=["full"],
            callbacks=[lgb.log_evaluation(period=500)],
        )
        test_preds += full_model.predict(X_test.values.astype(np.float32))
    elif model_type == "xgb":
        print(f"  Retraining XGBoost on full data (using {best_iterations[i]} rounds)...")
        full_xgb = xgb.XGBClassifier(
            max_depth=xgb_config["max_depth"],
            colsample_bytree=xgb_config["colsample_bytree"],
            subsample=xgb_config["subsample"],
            reg_alpha=xgb_config["reg_alpha"],
            reg_lambda=xgb_config["reg_lambda"],
            learning_rate=xgb_config["learning_rate"],
            tree_method=xgb_config["tree_method"],
            objective=xgb_config["objective"],
            eval_metric=xgb_config["eval_metric"],
            random_state=xgb_config["seed"],
            n_jobs=xgb_config["nthread"],
            n_estimators=best_iterations[i],
        )
        full_xgb.fit(X_full, target_full, verbose=False)
        test_preds += full_xgb.predict_proba(X_test.values.astype(np.float32))[:, 1]
    elif model_type == "catboost":
        print(f"  Retraining CatBoost on full data (using {best_iterations[i]} rounds)...")
        full_cb = CatBoostClassifier(
            iterations=best_iterations[i],
            learning_rate=catboost_config["learning_rate"],
            depth=catboost_config["depth"],
            l2_leaf_reg=catboost_config["l2_leaf_reg"],
            subsample=catboost_config["subsample"],
            boosting_type=catboost_config["boosting_type"],
            eval_metric=catboost_config["eval_metric"],
            verbose=catboost_config["verbose"],
            random_seed=catboost_config["random_seed"],
        )
        full_cb.fit(X_full, target_full, verbose=catboost_config["verbose"])
        test_preds += full_cb.predict_proba(X_test.values.astype(np.float32))[:, 1]

test_preds /= len(models)
print(
    f"Test predictions: shape={test_preds.shape}, mean={test_preds.mean():.4f}, std={test_preds.std():.4f}"
)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_ids.astype(int), "target": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved to ./submission/submission.csv")
print(f"  Shape: {submission.shape}")
print(f"  Columns: {list(submission.columns)}")

# ============================================================
# 12. FEATURE IMPORTANCE (top 15, averaged across ensemble)
# ============================================================
total_importance = np.zeros(len(X_train.columns))
count = 0
for model_type, model in models:
    if model_type == "lgbm":
        total_importance += model.feature_importance(importance_type="gain")
        count += 1
    elif model_type == "xgb":
        total_importance += model.feature_importances_
        count += 1
    elif model_type == "catboost":
        total_importance += model.get_feature_importance()
        count += 1
total_importance /= count
feat_imp = pd.Series(total_importance, index=list(X_train.columns)).sort_values(
    ascending=False
)
print("\nTop 15 features by gain (ensemble average):")
for i, (feat, imp) in enumerate(feat_imp.head(15).items()):
    print(f"  {i+1}. {feat}: {imp:.1f}")

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_auc
print(f"Final Ensemble Validation Score: {score:.6f}")
