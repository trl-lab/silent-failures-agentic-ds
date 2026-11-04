import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Separate target
y = train["target"].values
test_ids = test["id"].values if "id" in test.columns else np.arange(len(test))

# Drop id and target from train
if "id" in train.columns:
    train = train.drop("id", axis=1)
if "id" in test.columns:
    test = test.drop("id", axis=1)
train = train.drop("target", axis=1)

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any target-dependent ops)
# ============================================================
train_df_full = pd.concat([train.reset_index(drop=True), pd.DataFrame({"target": y})], axis=1)
train_df_full, val_df_full = train_test_split(train_df_full, test_size=0.2, random_state=42)
train_data = train_df_full.drop("target", axis=1)
val_data = val_df_full.drop("target", axis=1)
y_train = train_df_full["target"].values
y_val = val_df_full["target"].values

print(f"Train split: {train_data.shape}, Val split: {val_data.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# --- 3a. Normalize binary features to 0/1 ---
bin_mappings = {
    "bin_0": {0: 0, 1: 1},
    "bin_1": {0: 0, 1: 1},
    "bin_2": {0: 0, 1: 1},
    "bin_3": {"F": 0, "T": 1},
    "bin_4": {"N": 0, "Y": 1},
}

for col, mapping in bin_mappings.items():
    if col in train_data.columns:
        train_data[col] = train_data[col].map(mapping).astype(np.int8)
    if col in val_data.columns:
        val_data[col] = val_data[col].map(mapping).astype(np.int8)
    if col in test.columns:
        test[col] = test[col].map(mapping).astype(np.int8)

# --- 3b. Map ordinal features to natural numeric order ---
ord_mappings = {
    "ord_0": {1: 0, 2: 1, 3: 2},
    "ord_1": {
        "Novice": 0,
        "Contributor": 1,
        "Expert": 2,
        "Master": 3,
        "Grandmaster": 4,
    },
    "ord_2": {
        "Freezing": 0,
        "Cold": 1,
        "Warm": 2,
        "Hot": 3,
        "Lava Hot": 4,
        "Boiling Hot": 5,
    },
    "ord_3": None,
    "ord_4": None,
    "ord_5": None,
}

for col, mapping in ord_mappings.items():
    if col not in train_data.columns:
        continue
    if mapping is not None:
        train_data[col] = train_data[col].map(mapping).astype(np.int8)
        val_data[col] = val_data[col].map(mapping).astype(np.int8)
        test[col] = test[col].map(mapping).astype(np.int8)
    else:
        categories = sorted(train_data[col].unique())
        cat_to_num = {c: i for i, c in enumerate(categories)}
        train_data[col] = train_data[col].map(cat_to_num).astype(np.int16)
        val_data[col] = val_data[col].map(cat_to_num).fillna(-1).astype(np.int16)
        test[col] = test[col].map(cat_to_num).fillna(-1).astype(np.int16)

# --- 3c. Convert low-cardinality nominal features to category dtype for LightGBM native support ---
low_card_nom_cols = ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]

for col in low_card_nom_cols:
    if col not in train_data.columns:
        continue
    train_data[col] = train_data[col].astype("category")
    val_data[col] = val_data[col].astype("category")
    test[col] = test[col].astype("category")

# --- 3d. Target encoding for high-cardinality nominal features (for LightGBM) ---
high_card_nom_cols = ["nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]


def target_encode_fit(train_df, col, target, smoothing=20):
    """Fit target encoding on training data only."""
    global_mean = target.mean()
    agg = pd.DataFrame({"col": train_df[col].values, "target": target})
    stats = agg.groupby("col")["target"].agg(["mean", "count"])
    stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    mapping = stats["smoothed"].to_dict()
    return mapping, global_mean


def target_encode_transform(df, col, mapping, global_mean):
    """Apply target encoding using pre-fitted mapping."""
    return df[col].map(mapping).fillna(global_mean).astype(np.float32)


for col in high_card_nom_cols:
    if col not in train_data.columns:
        continue
    mapping, global_mean = target_encode_fit(train_data, col, y_train, smoothing=20)
    train_data[col] = target_encode_transform(train_data, col, mapping, global_mean)
    val_data[col] = target_encode_transform(val_data, col, mapping, global_mean)
    test[col] = target_encode_transform(test, col, mapping, global_mean)

# --- 3d2. Preserve original categorical columns for CatBoost ---
# We need the original string versions for CatBoost's native categorical handling
# Re-read from the original data to get clean string versions
train_orig = pd.read_parquet("./input/train.parquet")
test_orig = pd.read_parquet("./input/test.parquet")

# Align with the train/val split (same random_state=42, test_size=0.2)
train_orig_idx, val_orig_idx = train_test_split(
    train_orig.index, test_size=0.2, random_state=42
)
train_catboost = train_orig.loc[train_orig_idx].copy()
val_catboost = train_orig.loc[val_orig_idx].copy()
test_catboost = test_orig.copy()

# Drop id and target for CatBoost
if "id" in train_catboost.columns:
    train_catboost = train_catboost.drop("id", axis=1)
if "id" in val_catboost.columns:
    val_catboost = val_catboost.drop("id", axis=1)
if "id" in test_catboost.columns:
    test_catboost = test_catboost.drop("id", axis=1)
if "target" in train_catboost.columns:
    train_catboost = train_catboost.drop("target", axis=1)
if "target" in val_catboost.columns:
    val_catboost = val_catboost.drop("target", axis=1)

# Define all categorical columns for CatBoost (original string-based)
catboost_cat_cols = [
    "bin_3", "bin_4",
    "nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
    "nom_5", "nom_6", "nom_7", "nom_8", "nom_9",
    "ord_1", "ord_2", "ord_3", "ord_4", "ord_5",
]

# --- 3e. Cyclic encoding for day and month ---
for col in ["day", "month"]:
    if col not in train_data.columns:
        continue
    max_val = train_data[col].max()
    for df in [train_data, val_data, test]:
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val).astype(np.float32)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val).astype(np.float32)

# --- 3f. Interaction features ---
bin_cols = ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4"]
for i in range(len(bin_cols)):
    for j in range(i + 1, len(bin_cols)):
        col_name = f"{bin_cols[i]}_x_{bin_cols[j]}"
        for df in [train_data, val_data, test]:
            df[col_name] = (df[bin_cols[i]] * df[bin_cols[j]]).astype(np.int8)

ord_cols = ["ord_0", "ord_1", "ord_2"]
for i in range(len(ord_cols)):
    for j in range(i + 1, len(ord_cols)):
        c1, c2 = ord_cols[i], ord_cols[j]
        col_name = f"{c1}_x_{c2}"
        for df in [train_data, val_data, test]:
            df[col_name] = (df[c1] * df[c2]).astype(np.int16)

print(f"Processed train: {train_data.shape}")
print(f"Processed val: {val_data.shape}")
print(f"Processed test: {test.shape}")
print(f"Feature columns: {list(train_data.columns)}")

# ============================================================
# 4. MODEL TRAINING (LightGBM + CatBoost Ensemble)
# ============================================================

# --- 4a. LightGBM with native categorical support ---
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Check class balance
pos_ratio = y_train.mean()
neg_ratio = 1 - pos_ratio
print(f"Class balance: pos={pos_ratio:.4f}, neg={neg_ratio:.4f}")
if pos_ratio < 0.4 or pos_ratio > 0.6:
    params["scale_pos_weight"] = neg_ratio / pos_ratio
    print(f"Adjusted scale_pos_weight to {params['scale_pos_weight']:.4f}")

# Identify categorical feature indices for LightGBM native support
# Low-cardinality nominals are now 'category' dtype
cat_feature_indices = [
    train_data.columns.get_loc(col) for col in low_card_nom_cols if col in train_data.columns
]

train_set = lgb.Dataset(
    train_data, label=y_train, feature_name=train_data.columns.tolist(),
    categorical_feature=cat_feature_indices
)
val_set = lgb.Dataset(
    val_data, label=y_val, feature_name=val_data.columns.tolist(),
    categorical_feature=cat_feature_indices
)

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[val_set],
    valid_names=["val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"LightGBM Best iteration: {model.best_iteration}")
print(f"LightGBM Best validation AUC: {model.best_score['val']['auc']:.6f}")

# --- 4b. CatBoost with native categorical handling ---
print("\n--- Training CatBoost ---")
catboost_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200,
    auto_class_weights="Balanced",
    task_type="CPU",
    cat_features=catboost_cat_cols,
)

catboost_model.fit(
    train_catboost,
    y_train,
    eval_set=(val_catboost, y_val),
    use_best_model=True,
)

print(f"CatBoost Best iteration: {catboost_model.get_best_iteration()}")
print(f"CatBoost Best validation AUC: {catboost_model.get_best_score()['validation']['AUC']:.6f}")

# ============================================================
# 5. PREDICTIONS (Ensemble: 50% LightGBM + 50% CatBoost)
# ============================================================
# LightGBM predictions
val_pred_lgb = model.predict(val_data, num_iteration=model.best_iteration)
test_pred_lgb = model.predict(test, num_iteration=model.best_iteration)

# CatBoost predictions
val_pred_cb = catboost_model.predict_proba(val_catboost)[:, 1]
test_pred_cb = catboost_model.predict_proba(test_catboost)[:, 1]

# Ensemble: 50/50 weighted average
val_pred = 0.5 * val_pred_lgb + 0.5 * val_pred_cb
test_pred = 0.5 * test_pred_lgb + 0.5 * test_pred_cb

# ============================================================
# 6. EVALUATE (ROC-AUC - official metric)
# ============================================================
val_auc_lgb = roc_auc_score(y_val, val_pred_lgb)
val_auc_cb = roc_auc_score(y_val, val_pred_cb)
val_auc = roc_auc_score(y_val, val_pred)
print(f"LightGBM Validation ROC-AUC: {val_auc_lgb:.6f}")
print(f"CatBoost Validation ROC-AUC: {val_auc_cb:.6f}")
print(f"Ensemble Validation ROC-AUC: {val_auc:.6f}")

# ============================================================
# 7. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "target": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame(
    {"feature": train_data.columns, "gain": importance}
).sort_values("gain", ascending=False)
print("\nTop 10 features by gain:")
print(feat_imp.head(10).to_string(index=False))

# ============================================================
# FINAL SCORE
# ============================================================
print(f"Final Validation Score: {val_auc:.6f}")