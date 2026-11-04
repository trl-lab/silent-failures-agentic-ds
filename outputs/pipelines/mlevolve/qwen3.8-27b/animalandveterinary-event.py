import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['serious_ae'].value_counts(normalize=True)}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES & SPLIT
# ============================================================
target = "serious_ae"
id_col = "id" if "id" in train.columns else None

y = train[target].values
# Convert string "true"/"false" to numeric 1/0
if y.dtype == object:
    y = np.array([1 if str(v).strip().lower() == "true" else 0 for v in y], dtype=np.int64)
elif y.dtype == bool:
    y = y.astype(np.int64)
train_features = train.drop(columns=[target] + ([id_col] if id_col else []))
test_features = test.drop(columns=[id_col] if id_col else [])

cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train_features.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# Stratified split
train_feat_split, val_feat_split, y_train, y_val = train_test_split(
    train_features, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train split: {train_feat_split.shape}, Val split: {val_feat_split.shape}")
print(f"Target balance - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def kfold_target_encode(
    train_df, val_df, test_df, col, target_vals, n_splits=5, smoothing=10
):
    global_mean = target_vals.mean()
    train_enc = pd.Series(np.nan, index=train_df.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in skf.split(train_df, target_vals):
        tr_target = target_vals[tr_idx]
        tr_col = train_df.iloc[tr_idx][col]
        stats = (
            pd.DataFrame({"col": tr_col, "target": tr_target})
            .groupby("col")["target"]
            .agg(["mean", "count"])
        )
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        va_col = train_df.iloc[va_idx][col]
        train_enc.iloc[va_idx] = va_col.map(smoothed).fillna(global_mean)

    full_stats = (
        pd.DataFrame({"col": train_df[col], "target": target_vals})
        .groupby("col")["target"]
        .agg(["mean", "count"])
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    val_enc = val_df[col].map(full_smoothed).fillna(global_mean)
    test_enc = test_df[col].map(full_smoothed).fillna(global_mean)
    return train_enc, val_enc, test_enc


def frequency_encode(train_df, val_df, test_df, col):
    freq = train_df[col].value_counts(normalize=True)
    return (
        train_df[col].map(freq).fillna(0),
        val_df[col].map(freq).fillna(0),
        test_df[col].map(freq).fillna(0),
    )


new_train = train_feat_split.copy()
new_val = val_feat_split.copy()
new_test = test_features.copy()

# Target encoding
for col in cat_cols:
    tr_enc, va_enc, te_enc = kfold_target_encode(
        train_feat_split, val_feat_split, test_features, col, y_train
    )
    new_train[f"{col}_target_enc"] = tr_enc
    new_val[f"{col}_target_enc"] = va_enc
    new_test[f"{col}_target_enc"] = te_enc

# Frequency encoding
for col in cat_cols:
    tr_f, va_f, te_f = frequency_encode(
        train_feat_split, val_feat_split, test_features, col
    )
    new_train[f"{col}_freq"] = tr_f
    new_val[f"{col}_freq"] = va_f
    new_test[f"{col}_freq"] = te_f

# Missing value indicators
for col in train_feat_split.columns:
    if train_feat_split[col].isna().any() or val_feat_split[col].isna().any():
        new_train[f"{col}_missing"] = train_feat_split[col].isna().astype(int)
        new_val[f"{col}_missing"] = val_feat_split[col].isna().astype(int)
        new_test[f"{col}_missing"] = test_features[col].isna().astype(int)

# Interaction features
if len(cat_cols) >= 2:
    for i in range(min(len(cat_cols), 6)):
        for j in range(i + 1, min(len(cat_cols), 6)):
            col_a, col_b = cat_cols[i], cat_cols[j]
            interaction = f"{col_a}_x_{col_b}"
            new_train[interaction] = (
                train_feat_split[col_a].astype(str)
                + "_"
                + train_feat_split[col_b].astype(str)
            )
            new_val[interaction] = (
                val_feat_split[col_a].astype(str)
                + "_"
                + val_feat_split[col_b].astype(str)
            )
            new_test[interaction] = (
                test_features[col_a].astype(str)
                + "_"
                + test_features[col_b].astype(str)
            )

    # Target encode interactions (simple smoothed mean on full train)
    global_mean_inter = y_train.mean()
    smoothing_inter = 10
    for i in range(min(len(cat_cols), 6)):
        for j in range(i + 1, min(len(cat_cols), 6)):
            col_a, col_b = cat_cols[i], cat_cols[j]
            interaction = f"{col_a}_x_{col_b}"
            stats = (
                pd.DataFrame({"col": new_train[interaction], "target": y_train})
                .groupby("col")["target"]
                .agg(["mean", "count"])
            )
            smoothed = (stats["count"] * stats["mean"] + smoothing_inter * global_mean_inter) / (
                stats["count"] + smoothing_inter
            )
            new_train[f"{interaction}_target_enc"] = new_train[interaction].map(smoothed).fillna(global_mean_inter)
            new_val[f"{interaction}_target_enc"] = new_val[interaction].map(smoothed).fillna(global_mean_inter)
            new_test[f"{interaction}_target_enc"] = new_test[interaction].map(smoothed).fillna(global_mean_inter)

# Log transform for skewed numerical features
for col in num_cols:
    if col in new_train.columns:
        non_null = new_train[col].dropna()
        if len(non_null) > 0 and non_null.std() > 0:
            skew = non_null.skew()
            if abs(skew) > 1:
                new_train[f"{col}_log"] = np.log1p(new_train[col].clip(lower=0))
                new_val[f"{col}_log"] = np.log1p(new_val[col].clip(lower=0))
                new_test[f"{col}_log"] = np.log1p(new_test[col].clip(lower=0))

# Ratio features
if len(num_cols) > 1:
    for col in num_cols[:10]:
        if col in new_train.columns:
            for other in num_cols[:5]:
                if other != col and other in new_train.columns:
                    ratio_name = f"{col}_div_{other}"
                    new_train[ratio_name] = new_train[col] / new_train[other].replace(
                        0, np.nan
                    )
                    new_val[ratio_name] = new_val[col] / new_val[other].replace(
                        0, np.nan
                    )
                    new_test[ratio_name] = new_test[col] / new_test[other].replace(
                        0, np.nan
                    )

# Convert categoricals to category dtype
for col in cat_cols:
    if col in new_train.columns:
        new_train[col] = new_train[col].astype("category")
        new_val[col] = new_val[col].astype("category")
        new_test[col] = new_test[col].astype("category")

# Fill NaNs
num_fill_cols = [
    c
    for c in new_train.columns
    if new_train[c].dtype in ["float64", "float32", "int64", "int32"]
    and new_train[c].isna().any()
]
for col in num_fill_cols:
    median_val = new_train[col].median()
    new_train[col] = new_train[col].fillna(median_val)
    new_val[col] = new_val[col].fillna(median_val)
    new_test[col] = new_test[col].fillna(median_val)

cat_fill_cols = [
    c
    for c in new_train.columns
    if new_train[c].dtype == "category" and new_train[c].isna().any()
]
for col in cat_fill_cols:
    new_train[col] = new_train[col].cat.add_categories(["missing"]).fillna("missing")
    new_val[col] = new_val[col].cat.add_categories(["missing"]).fillna("missing")
    new_test[col] = new_test[col].cat.add_categories(["missing"]).fillna("missing")

for col in new_train.columns:
    if new_train[col].dtype == "object" and new_train[col].isna().any():
        new_train[col] = new_train[col].fillna("missing")
        new_val[col] = new_val[col].fillna("missing")
        new_test[col] = new_test[col].fillna("missing")

print(f"Final feature count: {new_train.shape[1]}")

# ============================================================
# 4. TRAINING & EVALUATION
# ============================================================
X_train = new_train
X_val = new_val
X_test = new_test

cat_feature_names = [
    col
    for col in X_train.columns
    if X_train[col].dtype == "category" or X_train[col].dtype == "object"
]
cat_indices = [X_train.columns.tolist().index(c) for c in cat_feature_names]
print(f"Categorical features for CatBoost: {len(cat_feature_names)}")

# Model 1: Conservative
model_1 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=10.0,
    min_data_in_leaf=20,
    subsample=0.7,
    colsample_bylevel=0.7,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=100,
    auto_class_weights="Balanced",
    one_hot_max_size=10,
    border_count=128,
)

# Model 2: Aggressive
model_2 = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=5.0,
    min_data_in_leaf=15,
    subsample=0.8,
    colsample_bylevel=0.6,
    random_seed=123,
    verbose=0,
    early_stopping_rounds=150,
    auto_class_weights="Balanced",
    one_hot_max_size=15,
    border_count=255,
)

print("\n--- Training Model 1 (conservative, depth=4) ---")
model_1.fit(
    X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices, verbose=0
)
best_iter_1 = model_1.get_best_iteration()
val_pred_1 = model_1.predict_proba(X_val)[:, 1]
val_auc_1 = roc_auc_score(y_val, val_pred_1)
print(f"Model 1: best_iter={best_iter_1}, Val AUC={val_auc_1:.6f}")

print("\n--- Training Model 2 (aggressive, depth=6) ---")
model_2.fit(
    X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices, verbose=0
)
best_iter_2 = model_2.get_best_iteration()
val_pred_2 = model_2.predict_proba(X_val)[:, 1]
val_auc_2 = roc_auc_score(y_val, val_pred_2)
print(f"Model 2: best_iter={best_iter_2}, Val AUC={val_auc_2:.6f}")

# ============================================================
# 5. BLENDING
# ============================================================
w1 = val_auc_1 / (val_auc_1 + val_auc_2)
w2 = val_auc_2 / (val_auc_1 + val_auc_2)
print(f"\nBlend weights: Model1={w1:.4f}, Model2={w2:.4f}")

val_pred_blend = w1 * val_pred_1 + w2 * val_pred_2
val_auc_blend = roc_auc_score(y_val, val_pred_blend)
print(f"Blended Val AUC: {val_auc_blend:.6f}")

if val_auc_blend < max(val_auc_1, val_auc_2):
    print("Blend underperforms best single model — using best single model.")
    if val_auc_1 >= val_auc_2:
        test_pred_final = model_1.predict_proba(X_test)[:, 1]
        score = val_auc_1
    else:
        test_pred_final = model_2.predict_proba(X_test)[:, 1]
        score = val_auc_2
else:
    test_pred_1 = model_1.predict_proba(X_test)[:, 1]
    test_pred_2 = model_2.predict_proba(X_test)[:, 1]
    test_pred_final = w1 * test_pred_1 + w2 * test_pred_2
    score = val_auc_blend

# ============================================================
# 6. SAVE SUBMISSION
# ============================================================
test_ids = np.arange(len(test_pred_final))
submission = pd.DataFrame({"id": test_ids, "serious_ae": test_pred_final})
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred_final.mean():.4f}, std={test_pred_final.std():.4f}"
)
print(f"Prediction range: [{test_pred_final.min():.6f}, {test_pred_final.max():.6f}]")
print(f"\n{'='*50}")
print(f"Validation Setup: Stratified 85/15 split, random_state=42")
print(f"Metric: ROC-AUC (official competition metric)")
print(f"Model 1 AUC: {val_auc_1:.6f}")
print(f"Model 2 AUC: {val_auc_2:.6f}")
print(f"Final Validation Score: {score}")