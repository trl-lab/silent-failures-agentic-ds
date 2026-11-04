import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nTarget stats:\n{train['AwardTotal'].describe()}")
print(f"\nMissing values in train:\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values in test:\n{test.isnull().sum()[test.isnull().sum() > 0]}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "AwardTotal"
id_col = "id"

train_target = train[target_col].values
train_id = train[id_col].values if id_col in train.columns else np.arange(len(train))
test_id = test[id_col].values if id_col in test.columns else np.arange(len(test))

train_features = train.drop(
    columns=[c for c in [target_col, id_col] if c in train.columns]
)
test_features = test.drop(columns=[c for c in [id_col] if c in test.columns])

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = [c for c in train_features.columns if train_features[c].dtype == bool]

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:20]}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:20]}")
print(f"Boolean columns ({len(bool_cols)}): {bool_cols[:10]}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (FIRST, before any fitting)
# ============================================================
target_for_split = train[target_col].values

try:
    bins = pd.qcut(target_for_split, q=5, duplicates="drop")
    strat_labels = bins.cat.codes
except:
    strat_labels = None

if strat_labels is not None and len(np.unique(strat_labels)) > 1:
    train_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=0.15, random_state=42, stratify=strat_labels
    )
else:
    train_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=0.15, random_state=42
    )

print(f"\nTrain split: {len(train_idx)}, Val split: {len(val_idx)}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nPerforming feature engineering...")

all_data = pd.concat([train_features, test_features], axis=0, ignore_index=True)
n_train = len(train_features)
n_test = len(test_features)

# --- Handle missing values ---
for col in numeric_cols:
    if col in all_data.columns:
        train_median = all_data.loc[: n_train - 1, col].median()
        all_data[col] = all_data[col].fillna(train_median)

for col in categorical_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna("missing").astype(str)

for col in bool_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(False)

# --- Numeric feature engineering ---
skew_threshold = 1.0
log_features = []
for col in numeric_cols:
    if col not in all_data.columns:
        continue
    train_vals = all_data.loc[: n_train - 1, col]
    if (train_vals > 0).sum() > len(train_vals) * 0.8 and abs(
        train_vals.skew()
    ) > skew_threshold:
        all_data[f"{col}_log"] = np.log1p(all_data[col].clip(lower=0))
        log_features.append(f"{col}_log")

# Interaction features between top numeric columns (by correlation with target)
if len(numeric_cols) >= 2:
    # Use ONLY train_idx rows to determine top features (no val leakage)
    train_only_part = all_data.iloc[train_idx].copy()
    train_only_part["_target"] = target_for_split[train_idx]

    corr_with_target = (
        train_only_part[numeric_cols]
        .corrwith(train_only_part["_target"])
        .abs()
        .sort_values(ascending=False)
    )
    top_numeric = corr_with_target.head(5).index.tolist()

    for i in range(len(top_numeric)):
        for j in range(i + 1, len(top_numeric)):
            c1, c2 = top_numeric[i], top_numeric[j]
            all_data[f"{c1}_x_{c2}"] = all_data[c1] * all_data[c2]
            all_data[f"{c1}_div_{c2}"] = all_data[c1] / (all_data[c2] + 1e-8)

# Ratio features
for i in range(len(numeric_cols)):
    for j in range(i + 1, min(i + 3, len(numeric_cols))):
        c1, c2 = numeric_cols[i], numeric_cols[j]
        if c1 in all_data.columns and c2 in all_data.columns:
            all_data[f"{c1}_ratio_{c2}"] = all_data[c1] / (all_data[c2].abs() + 1e-8)

# --- Categorical feature engineering ---
print("  Encoding categorical features...")

# Label encode categoricals (fit on train, handle unseen in test)
label_encoders = {}
for col in categorical_cols:
    if col not in all_data.columns:
        continue
    train_vals = all_data.loc[: n_train - 1, col].astype(str)
    unique_train = sorted(train_vals.unique())
    mapping = {v: i for i, v in enumerate(unique_train)}
    unseen_code = len(unique_train)
    all_data[col] = all_data[col].astype(str).map(mapping).fillna(unseen_code).astype(int)
    label_encoders[col] = mapping

# Frequency encoding (fit on train only)
for col in categorical_cols:
    if col not in train_features.columns:
        continue
    train_freq = train_features[col].astype(str).value_counts(normalize=True)
    train_part_freq = train_features[col].astype(str).map(train_freq).fillna(0).values
    test_part_freq = test_features[col].astype(str).map(train_freq).fillna(0).values
    all_data[f"{col}_freq"] = np.concatenate([train_part_freq, test_part_freq])

# Target encoding (fit on train_idx only, with smoothing)
print("  Creating target-encoded features...")
smoothing = 10
global_mean = target_for_split[train_idx].mean()

for col in categorical_cols:
    if col not in train_features.columns:
        continue
    # Use ONLY train_idx rows to compute target stats (avoid val leakage)
    train_only_cat = train_features.iloc[train_idx][col].astype(str).values
    train_only_target = target_for_split[train_idx]
    val_cat = train_features.iloc[val_idx][col].astype(str).values
    test_cat = test_features[col].astype(str).values

    temp_df = pd.DataFrame({"cat": train_only_cat, "target": train_only_target})
    cat_stats = temp_df.groupby("cat")["target"].agg(["mean", "count"])

    smoothed = (cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean) / (
        cat_stats["count"] + smoothing
    )

    train_only_te = pd.Series(train_only_cat).map(smoothed).fillna(global_mean).values
    val_te = pd.Series(val_cat).map(smoothed).fillna(global_mean).values
    test_te = pd.Series(test_cat).map(smoothed).fillna(global_mean).values
    all_data[f"{col}_target_enc"] = np.concatenate([train_only_te, val_te, test_te])

# Polynomial features for top numeric
if len(numeric_cols) > 0:
    top_num = numeric_cols[:3]
    for col in top_num:
        if col in all_data.columns:
            all_data[f"{col}_sq"] = all_data[col] ** 2

# Aggregate/statistical features if there's a grouping column
group_cols = [
    c
    for c in categorical_cols
    if c in all_data.columns
    and all_data.loc[: n_train - 1, c].nunique() < 100
    and all_data.loc[: n_train - 1, c].nunique() > 2
]

for gcol in group_cols[:3]:
    for ncol in numeric_cols[:5]:
        if ncol not in all_data.columns or gcol not in all_data.columns:
            continue
        # Use ONLY train_idx rows to compute group stats (no val leakage)
        train_only_part = all_data.iloc[train_idx].copy()
        group_stats = train_only_part.groupby(gcol)[ncol].agg(["mean", "std"])
        global_mean_ncol = train_only_part[ncol].mean()

        # Apply to train rows (both train_idx and val_idx use train_idx-derived stats)
        all_data[f"{gcol}_{ncol}_group_mean"] = (
            all_data[gcol].map(group_stats["mean"]).fillna(global_mean_ncol).values
        )
        all_data[f"{gcol}_{ncol}_group_std"] = (
            all_data[gcol].map(group_stats["std"]).fillna(0).values
        )
        all_data[f"{gcol}_{ncol}_group_diff"] = (
            all_data[ncol] - all_data[f"{gcol}_{ncol}_group_mean"]
        )

# ============================================================
# 5. SEPARATE BACK INTO TRAIN/VAL/TEST
# ============================================================
train_processed = all_data.iloc[:n_train].reset_index(drop=True)
test_processed = all_data.iloc[n_train:].reset_index(drop=True)

train_train = train_processed.iloc[train_idx].reset_index(drop=True)
train_val = train_processed.iloc[val_idx].reset_index(drop=True)

y_train = target_for_split[train_idx]
y_val = target_for_split[val_idx]

feature_cols = train_train.columns.tolist()

print(f"\nFinal shapes:")
print(f"  Train: {train_train.shape}")
print(f"  Val: {train_val.shape}")
print(f"  Test: {test_processed.shape}")
print(f"Total features: {len(feature_cols)}")
print(f"Target mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
print(f"Val target mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")

# ============================================================
# 6. BASELINE (mean predictor) for reference
# ============================================================
baseline_rmse = root_mean_squared_error(y_val, np.full(len(y_val), y_train.mean()))
print(f"\nBaseline RMSE (mean predictor): {baseline_rmse:.4f}")

# ============================================================
# 7. TRAINING LOOP WITH HYPERPARAMETER REFINEMENT
# ============================================================
base_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_split_gain": 0.01,
    "feature_pre_filter": False,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

param_grid = [
    base_params,
    {**base_params, "num_leaves": 31, "min_child_samples": 30, "lambda_l2": 2.0},
    {
        **base_params,
        "num_leaves": 127,
        "min_child_samples": 15,
        "feature_fraction": 0.8,
    },
    {**base_params, "learning_rate": 0.02, "num_leaves": 47, "min_child_samples": 25},
    {
        **base_params,
        "num_leaves": 63,
        "min_child_samples": 10,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
    },
]

train_data = lgb.Dataset(
    train_train[feature_cols], label=y_train
)
val_data = lgb.Dataset(
    train_val[feature_cols],
    label=y_val,
    reference=train_data,
)

best_model = None
best_val_rmse = float("inf")
best_params = None
results = []

print("\n--- Hyperparameter refinement ---")
for i, params in enumerate(param_grid):
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=1000),
        ],
    )

    val_pred = model.predict(
        train_val[feature_cols], num_iteration=model.best_iteration
    )
    val_rmse = root_mean_squared_error(y_val, val_pred)

    train_pred = model.predict(
        train_train[feature_cols], num_iteration=model.best_iteration
    )
    train_rmse = root_mean_squared_error(y_train, train_pred)

    gap = val_rmse - train_rmse
    results.append(
        {
            "config": i,
            "best_iter": model.best_iteration,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "gap": gap,
        }
    )

    print(
        f"Config {i}: iter={model.best_iteration}, train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, gap={gap:.4f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model = model
        best_params = params

print(f"\nBest config val RMSE: {best_val_rmse:.4f} (baseline: {baseline_rmse:.4f})")
print(
    f"Improvement over baseline: {(baseline_rmse - best_val_rmse) / baseline_rmse * 100:.2f}%"
)

# ============================================================
# 8. FINAL VALIDATION & TEST INFERENCE (consistent pipeline)
# ============================================================
val_preds = best_model.predict(
    train_val[feature_cols], num_iteration=best_model.best_iteration
)
val_rmse_final = root_mean_squared_error(y_val, val_preds)

test_preds = best_model.predict(
    test_processed[feature_cols], num_iteration=best_model.best_iteration
)

# Ensure no negative predictions (AwardTotal is monetary, should be >= 0)
val_preds = np.maximum(val_preds, 0)
test_preds = np.maximum(test_preds, 0)

# Recompute validation RMSE after clipping (consistent with test processing)
val_rmse_clipped = root_mean_squared_error(y_val, val_preds)

print(f"\n--- Final Results ---")
print(f"Validation RMSE (before clip): {val_rmse_final:.4f}")
print(f"Validation RMSE (after clip): {val_rmse_clipped:.4f}")
print(
    f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}, min={test_preds.min():.4f}, max={test_preds.max():.4f}"
)

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": test_id.astype(int), "AwardTotal": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 10. FEATURE IMPORTANCE (top 15)
# ============================================================
importance = pd.Series(
    best_model.feature_importance(importance_type="gain"), index=feature_cols
)
print(f"\nTop 15 features by gain:")
print(importance.sort_values(ascending=False).head(15).to_string())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
score = val_rmse_clipped
print(f"Final Validation Score: {score}")