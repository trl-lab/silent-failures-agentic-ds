import pandas as pd
import numpy as np
import json
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Identify feature types
target_col = "capacity_mw"
id_col = "id" if "id" in train.columns else None

if id_col:
    feature_cols_raw = [c for c in train.columns if c not in [target_col, id_col]]
else:
    feature_cols_raw = [c for c in train.columns if c != target_col]

# Identify categorical and numerical features
cat_cols = (
    train[feature_cols_raw]
    .select_dtypes(include=["object", "category"])
    .columns.tolist()
)
num_cols = train[feature_cols_raw].select_dtypes(include=[np.number]).columns.tolist()

# Check for low-cardinality integer columns that might be categorical
int_cols = (
    train[feature_cols_raw].select_dtypes(include=["int64", "int32"]).columns.tolist()
)
for col in int_cols:
    if train[col].nunique() < 20:
        cat_cols.append(col)
        num_cols.remove(col)

cat_cols = list(dict.fromkeys(cat_cols))

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============ FEATURE ENGINEERING ============


def engineer_features(
    df, train_df=None, fit=False, cat_cols=None, num_cols=None, target_col=None
):
    """Apply feature engineering. If fit=True, compute statistics from train_df."""
    df = df.copy()

    # 1. Log transform target
    if target_col in df.columns:
        df["log_capacity"] = np.log1p(df[target_col])

    # 2. Handle missing values in numerical features
    for col in num_cols:
        if col in df.columns:
            if fit:
                med = train_df[col].median()
                df[col + "_is_missing"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(med)
            else:
                if col + "_is_missing" not in df.columns:
                    df[col + "_is_missing"] = df[col].isnull().astype(int)
                # Will be filled later with train medians

    # 3. Handle missing values in categorical features
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__")

    # 4. Target encoding for categoricals (fit on train only)
    if fit:
        for col in cat_cols:
            if col in train_df.columns and "log_capacity" in train_df.columns:
                temp = (
                    train_df[[col, "log_capacity"]]
                    .groupby(col)["log_capacity"]
                    .agg(["mean", "std", "count"])
                )
                df[f"{col}_target_mean"] = df[col].map(temp["mean"])
                df[f"{col}_target_std"] = df[col].map(temp["std"])
                df[f"{col}_target_count"] = df[col].map(temp["count"])
                global_mean = train_df["log_capacity"].mean()
                global_std = train_df["log_capacity"].std()
                df[f"{col}_target_mean"] = df[f"{col}_target_mean"].fillna(global_mean)
                df[f"{col}_target_std"] = df[f"{col}_target_std"].fillna(global_std)
                df[f"{col}_target_count"] = df[f"{col}_target_count"].fillna(0)
    else:
        # Apply target encoding using train mappings
        for col in cat_cols:
            if col in train_df.columns and "log_capacity" in train_df.columns:
                temp = (
                    train_df[[col, "log_capacity"]]
                    .groupby(col)["log_capacity"]
                    .agg(["mean", "std", "count"])
                )
                global_mean = train_df["log_capacity"].mean()
                global_std = train_df["log_capacity"].std()
                df[f"{col}_target_mean"] = df[col].map(temp["mean"]).fillna(global_mean)
                df[f"{col}_target_std"] = df[col].map(temp["std"]).fillna(global_std)
                df[f"{col}_target_count"] = df[col].map(temp["count"]).fillna(0)

    # 5. Interaction features between top categorical features
    if len(cat_cols) >= 2:
        top_cats = cat_cols[: min(4, len(cat_cols))]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                col_name = f"{top_cats[i]}_x_{top_cats[j]}"
                df[col_name] = (
                    df[top_cats[i]].astype(str) + "|" + df[top_cats[j]].astype(str)
                )

    # 6. Polynomial and log features for numerical columns
    for col in num_cols[:5]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2
            if (df[col] > 0).any():
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    return df


# ============ SPLIT TRAIN INTO TRAIN/VAL (BEFORE FEATURE ENGINEERING) ============
stratify_col = None
for col in cat_cols:
    if train[col].nunique() > 1 and train[col].nunique() < 50:
        stratify_col = col
        break

if stratify_col:
    train_raw, val_raw = train_test_split(
        train, test_size=0.15, random_state=42, stratify=train[stratify_col]
    )
    print(f"Stratified by: {stratify_col}")
else:
    train_raw, val_raw = train_test_split(train, test_size=0.15, random_state=42)

print(f"Raw Train split: {train_raw.shape}, Raw Val split: {val_raw.shape}")

# Apply feature engineering - FIT on training split only
train_split = engineer_features(
    train_raw,
    train_df=train_raw,
    fit=True,
    cat_cols=cat_cols,
    num_cols=num_cols,
    target_col=target_col,
)

# Apply to validation with train-split mappings
val_split = engineer_features(
    val_raw,
    train_df=train_raw,
    fit=False,
    cat_cols=cat_cols,
    num_cols=num_cols,
    target_col=target_col,
)

# Apply to test with full-train mappings (use full train for best generalization)
test_fe = engineer_features(
    test,
    train_df=train,
    fit=False,
    cat_cols=cat_cols,
    num_cols=num_cols,
    target_col=target_col,
)

# Fill numerical NaN in val and test with train-split medians
for col in num_cols:
    if col in train_raw.columns and col in val_split.columns:
        med = train_raw[col].median()
        val_split[col] = val_split[col].fillna(med)
    if col in train.columns and col in test_fe.columns:
        med = train[col].median()
        test_fe[col] = test_fe[col].fillna(med)

print(
    f"Train split: {train_split.shape}, Val split: {val_split.shape}, Test: {test_fe.shape}"
)

# ============ DETERMINE FINAL FEATURE COLUMNS ============
if id_col:
    final_feat_cols = [
        c for c in train_split.columns if c not in [target_col, "log_capacity", id_col]
    ]
else:
    final_feat_cols = [
        c for c in train_split.columns if c not in [target_col, "log_capacity"]
    ]

print(f"Final feature count: {len(final_feat_cols)}")

# Reset indices and add id to test
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)
test_fe = test_fe.reset_index(drop=True)
test_fe["id"] = range(len(test_fe))

# Save processed dataframes
train_split.to_parquet("./working/train_processed.parquet", index=False)
val_split.to_parquet("./working/val_processed.parquet", index=False)
test_fe.to_parquet("./working/test_processed.parquet", index=False)

with open("./working/feature_cols.json", "w") as f:
    json.dump(final_feat_cols, f)

# ============ PREPARE MODEL INPUTS ============
# Identify categorical vs numerical features in final feature set
cat_feature_cols = []
num_feature_cols = []
for col in final_feat_cols:
    if col in train_split.columns:
        if (
            train_split[col].dtype == object
            or train_split[col].dtype.name == "category"
        ):
            cat_feature_cols.append(col)
        else:
            num_feature_cols.append(col)

print(f"Categorical: {len(cat_feature_cols)}, Numerical: {len(num_feature_cols)}")

# Ensure all dataframes have the same columns
for df_ref in [val_split, test_fe]:
    for col in final_feat_cols:
        if col not in df_ref.columns:
            if col in cat_feature_cols:
                df_ref[col] = "__MISSING__"
            else:
                df_ref[col] = 0.0

X_train = train_split[final_feat_cols].copy()
y_train = train_split["log_capacity"].values

X_val = val_split[final_feat_cols].copy()
y_val = val_split["log_capacity"].values

# Remove any rows with NaN in target
mask_train = ~np.isnan(y_train)
mask_val = ~np.isnan(y_val)
if not mask_train.all():
    print(f"Removing {(~mask_train).sum()} rows with NaN target from train")
    X_train = X_train[mask_train].reset_index(drop=True)
    y_train = y_train[mask_train]
if not mask_val.all():
    print(f"Removing {(~mask_val).sum()} rows with NaN target from val")
    X_val = X_val[mask_val].reset_index(drop=True)
    y_val = y_val[mask_val]

X_test = test_fe[final_feat_cols].copy()

# Convert categoricals to category dtype for LightGBM
for col in cat_feature_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col] = X_val[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Fill remaining NaN in numerical features using train medians
for col in num_feature_cols:
    med = X_train[col].median()
    if np.isnan(med):
        med = 0.0
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)

# Final NaN check - replace any remaining NaN with 0
for col in num_feature_cols:
    if X_train[col].isnull().any():
        X_train[col] = X_train[col].fillna(0)
    if X_val[col].isnull().any():
        X_val[col] = X_val[col].fillna(0)
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(0)

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
print(f"NaN check - train: {X_train.isnull().sum().sum()}, val: {X_val.isnull().sum().sum()}, test: {X_test.isnull().sum().sum()}")

# ============ MODEL 1: LightGBM with Huber loss ============
params1 = {
    "objective": "huber",
    "alpha": 0.9,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

train_data1 = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_feature_cols, free_raw_data=False
)
val_data1 = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=cat_feature_cols,
    reference=train_data1,
    free_raw_data=False,
)

model1 = lgb.train(
    params1,
    train_data1,
    num_boost_round=5000,
    valid_sets=[val_data1],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

if model1.best_iteration and model1.best_iteration > 0:
    print(f"Model 1: best_iter={model1.best_iteration}")
else:
    print(f"Model 1: WARNING: best_iteration not set, using num_trees()")
    model1.best_iteration = model1.num_trees()

# ============ MODEL 2: LightGBM with MSE loss, different structure ============
params2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 3,
    "min_child_samples": 15,
    "min_child_weight": 1e-4,
    "lambda_l1": 0.05,
    "lambda_l2": 0.5,
    "max_depth": -1,
    "verbose": -1,
    "random_state": 123,
    "n_jobs": -1,
}

train_data2 = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_feature_cols, free_raw_data=False
)
val_data2 = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=cat_feature_cols,
    reference=train_data2,
    free_raw_data=False,
)

model2 = lgb.train(
    params2,
    train_data2,
    num_boost_round=5000,
    valid_sets=[val_data2],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

if model2.best_iteration and model2.best_iteration > 0:
    print(f"Model 2: best_iter={model2.best_iteration}")
else:
    print(f"Model 2: WARNING: best_iteration not set, using num_trees()")
    model2.best_iteration = model2.num_trees()

# ============ MODEL 3: LightGBM with Huber loss, more regularized ============
params3 = {
    "objective": "huber",
    "alpha": 1.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 4,
    "min_child_samples": 30,
    "min_child_weight": 5e-3,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "max_depth": 8,
    "verbose": -1,
    "random_state": 7,
    "n_jobs": -1,
}

train_data3 = lgb.Dataset(
    X_train, label=y_train, categorical_feature=cat_feature_cols, free_raw_data=False
)
val_data3 = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=cat_feature_cols,
    reference=train_data3,
    free_raw_data=False,
)

model3 = lgb.train(
    params3,
    train_data3,
    num_boost_round=5000,
    valid_sets=[val_data3],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

if model3.best_iteration and model3.best_iteration > 0:
    print(f"Model 3: best_iter={model3.best_iteration}")
else:
    print(f"Model 3: WARNING: best_iteration not set, using num_trees()")
    model3.best_iteration = model3.num_trees()

# ============ ENSEMBLE PREDICTIONS ============
val_pred1_log = model1.predict(X_val, num_iteration=model1.best_iteration)
val_pred2_log = model2.predict(X_val, num_iteration=model2.best_iteration)
val_pred3_log = model3.predict(X_val, num_iteration=model3.best_iteration)

test_pred1_log = model1.predict(X_test, num_iteration=model1.best_iteration)
test_pred2_log = model2.predict(X_test, num_iteration=model2.best_iteration)
test_pred3_log = model3.predict(X_test, num_iteration=model3.best_iteration)

# Optimize ensemble weights on validation
val_target_orig = np.expm1(y_val)
best_rmse = float("inf")
best_weights = (1 / 3, 1 / 3, 1 / 3)

for w1 in np.arange(0.1, 0.7, 0.1):
    for w2 in np.arange(0.1, 0.7 - w1 + 0.01, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        val_ens_log = w1 * val_pred1_log + w2 * val_pred2_log + w3 * val_pred3_log
        val_ens_orig = np.expm1(val_ens_log)
        val_ens_orig = np.maximum(val_ens_orig, 0)
        rmse = np.sqrt(mean_squared_error(val_target_orig, val_ens_orig))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)

print(f"Best ensemble weights: {best_weights}, val RMSE (orig): {best_rmse:.4f}")

# Apply best weights to test
w1, w2, w3 = best_weights
test_pred_log = w1 * test_pred1_log + w2 * test_pred2_log + w3 * test_pred3_log
test_pred_orig = np.expm1(test_pred_log)
test_pred_orig = np.maximum(test_pred_orig, 0)

# Validation with best weights
val_pred_log_ens = w1 * val_pred1_log + w2 * val_pred2_log + w3 * val_pred3_log
val_pred_orig_ens = np.expm1(val_pred_log_ens)
val_pred_orig_ens = np.maximum(val_pred_orig_ens, 0)

# ============ FINAL VALIDATION SCORE (RMSE on original scale) ============
val_rmse = np.sqrt(mean_squared_error(val_target_orig, val_pred_orig_ens))
print(f"\n=== VALIDATION SUMMARY ===")
print(
    f"Model 1 val RMSE (orig): {np.sqrt(mean_squared_error(val_target_orig, np.maximum(np.expm1(val_pred1_log), 0))):.4f}"
)
print(
    f"Model 2 val RMSE (orig): {np.sqrt(mean_squared_error(val_target_orig, np.maximum(np.expm1(val_pred2_log), 0))):.4f}"
)
print(
    f"Model 3 val RMSE (orig): {np.sqrt(mean_squared_error(val_target_orig, np.maximum(np.expm1(val_pred3_log), 0))):.4f}"
)
print(f"Ensemble val RMSE (orig): {val_rmse:.4f}")

# ============ SAVE SUBMISSION ============
submission = pd.DataFrame(
    {"id": range(len(test_pred_orig)), "capacity_mw": test_pred_orig}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred_orig.mean():.2f}, median={np.median(test_pred_orig):.2f}, min={test_pred_orig.min():.2f}, max={test_pred_orig.max():.2f}"
)

# Save models
model1.save_model("./working/lgb_model1.txt")
model2.save_model("./working/lgb_model2.txt")
model3.save_model("./working/lgb_model3.txt")

score = val_rmse
print(f"Final Validation Score: {score}")