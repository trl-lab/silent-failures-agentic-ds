import numpy as np
import pandas as pd
import json
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

# ============================================================
# SETUP
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "DEPSUM"

# Identify ID column
id_col = None
for col in ["id", "ID", "Id", "index"]:
    if col in train.columns:
        id_col = col
        break

# Identify categorical and numerical columns
cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

if target_col in num_cols:
    num_cols.remove(target_col)
if id_col and id_col in num_cols:
    num_cols.remove(id_col)
if id_col and id_col in cat_cols:
    cat_cols.remove(id_col)

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT (before any processing)
# ============================================================
train["log_target"] = np.log1p(train[target_col].clip(lower=0))
train["target_bin"] = pd.qcut(train["log_target"], q=10, duplicates="drop")

train_split, val_split = train_test_split(
    train, test_size=0.15, random_state=42, stratify=train["target_bin"]
)
train_split = train_split.drop(columns=["log_target", "target_bin"])
val_split = val_split.drop(columns=["log_target", "target_bin"])

print(f"Train split: {train_split.shape[0]}, Val split: {val_split.shape[0]}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df, cat_cols, num_cols, is_train=True, encoder=None, medians=None
):
    """Apply feature engineering. Fit on train, transform on val/test."""
    df = df.copy()

    # Handle missing values
    if is_train:
        medians = df[num_cols].median()
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    # Log transform skewed numerical features
    for col in num_cols:
        if col in df.columns:
            if df[col].min() >= 0:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Ratio features between deposit categories
    deposit_cols = [
        c
        for c in num_cols
        if any(
            kw in c.upper()
            for kw in [
                "DEP",
                "DAS",
                "DAP",
                "DBA",
                "DBP",
                "DASU",
                "DAPU",
                "DBAU",
                "DBPU",
            ]
        )
    ]
    if len(deposit_cols) >= 2:
        for i in range(len(deposit_cols)):
            for j in range(i + 1, len(deposit_cols)):
                col_a, col_b = deposit_cols[i], deposit_cols[j]
                if col_a in df.columns and col_b in df.columns:
                    denom = df[col_b].replace(0, np.nan)
                    df[f"ratio_{col_a}_to_{col_b}"] = (df[col_a] / denom).fillna(0)

    # Squared features (top 10)
    for col in num_cols[:10]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    # Categorical encoding
    if is_train:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if cat_cols:
            encoder.fit(df[cat_cols])
    else:
        if cat_cols:
            df[cat_cols] = encoder.transform(df[cat_cols])

    if is_train and cat_cols:
        df[cat_cols] = encoder.transform(df[cat_cols])

    # Interaction between top categorical and numerical
    if cat_cols and num_cols:
        top_num = num_cols[:5]
        top_cat = cat_cols[:3]
        for c in top_cat:
            for n in top_num:
                if c in df.columns and n in df.columns:
                    df[f"{c}_x_{n}"] = df[c] * df[n]

    return df, encoder, medians


train_split, encoder, medians = engineer_features(
    train_split, cat_cols, num_cols, is_train=True
)
val_split, _, _ = engineer_features(
    val_split, cat_cols, num_cols, is_train=False, encoder=encoder, medians=medians
)
test, _, _ = engineer_features(
    test, cat_cols, num_cols, is_train=False, encoder=encoder, medians=medians
)

# ============================================================
# 4. FINALIZE FEATURE SET
# ============================================================
feature_cols = [c for c in train_split.columns if c != target_col and c != id_col]
print(f"Final feature count: {len(feature_cols)}")

for col in feature_cols:
    if col not in val_split.columns:
        val_split[col] = 0
    if col not in test.columns:
        test[col] = 0

for col in feature_cols:
    train_split[col] = pd.to_numeric(train_split[col], errors="coerce").fillna(0)
    val_split[col] = pd.to_numeric(val_split[col], errors="coerce").fillna(0)
    test[col] = pd.to_numeric(test[col], errors="coerce").fillna(0)

# ============================================================
# 5. PREPARE TARGET (log-transform for modeling)
# ============================================================
train_split["log_DEPSUM"] = np.log1p(train_split[target_col].clip(lower=0))
val_split["log_DEPSUM"] = np.log1p(val_split[target_col].clip(lower=0))

# ============================================================
# 6. PREPARE FEATURES AND TARGETS FOR MODEL
# ============================================================
X_train = train_split[feature_cols].values.astype(np.float32)
y_train_log = train_split["log_DEPSUM"].values.astype(np.float32)
y_train_orig = train_split[target_col].values.astype(np.float64)

X_val = val_split[feature_cols].values.astype(np.float32)
y_val_log = val_split["log_DEPSUM"].values.astype(np.float32)
y_val_orig = val_split[target_col].values.astype(np.float64)

X_test = test[feature_cols].values.astype(np.float32)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"y_train_log: mean={y_train_log.mean():.4f}, std={y_train_log.std():.4f}")
print(
    f"y_train_orig: mean={y_train_orig.mean():.2f}, median={np.median(y_train_orig):.2f}"
)

# ============================================================
# 7. DEFINE METRIC (RMSE on original scale — matches official metric)
# ============================================================


def lgb_custom_metric(preds, dataset):
    """LightGBM custom metric: RMSE on original scale."""
    y_true_log = dataset.get_label()
    y_true_orig = np.expm1(np.clip(y_true_log, 0, None))
    y_pred_orig = np.expm1(np.clip(preds, 0, None))
    y_pred_orig = np.clip(y_pred_orig, 0, None)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    return "rmse_orig", rmse, False  # minimize


# ============================================================
# 8. TRAIN LIGHTGBM WITH EARLY STOPPING
# ============================================================
lgb_params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.01,
    "verbosity": -1,
    "seed": 42,
    "num_threads": -1,
}

train_data = lgb.Dataset(X_train, label=y_train_log, feature_name=feature_cols)
val_data = lgb.Dataset(
    X_val, label=y_val_log, feature_name=feature_cols, reference=train_data
)

callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=100),
]

print("\n--- Training LightGBM ---")
model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=5000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=callbacks,
    feval=lgb_custom_metric,
)

best_iter = model.best_iteration
print(f"Best iteration: {best_iter}")

# ============================================================
# 9. GENERATE PREDICTIONS (identical logic for val and test)
# ============================================================


def predict_from_model(model, X, best_iter):
    """Standardized inference: predict in log-space, inverse transform, clip at 0."""
    pred_log = model.predict(X, num_iteration=best_iter)
    pred_orig = np.expm1(np.clip(pred_log, 0, None))
    pred_orig = np.clip(pred_orig, 0, None)
    return pred_orig


val_pred_orig = predict_from_model(model, X_val, best_iter)
test_pred_orig = predict_from_model(model, X_test, best_iter)

# ============================================================
# 10. COMPUTE VALIDATION SCORE (official metric: RMSE on original scale)
# ============================================================
val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
print(f"\nValidation RMSE (original scale): {val_rmse:.4f}")
print(
    f"Val pred stats: mean={val_pred_orig.mean():.2f}, min={val_pred_orig.min():.2f}, max={val_pred_orig.max():.2f}"
)
print(
    f"Val true stats: mean={y_val_orig.mean():.2f}, min={y_val_orig.min():.2f}, max={y_val_orig.max():.2f}"
)

# ============================================================
# 11. GENERATE SUBMISSION
# ============================================================
if id_col and id_col in test.columns:
    submission_ids = test[id_col].values
else:
    submission_ids = np.arange(len(test))

submission = pd.DataFrame({"id": submission_ids.astype(int), "DEPSUM": test_pred_orig})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Test pred stats: mean={test_pred_orig.mean():.2f}, min={test_pred_orig.min():.2f}, max={test_pred_orig.max():.2f}"
)

# ============================================================
# 12. FEATURE IMPORTANCE (top 15)
# ============================================================
importance = model.feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features by gain:")
for i, (feat, imp) in enumerate(feat_imp.head(15).items()):
    print(f"  {i+1}. {feat}: {imp:.1f}")

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score}")
