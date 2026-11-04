import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

TARGET = "RATIO_ENROLL_EMP"

# Identify id column
id_col = None
for col in ["id", "ID", "Id"]:
    if col in train.columns:
        id_col = col
        break

# ============================================================
# 2. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any processing)
# ============================================================
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)
print(f"Train split: {train_data.shape}, Val split: {val_data.shape}")

# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================
feature_cols = [c for c in train.columns if c != TARGET and c != id_col]

numeric_cols = []
cat_cols = []

for col in feature_cols:
    if col in train_data.columns:
        if train_data[col].dtype in ["int64", "int32", "float64", "float32"]:
            numeric_cols.append(col)
        else:
            if train_data[col].dtype == "object" or train_data[col].nunique() < 50:
                cat_cols.append(col)
            else:
                numeric_cols.append(col)

print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(cat_cols)}")

# ============================================================
# 4. FEATURE ENGINEERING (fit on train_data only)
# ============================================================
# Fit imputers on train_data only
numeric_fill = {}
for col in numeric_cols:
    if col in train_data.columns:
        numeric_fill[col] = train_data[col].median()
        if pd.isna(numeric_fill[col]):
            numeric_fill[col] = 0.0

cat_fill = {}
for col in cat_cols:
    if col in train_data.columns:
        mode_val = train_data[col].mode()
        cat_fill[col] = mode_val[0] if len(mode_val) > 0 else "UNKNOWN"


def engineer_features(df, numeric_fill, cat_fill, numeric_cols, cat_cols):
    df = df.copy()

    # Fill missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(numeric_fill.get(col, 0.0))

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(cat_fill.get(col, "UNKNOWN"))

    # Interaction features for numeric columns
    if len(numeric_cols) >= 2:
        key_numeric = [c for c in numeric_cols if c in df.columns][:8]
        for i in range(len(key_numeric)):
            for j in range(i + 1, min(i + 3, len(key_numeric))):
                c1, c2 = key_numeric[i], key_numeric[j]
                denom = df[c2].replace(0, np.nan)
                df[f"{c1}_div_{c2}"] = (df[c1] / denom).fillna(0)
                df[f"{c1}_mul_{c2}"] = df[c1] * df[c2]
                df[f"{c1}_diff_{c2}"] = df[c1] - df[c2]

    # Log transform for highly skewed numeric features
    for col in numeric_cols:
        if col in df.columns:
            vals = df[col].values
            if vals.min() >= 0 and vals.std() > 0:
                skew = pd.Series(vals).skew()
                if abs(skew) > 1.0:
                    df[f"{col}_log"] = np.log1p(np.abs(vals)) * np.sign(vals)

    # Cardinality features for categoricals
    for col in cat_cols:
        if col in df.columns:
            df[f"{col}_nunique"] = df[col].nunique()

    return df


train_processed = engineer_features(
    train_data, numeric_fill, cat_fill, numeric_cols, cat_cols
)
val_processed = engineer_features(
    val_data, numeric_fill, cat_fill, numeric_cols, cat_cols
)
test_processed = engineer_features(test, numeric_fill, cat_fill, numeric_cols, cat_cols)

# ============================================================
# 5. ENCODE CATEGORICALS (fit on train_data only)
# ============================================================
all_cat_cols = [c for c in cat_cols if c in train_processed.columns]

if len(all_cat_cols) > 0:
    try:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_processed[all_cat_cols] = oe.fit_transform(train_processed[all_cat_cols])
        val_processed[all_cat_cols] = oe.transform(val_processed[all_cat_cols])
        test_processed[all_cat_cols] = oe.transform(test_processed[all_cat_cols])
    except Exception as e:
        print(f"Ordinal encoding issue: {e}, falling back to label encoding")
        for col in all_cat_cols:
            mapping = {v: i for i, v in enumerate(train_data[col].unique())}
            train_processed[col] = (
                train_processed[col].map(mapping).fillna(-1).astype(int)
            )
            val_processed[col] = val_processed[col].map(mapping).fillna(-1).astype(int)
            test_processed[col] = (
                test_processed[col].map(mapping).fillna(-1).astype(int)
            )

# ============================================================
# 6. TARGET TRANSFORMATION (log1p for skewed target)
# ============================================================
target_vals = train_data[TARGET].values
print(
    f"Target stats - min: {target_vals.min():.4f}, max: {target_vals.max():.4f}, mean: {target_vals.mean():.4f}"
)
print(f"Target skew: {pd.Series(target_vals).skew():.2f}")

use_log_target = (target_vals.min() >= 0) and (abs(pd.Series(target_vals).skew()) > 1.0)

if use_log_target:
    train_processed[TARGET] = np.log1p(train_data[TARGET])
    val_processed[TARGET] = np.log1p(val_data[TARGET])
    print(
        f"Applied log1p to target. New range: [{train_processed[TARGET].min():.4f}, {train_processed[TARGET].max():.4f}]"
    )
else:
    train_processed[TARGET] = train_data[TARGET]
    val_processed[TARGET] = val_data[TARGET]
    print("Target not transformed")

# ============================================================
# 7. FINALIZE FEATURE COLUMNS
# ============================================================
final_feature_cols = [c for c in train_processed.columns if c != TARGET and c != id_col]

# Ensure all features are numeric
for col in final_feature_cols:
    if train_processed[col].dtype == "object":
        mapping = {v: i for i, v in enumerate(train_processed[col].unique())}
        train_processed[col] = train_processed[col].map(mapping).fillna(-1).astype(int)
        val_processed[col] = val_processed[col].map(mapping).fillna(-1).astype(int)
        test_processed[col] = test_processed[col].map(mapping).fillna(-1).astype(int)
    elif not np.issubdtype(train_processed[col].dtype, np.number):
        train_processed[col] = pd.to_numeric(
            train_processed[col], errors="coerce"
        ).fillna(0)
        val_processed[col] = pd.to_numeric(val_processed[col], errors="coerce").fillna(
            0
        )
        test_processed[col] = pd.to_numeric(
            test_processed[col], errors="coerce"
        ).fillna(0)

# Replace any remaining inf values
for df_ref in [train_processed, val_processed, test_processed]:
    df_ref[final_feature_cols] = df_ref[final_feature_cols].replace(
        [np.inf, -np.inf], 0
    )

print(f"Final feature count: {len(final_feature_cols)}")
print(
    f"Train: {train_processed.shape}, Val: {val_processed.shape}, Test: {test_processed.shape}"
)

# ============================================================
# 8. PREPARE NUMERIC MATRICES
# ============================================================
X_train = train_processed[final_feature_cols].values.astype(np.float64)
y_train = train_processed[TARGET].values.astype(np.float64)
X_val = val_processed[final_feature_cols].values.astype(np.float64)
y_val = val_processed[TARGET].values.astype(np.float64)
X_test = test_processed[final_feature_cols].values.astype(np.float64)

# Clean any remaining NaN/Inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

print(
    f"X_train: {X_train.shape}, y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]"
)


# ============================================================
# 9. TRAIN LIGHTGBM (Huber loss, multi-seed for robustness)
# ============================================================
def train_lgb(seed):
    params = {
        "objective": "huber",
        "huber_delta": 1.0,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }

    lgb_train = lgb.Dataset(
        X_train, label=y_train, feature_name=final_feature_cols, free_raw_data=True
    )
    lgb_val = lgb.Dataset(
        X_val,
        label=y_val,
        feature_name=final_feature_cols,
        reference=lgb_train,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    )
    return model


seeds = [42, 123, 7]
val_preds = []
test_preds = []

for i, seed in enumerate(seeds):
    print(f"Training seed {seed} ({i+1}/{len(seeds)})...")
    model = train_lgb(seed)
    print(f"  Best iteration: {model.best_iteration}")

    vp = model.predict(X_val, num_iteration=model.best_iteration)
    tp = model.predict(X_test, num_iteration=model.best_iteration)
    val_preds.append(vp)
    test_preds.append(tp)

# Average predictions across seeds
val_pred_log = np.mean(val_preds, axis=0)
test_pred_log = np.mean(test_preds, axis=0)

# ============================================================
# 10. INVERSE TRANSFORM AND COMPUTE OFFICIAL METRIC (RMSE)
# ============================================================
if use_log_target:
    val_pred = np.expm1(val_pred_log)
    test_pred = np.expm1(test_pred_log)
    val_pred = np.clip(val_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)
    y_val_original = np.expm1(y_val)
else:
    val_pred = val_pred_log
    test_pred = test_pred_log
    y_val_original = y_val

val_rmse = root_mean_squared_error(y_val_original, val_pred)
print(f"\nValidation RMSE (original scale): {val_rmse:.6f}")
print(
    f"Val pred stats - min: {val_pred.min():.4f}, max: {val_pred.max():.4f}, mean: {val_pred.mean():.4f}"
)
print(
    f"Test pred stats - min: {test_pred.min():.4f}, max: {test_pred.max():.4f}, mean: {test_pred.mean():.4f}"
)

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

if id_col is not None:
    test_ids = test[id_col].values.astype(int)
else:
    test_ids = np.arange(len(test))

submission = pd.DataFrame({"id": test_ids, "RATIO_ENROLL_EMP": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# ============================================================
# 12. FINAL OUTPUT
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score}")
