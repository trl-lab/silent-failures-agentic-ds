import pandas as pd
import numpy as np
import os
import pickle
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target stats:\n{train['NADAC Per Unit'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES
# ============================================================
target_col = "NADAC Per Unit"
id_col = "id" if "id" in train.columns else None

if target_col in train.columns:
    y_train_full = train[target_col].copy()
    train = train.drop(columns=[target_col])
else:
    y_train_full = pd.Series(np.zeros(len(train)))
    print("WARNING: Target column not found!")

if id_col and id_col in train.columns:
    test_ids = (
        test[id_col].copy() if id_col in test.columns else pd.Series(range(len(test)))
    )
    train = train.drop(columns=[id_col])
    test = test.drop(columns=[id_col])
else:
    test_ids = pd.Series(range(len(test)))

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (BEFORE any stateful transforms)
# ============================================================
if y_train_full.notna().sum() > 0:
    valid_mask = y_train_full.notna()
    train_valid = train[valid_mask].copy()
    y_valid = y_train_full[valid_mask].copy()

    quantile_groups = pd.qcut(y_valid, q=10, labels=False, duplicates="drop")

    train_part, val_part, y_train_part, y_val_part, _, _ = train_test_split(
        train_valid,
        y_valid,
        quantile_groups,
        test_size=0.15,
        random_state=42,
        stratify=quantile_groups,
    )

    if (~valid_mask).sum() > 0:
        train_part = pd.concat([train_part, train[~valid_mask]], ignore_index=True)
        y_train_part = pd.concat(
            [y_train_part, y_train_full[~valid_mask]], ignore_index=True
        )
else:
    train_part = train.copy()
    val_part = train.iloc[: int(len(train) * 0.15)].copy()
    y_train_part = y_train_full.copy()
    y_val_part = y_train_full.iloc[: int(len(train) * 0.15)].copy()

print(f"Train part shape: {train_part.shape}")
print(f"Val part shape: {val_part.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
all_data = pd.concat([train_part, val_part, test], axis=0, ignore_index=True)

numeric_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = all_data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if id_col:
    numeric_cols = [c for c in numeric_cols if c != id_col]
    categorical_cols = [c for c in categorical_cols if c != id_col]

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# --- Target Encoding (fit on train_part only) ---
te_mappings = {}
global_target_mean = y_train_part.dropna().mean() if y_train_part.notna().any() else 0.0

for col in categorical_cols:
    tmp = pd.DataFrame({"col": train_part[col], "target": y_train_part})
    stats = tmp.groupby("col")["target"].agg(["mean", "count"])
    smoothing = 10
    stats["encoded"] = (
        stats["count"] * stats["mean"] + smoothing * global_target_mean
    ) / (stats["count"] + smoothing)
    stats = stats.reset_index()
    mapping = stats.set_index("col")["encoded"].to_dict()
    te_mappings[col] = mapping
    all_data[f"{col}_te"] = all_data[col].map(mapping).fillna(global_target_mean)

# Frequency encoding for low-cardinality categoricals
card_threshold = 15
low_card_cats = [
    col for col in categorical_cols if all_data[col].nunique() <= card_threshold
]

for col in low_card_cats:
    freq = train_part[col].value_counts(normalize=True)
    all_data[f"{col}_freq"] = all_data[col].map(freq).fillna(0)

# --- Handle numeric features ---
for col in numeric_cols:
    col_data = all_data[col].dropna()
    if len(col_data) > 0 and col_data.min() >= 0 and col_data.std() > 0:
        if col_data.skew() > 1.0:
            all_data[f"{col}_log"] = np.log1p(all_data[col].clip(lower=0))
        else:
            all_data[f"{col}_log"] = all_data[col]
    else:
        all_data[f"{col}_log"] = all_data[col]

# --- NDC code decomposition ---
ndc_col = None
for col in categorical_cols + numeric_cols:
    if "ndc" in col.lower():
        ndc_col = col
        break

if ndc_col is not None:
    ndc_str = all_data[ndc_col].astype(str)
    all_data["ndc_labeler"] = ndc_str.str[:5]
    all_data["ndc_product"] = ndc_str.str[5:9]
    all_data["ndc_package"] = ndc_str.str[9:11]

    for comp in ["ndc_labeler", "ndc_product", "ndc_package"]:
        if comp == "ndc_labeler":
            col_val = train_part[ndc_col].astype(str).str[:5]
        elif comp == "ndc_product":
            col_val = train_part[ndc_col].astype(str).str[5:9]
        else:
            col_val = train_part[ndc_col].astype(str).str[9:11]

        tmp = pd.DataFrame({"col": col_val, "target": y_train_part})
        stats = tmp.groupby("col")["target"].agg(["mean", "count"])
        stats["encoded"] = (
            stats["count"] * stats["mean"] + 10 * global_target_mean
        ) / (stats["count"] + 10)
        stats = stats.reset_index()
        mapping = stats.set_index("col")["encoded"].to_dict()
        all_data[f"{comp}_te"] = all_data[comp].map(mapping).fillna(global_target_mean)

# --- Interaction features ---
if len(categorical_cols) >= 2:
    cat_variances = []
    for col in categorical_cols[:10]:
        tmp = pd.DataFrame({"col": train_part[col], "target": y_train_part})
        grp_var = tmp.groupby("col")["target"].var().mean()
        cat_variances.append((col, grp_var))
    cat_variances.sort(key=lambda x: x[1], reverse=True)

    top_cats = [c for c, v in cat_variances[:3]]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            inter_col = f"{top_cats[i]}_x_{top_cats[j]}"
            all_data[inter_col] = (
                all_data[top_cats[i]].astype(str)
                + "_"
                + all_data[top_cats[j]].astype(str)
            )
            tmp = pd.DataFrame(
                {
                    "col": train_part[top_cats[i]].astype(str)
                    + "_"
                    + train_part[top_cats[j]].astype(str),
                    "target": y_train_part,
                }
            )
            stats = tmp.groupby("col")["target"].agg(["mean", "count"])
            stats["encoded"] = (
                stats["count"] * stats["mean"] + 10 * global_target_mean
            ) / (stats["count"] + 10)
            stats = stats.reset_index()
            mapping = stats.set_index("col")["encoded"].to_dict()
            all_data[f"{inter_col}_te"] = (
                all_data[inter_col].map(mapping).fillna(global_target_mean)
            )
            all_data = all_data.drop(columns=[inter_col])

# --- Determine final feature columns ---
cols_to_drop = categorical_cols[:]
for col in categorical_cols:
    if f"{col}_te" in all_data.columns:
        cols_to_drop.append(col)
    if f"{col}_freq" in all_data.columns:
        cols_to_drop.append(col)

# Remove raw NDC string components (we keep their target-encoded versions)
for comp in ["ndc_labeler", "ndc_product", "ndc_package"]:
    if comp in all_data.columns:
        cols_to_drop.append(comp)

feature_cols = [
    c
    for c in all_data.columns
    if c not in cols_to_drop and c not in [id_col, target_col]
]

for col in categorical_cols:
    if f"{col}_te" in all_data.columns and f"{col}_te" not in feature_cols:
        feature_cols.append(f"{col}_te")
    if f"{col}_freq" in all_data.columns and f"{col}_freq" not in feature_cols:
        feature_cols.append(f"{col}_freq")

for col in numeric_cols:
    if f"{col}_log" in all_data.columns and f"{col}_log" not in feature_cols:
        feature_cols.append(f"{col}_log")
    if col in all_data.columns and col not in feature_cols:
        feature_cols.append(col)

for comp in ["ndc_labeler", "ndc_product", "ndc_package"]:
    if f"{comp}_te" in all_data.columns and f"{comp}_te" not in feature_cols:
        feature_cols.append(f"{comp}_te")

feature_cols = list(dict.fromkeys(feature_cols))

# Safety: ensure all feature columns are numeric
numeric_feature_cols = []
for col in feature_cols:
    try:
        pd.to_numeric(all_data[col], errors='raise')
        numeric_feature_cols.append(col)
    except (ValueError, TypeError):
        print(f"WARNING: Dropping non-numeric feature column: {col}")
feature_cols = numeric_feature_cols

print(f"Total feature columns: {len(feature_cols)}")

# ============================================================
# 5. SPLIT BACK INTO TRAIN/VAL/TEST
# ============================================================
n_train = len(train_part)
n_val = len(val_part)

X_train = all_data.iloc[:n_train][feature_cols].copy()
y_train = y_train_part.copy().reset_index(drop=True)

X_val = all_data.iloc[n_train : n_train + n_val][feature_cols].copy()
y_val = y_val_part.copy().reset_index(drop=True)

X_test = all_data.iloc[n_train + n_val :][feature_cols].copy()
test_ids_final = test_ids.reset_index(drop=True)

# Ensure all columns are numeric or category
for col in feature_cols:
    if X_train[col].dtype == object or not np.issubdtype(X_train[col].dtype, np.number):
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")
        X_test[col] = X_test[col].astype("category")

# ============================================================
# 6. USE RAW TARGET (no transformation needed)
# ============================================================
y_train_raw = y_train.copy()
y_val_raw = y_val.copy()

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Target mean: {y_train_raw.mean():.4f}, std: {y_train_raw.std():.4f}")
print(f"Target min: {y_train_raw.min():.4f}, max: {y_train_raw.max():.4f}")

# ============================================================
# 7. MODEL: LightGBM Regressor
# ============================================================
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


# ============================================================
# 8. PREPARE DATA FOR LIGHTGBM
# ============================================================
# Ensure all features are numeric for LightGBM
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

print(f"LightGBM | Features: {len(feature_cols)}")

# ============================================================
# 10. TRAIN LIGHTGBM WITH EARLY STOPPING
# ============================================================
baseline_rmse = np.sqrt(np.mean((y_val_raw.values - y_train_raw.mean()) ** 2))
print(f"Baseline RMSE (mean prediction): {baseline_rmse:.4f}")

print("Starting LightGBM training with early stopping (patience=150)...")

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    X_train,
    y_train_raw,
    eval_set=[(X_val, y_val_raw)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=True)],
)

best_iteration = model_lgb.best_iteration_
val_pred = model_lgb.predict(X_val)
val_rmse = np.sqrt(np.mean((y_val_raw.values - val_pred) ** 2))
print(f"Training complete. Best iteration: {best_iteration}, Val RMSE: {val_rmse:.4f}")

# ============================================================
# 11. RETRAIN ON FULL DATA AND GENERATE PREDICTIONS
# ============================================================
val_rmse_final = val_rmse
print(f"Validation RMSE (final, original scale): {val_rmse_final:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Improvement over baseline: {baseline_rmse - val_rmse_final:.4f}")

# Retrain on full training data (train_part + val_part) with best iteration count
X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
y_full = pd.concat([y_train_raw, y_val_raw], axis=0, ignore_index=True)

n_estimators_full = best_iteration if best_iteration > 0 else lgb_params["n_estimators"]
model_full = lgb.LGBMRegressor(**{**lgb_params, "n_estimators": n_estimators_full})
model_full.fit(X_full, y_full)

test_pred = model_full.predict(X_test)
print(f"Full model trained with {n_estimators_full} estimators")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids_final.values.astype(int), "NADAC Per Unit": test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}, min={test_pred.min():.4f}, max={test_pred.max():.4f}"
)

model_full.booster_.save_model("./working/best_model.txt")

# ============================================================
# FINAL OUTPUT
# ============================================================
score = val_rmse_final
print(f"Final Validation Score: {score}")