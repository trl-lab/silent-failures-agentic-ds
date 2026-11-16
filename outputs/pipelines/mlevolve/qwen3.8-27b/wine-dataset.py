import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import json

# ============================================================
# 1. LOAD DATA
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

train_full = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_full.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {train_full.columns.tolist()}")
print(f"\nTarget stats:\n{train_full['Price'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
target_col = "Price"
id_col = "id" if "id" in train_full.columns else None

y_full = train_full[target_col].copy()
train_full = train_full.drop(columns=[target_col])

cat_cols = train_full.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = train_full.select_dtypes(include=["number"]).columns.tolist()

if id_col and id_col in num_cols:
    num_cols.remove(id_col)

# Drop ultra-high-cardinality features (>50% unique values) as they act as IDs
high_card_cols = []
for col in cat_cols:
    if train_full[col].nunique() / len(train_full) > 0.5:
        high_card_cols.append(col)
print(f"Dropping high-cardinality columns: {high_card_cols}")
cat_cols = [c for c in cat_cols if c not in high_card_cols]

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT (BEFORE feature engineering)
# ============================================================
train_df, val_df = train_test_split(train_full, test_size=0.15, random_state=42)
y_train = y_full.loc[train_df.index]
y_val = y_full.loc[val_df.index]
# Price is already in log-space, use directly
y_train_log = y_train.copy()
y_val_log = y_val.copy()

print(f"Train split: {len(train_df)}, Val split: {len(val_df)}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test.copy()

# 4a. Handle missing values in numerical columns
num_median = train_df[num_cols].median()
for col in num_cols:
    train_df[col] = train_df[col].fillna(num_median[col])
    val_df[col] = val_df[col].fillna(num_median[col])
    test_df[col] = test_df[col].fillna(num_median[col])

# 4b. Handle missing values in categorical columns
for col in cat_cols:
    train_df[col] = train_df[col].fillna("Unknown")
    val_df[col] = val_df[col].fillna("Unknown")
    test_df[col] = test_df[col].fillna("Unknown")

# 4c. Target encoding for high-cardinality categoricals
global_mean = y_train_log.mean()
for col in cat_cols:
    if train_df[col].nunique() > 5:
        col_means = y_train_log.groupby(train_df[col]).mean()
        col_counts = y_train_log.groupby(train_df[col]).count()
        smoothing = 20
        smoothed_enc = (col_counts * col_means + smoothing * global_mean) / (
            col_counts + smoothing
        )
        train_df[f"{col}_target_enc"] = (
            train_df[col].map(smoothed_enc).fillna(global_mean)
        )
        val_df[f"{col}_target_enc"] = val_df[col].map(smoothed_enc).fillna(global_mean)
        test_df[f"{col}_target_enc"] = (
            test_df[col].map(smoothed_enc).fillna(global_mean)
        )


# 4d. Interaction features
def find_col(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None


region_col = find_col(train_df, ["region", "appellation", "area", "district"])
variety_col = find_col(train_df, ["variety", "grape", "varietal"])
producer_col = find_col(
    train_df, ["producer", "winery", "winery_name", "maker", "brand"]
)
vintage_col = find_col(train_df, ["vintage", "year"])
country_col = find_col(train_df, ["country", "nation"])

interactions = []

if region_col and vintage_col:
    train_df["region_vintage"] = (
        train_df[region_col].astype(str) + "_" + train_df[vintage_col].astype(str)
    )
    val_df["region_vintage"] = (
        val_df[region_col].astype(str) + "_" + val_df[vintage_col].astype(str)
    )
    test_df["region_vintage"] = (
        test_df[region_col].astype(str) + "_" + test_df[vintage_col].astype(str)
    )
    interactions.append("region_vintage")

if variety_col and region_col:
    train_df["variety_region"] = (
        train_df[variety_col].astype(str) + "_" + train_df[region_col].astype(str)
    )
    val_df["variety_region"] = (
        val_df[variety_col].astype(str) + "_" + val_df[region_col].astype(str)
    )
    test_df["variety_region"] = (
        test_df[variety_col].astype(str) + "_" + test_df[region_col].astype(str)
    )
    interactions.append("variety_region")

if producer_col and variety_col:
    train_df["producer_variety"] = (
        train_df[producer_col].astype(str) + "_" + train_df[variety_col].astype(str)
    )
    val_df["producer_variety"] = (
        val_df[producer_col].astype(str) + "_" + val_df[variety_col].astype(str)
    )
    test_df["producer_variety"] = (
        test_df[producer_col].astype(str) + "_" + test_df[variety_col].astype(str)
    )
    interactions.append("producer_variety")

if producer_col and region_col:
    train_df["producer_region"] = (
        train_df[producer_col].astype(str) + "_" + train_df[region_col].astype(str)
    )
    val_df["producer_region"] = (
        val_df[producer_col].astype(str) + "_" + val_df[region_col].astype(str)
    )
    test_df["producer_region"] = (
        test_df[producer_col].astype(str) + "_" + test_df[region_col].astype(str)
    )
    interactions.append("producer_region")

if country_col and region_col:
    train_df["country_region"] = (
        train_df[country_col].astype(str) + "_" + train_df[region_col].astype(str)
    )
    val_df["country_region"] = (
        val_df[country_col].astype(str) + "_" + val_df[region_col].astype(str)
    )
    test_df["country_region"] = (
        test_df[country_col].astype(str) + "_" + test_df[region_col].astype(str)
    )
    interactions.append("country_region")

# Target-encode interaction features
for inter_col in interactions:
    inter_means = y_train_log.groupby(train_df[inter_col]).mean()
    inter_counts = y_train_log.groupby(train_df[inter_col]).count()
    smoothing = 20
    smoothed_inter = (inter_counts * inter_means + smoothing * global_mean) / (
        inter_counts + smoothing
    )
    train_df[f"{inter_col}_enc"] = (
        train_df[inter_col].map(smoothed_inter).fillna(global_mean)
    )
    val_df[f"{inter_col}_enc"] = (
        val_df[inter_col].map(smoothed_inter).fillna(global_mean)
    )
    test_df[f"{inter_col}_enc"] = (
        test_df[inter_col].map(smoothed_inter).fillna(global_mean)
    )

# 4e. Vintage-based features
if vintage_col:
    current_year = 2024
    train_df["vintage_num"] = pd.to_numeric(train_df[vintage_col], errors="coerce")
    val_df["vintage_num"] = pd.to_numeric(val_df[vintage_col], errors="coerce")
    test_df["vintage_num"] = pd.to_numeric(test_df[vintage_col], errors="coerce")
    train_df["vintage_age"] = current_year - train_df["vintage_num"]
    val_df["vintage_age"] = current_year - val_df["vintage_num"]
    test_df["vintage_age"] = current_year - test_df["vintage_num"]

# 4f. Log-transform skewed numerical features
for col in num_cols:
    if train_df[col].min() >= 0 and train_df[col].skew() > 1:
        train_df[f"{col}_log"] = np.log1p(train_df[col])
        val_df[f"{col}_log"] = np.log1p(val_df[col])
        test_df[f"{col}_log"] = np.log1p(test_df[col])

# 4g. Frequency encoding
for col in cat_cols:
    freq = train_df[col].value_counts(normalize=True)
    train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
    val_df[f"{col}_freq"] = val_df[col].map(freq).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

# 4h. Drop original interaction string columns
for inter_col in interactions:
    if inter_col in train_df.columns:
        train_df = train_df.drop(columns=[inter_col])
        val_df = val_df.drop(columns=[inter_col])
        test_df = test_df.drop(columns=[inter_col])

# Final feature columns
feature_cols = [c for c in train_df.columns if c != id_col]
cat_cols_final = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols_final = train_df.select_dtypes(include=["number"]).columns.tolist()

print(f"Total features after engineering: {len(feature_cols)}")
print(f"Categorical features: {cat_cols_final}")
print(f"Numerical features: {len(num_cols_final)}")


# ============================================================
# 5. PREPARE DATA FOR MODELING
# ============================================================
def prepare_lgb_data(df, feature_cols, cat_cols):
    df = df[feature_cols].copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


train_lgb = prepare_lgb_data(train_df, feature_cols, cat_cols_final)
val_lgb = prepare_lgb_data(val_df, feature_cols, cat_cols_final)
test_lgb = prepare_lgb_data(test_df, feature_cols, cat_cols_final)

# ============================================================
# 6. MODEL PARAMETERS
# ============================================================
base_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 3,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "seed": 42,
}

base_params_2 = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.75,
    "bagging_freq": 3,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "seed": 123,
}

# ============================================================
# 7. 5-FOLD CROSS-VALIDATION
# ============================================================
from sklearn.model_selection import KFold

print("\n--- 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
cv_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_lgb)):
    dtrain_cv = lgb.Dataset(train_lgb.iloc[tr_idx], label=y_train_log.values[tr_idx])
    dval_cv = lgb.Dataset(train_lgb.iloc[va_idx], label=y_train_log.values[va_idx], reference=dtrain_cv)

    cv_model = lgb.train(
        base_params,
        dtrain_cv,
        num_boost_round=5000,
        valid_sets=[dval_cv],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    )

    cv_pred = cv_model.predict(train_lgb.iloc[va_idx], num_iteration=cv_model.best_iteration)
    cv_rmse = root_mean_squared_error(y_train_log.values[va_idx], cv_pred)
    cv_scores.append(cv_rmse)
    cv_best_iters.append(cv_model.best_iteration)
    print(f"  Fold {fold+1}: RMSE={cv_rmse:.4f}, best_iter={cv_model.best_iteration}")

mean_cv_rmse = np.mean(cv_scores)
std_cv_rmse = np.std(cv_scores)
print(f"CV Mean RMSE: {mean_cv_rmse:.4f} (+/- {std_cv_rmse:.4f})")
print(f"CV Best Iters: {cv_best_iters}, mean={np.mean(cv_best_iters):.0f}")

# ============================================================
# 8. TRAIN FINAL MODELS ON ALL DATA
# ============================================================
print("\n--- Training Final Models on All Data ---")
n_estimators = int(np.mean(cv_best_iters))

# Model 1
dtrain_full = lgb.Dataset(train_lgb, label=y_train_log.values)
model_1 = lgb.train(
    base_params,
    dtrain_full,
    num_boost_round=n_estimators,
    callbacks=[lgb.log_evaluation(-1)],
)

# Model 2 (different seed and slightly different params)
dtrain_full_2 = lgb.Dataset(train_lgb, label=y_train_log.values)
model_2 = lgb.train(
    base_params_2,
    dtrain_full_2,
    num_boost_round=n_estimators,
    callbacks=[lgb.log_evaluation(-1)],
)

# ============================================================
# 9. GENERATE TEST PREDICTIONS (2-model average)
# ============================================================
test_pred_1 = model_1.predict(test_lgb, num_iteration=n_estimators)
test_pred_2 = model_2.predict(test_lgb, num_iteration=n_estimators)
test_final_pred = (test_pred_1 + test_pred_2) / 2.0

# ============================================================
# 10. SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_final_pred)), "Price": test_final_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: min={test_final_pred.min():.2f}, max={test_final_pred.max():.2f}, mean={test_final_pred.mean():.2f}"
)

# ============================================================
# 11. SAVE MODELS FOR REPRODUCIBILITY
# ============================================================
os.makedirs("./working/models", exist_ok=True)
model_1.save_model("./working/models/model_1.txt")
model_2.save_model("./working/models/model_2.txt")

metadata = {
    "cv_mean_rmse": float(mean_cv_rmse),
    "cv_std_rmse": float(std_cv_rmse),
    "cv_best_iterations": cv_best_iters,
    "n_estimators": n_estimators,
    "base_params": base_params,
    "base_params_2": base_params_2,
}
with open("./working/models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModels saved to ./working/models/")

# ============================================================
# 12. FINAL VALIDATION SCORE
# ============================================================
score = mean_cv_rmse
print(f"Final Validation Score: {score}")
