import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Contract Award Amount (USD)"

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(
    f"Target stats: mean={train[target_col].mean():.2f}, median={train[target_col].median():.2f}, skew={train[target_col].skew():.2f}"
)

# ============================================================
# 2. SPLIT FIRST (leakage prevention)
# ============================================================
y = train[target_col].copy()
train_no_target = train.drop(columns=[target_col])

X_train, X_val, y_train, y_val = train_test_split(
    train_no_target, y, test_size=0.15, random_state=42
)

print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={test.shape[0]}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = [c for c in X_train.columns if X_train[c].dtype == bool]
id_cols = [c for c in X_train.columns if "id" in c.lower() or "ID" in c]

print(
    f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, ID cols: {id_cols}"
)

# Combine for feature engineering
all_data = pd.concat([X_train, X_val, test], axis=0, ignore_index=True)

# --- Interaction features ---
interaction_features = []
if len(numeric_cols) >= 2:
    top_numeric = numeric_cols[:10]
    for i in range(len(top_numeric)):
        for j in range(i + 1, min(i + 4, len(top_numeric))):
            feat_name = f"{top_numeric[i]}_x_{top_numeric[j]}"
            all_data[feat_name] = all_data[top_numeric[i]].fillna(0) * all_data[
                top_numeric[j]
            ].fillna(0)
            interaction_features.append(feat_name)

    for i in range(len(top_numeric)):
        for j in range(len(top_numeric)):
            if i != j:
                feat_name = f"{top_numeric[i]}_div_{top_numeric[j]}"
                denom = all_data[top_numeric[j]].replace(0, np.nan)
                all_data[feat_name] = all_data[top_numeric[i]].fillna(0) / denom
                all_data[feat_name] = all_data[feat_name].replace(
                    [np.inf, -np.inf], np.nan
                )
                interaction_features.append(feat_name)

print(f"Created {len(interaction_features)} interaction features")

# --- Target encoding (train only) ---
target_enc_features = []
for col in categorical_cols:
    if col in X_train.columns:
        train_with_target = X_train.copy()
        train_with_target["_target"] = y_train.values

        cat_target_mean = train_with_target.groupby(col)["_target"].mean()
        cat_target_median = train_with_target.groupby(col)["_target"].median()
        cat_target_std = train_with_target.groupby(col)["_target"].std()
        cat_target_count = train_with_target.groupby(col)["_target"].count()

        global_mean = y_train.mean()

        all_data[f"{col}_target_mean"] = (
            all_data[col].map(cat_target_mean).fillna(global_mean)
        )
        all_data[f"{col}_target_median"] = (
            all_data[col].map(cat_target_median).fillna(global_mean)
        )
        all_data[f"{col}_target_std"] = all_data[col].map(cat_target_std).fillna(0)
        all_data[f"{col}_target_count"] = all_data[col].map(cat_target_count).fillna(0)

        target_enc_features.extend(
            [
                f"{col}_target_mean",
                f"{col}_target_median",
                f"{col}_target_std",
                f"{col}_target_count",
            ]
        )

print(f"Created {len(target_enc_features)} target encoding features")

# --- Frequency encoding (train only) ---
freq_features = []
for col in categorical_cols:
    if col in X_train.columns:
        freq = X_train[col].value_counts(normalize=True)
        all_data[f"{col}_freq"] = all_data[col].map(freq).fillna(0)
        freq_features.append(f"{col}_freq")

print(f"Created {len(freq_features)} frequency features")

# --- Numeric transforms ---
numeric_eng_features = []
for col in numeric_cols:
    if col in all_data.columns:
        if all_data[col].min() >= 0:
            all_data[f"{col}_log"] = np.log1p(all_data[col].clip(lower=0))
            numeric_eng_features.append(f"{col}_log")
        all_data[f"{col}_sq"] = all_data[col] ** 2
        numeric_eng_features.append(f"{col}_sq")
        all_data[f"{col}_cbrt"] = np.cbrt(all_data[col])
        numeric_eng_features.append(f"{col}_cbrt")

print(f"Created {len(numeric_eng_features)} numeric engineering features")

# --- Binning ---
bin_features = []
for col in numeric_cols:
    if col in all_data.columns and all_data[col].nunique() > 5:
        train_vals = X_train[col].dropna()
        if len(train_vals) > 10:
            quantiles = np.quantile(train_vals, [0.25, 0.5, 0.75])
            all_data[f"{col}_bin"] = pd.cut(
                all_data[col],
                bins=[-np.inf] + list(quantiles) + [np.inf],
                labels=False,
                include_lowest=True,
            )
            bin_features.append(f"{col}_bin")

print(f"Created {len(bin_features)} binning features")

# --- Categorical cross features ---
cross_features = []
if len(categorical_cols) >= 2:
    top_cats = categorical_cols[:5]
    for i in range(len(top_cats)):
        for j in range(i + 1, min(i + 3, len(top_cats))):
            feat_name = f"{top_cats[i]}_x_{top_cats[j]}"
            all_data[feat_name] = (
                all_data[top_cats[i]].astype(str)
                + "_"
                + all_data[top_cats[j]].astype(str)
            )
            cross_features.append(feat_name)

print(f"Created {len(cross_features)} cross features")

# ============================================================
# 4. SPLIT BACK INTO TRAIN/VAL/TEST
# ============================================================
n_train = len(X_train)
n_val = len(X_val)

X_train_proc = all_data.iloc[:n_train].copy().reset_index(drop=True)
X_val_proc = all_data.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
X_test_proc = all_data.iloc[n_train + n_val :].copy().reset_index(drop=True)

# Clean inf values
X_train_proc = X_train_proc.replace([np.inf, -np.inf], np.nan)
X_val_proc = X_val_proc.replace([np.inf, -np.inf], np.nan)
X_test_proc = X_test_proc.replace([np.inf, -np.inf], np.nan)

# Feature columns (exclude ID cols)
exclude_cols = [c for c in id_cols if c in X_train_proc.columns]
feature_cols = [c for c in X_train_proc.columns if c not in exclude_cols]

print(f"Final feature count: {len(feature_cols)}")

# Select feature columns
X_train_proc = X_train_proc[feature_cols]
X_val_proc = X_val_proc[feature_cols]
X_test_proc = X_test_proc[feature_cols]

# Convert all object/categorical columns to label-encoded integers
# (LightGBM requires numeric input; we already have target/freq encodings as separate features)
object_cols = X_train_proc.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"Converting {len(object_cols)} object columns to label-encoded integers")
for col in object_cols:
    combined = pd.concat(
        [X_train_proc[col], X_val_proc[col], X_test_proc[col]], axis=0
    ).astype(str)
    categories = combined.unique()
    cat_to_num = {c: i for i, c in enumerate(categories)}
    X_train_proc[col] = X_train_proc[col].astype(str).map(cat_to_num)
    X_val_proc[col] = X_val_proc[col].astype(str).map(cat_to_num)
    X_test_proc[col] = X_test_proc[col].astype(str).map(cat_to_num)

# Ensure all columns are numeric
for col in feature_cols:
    if X_train_proc[col].dtype == object:
        X_train_proc[col] = pd.to_numeric(X_train_proc[col], errors="coerce")
        X_val_proc[col] = pd.to_numeric(X_val_proc[col], errors="coerce")
        X_test_proc[col] = pd.to_numeric(X_test_proc[col], errors="coerce")

# ============================================================
# 5. TRAIN MODELS
# ============================================================
HUBER_DELTA = 5000

lgb_params_base = {
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

model_configs = [
    {"name": "l2", "objective": "regression", "metric": "rmse"},
    {"name": "l1", "objective": "regression_l1", "metric": "l1"},
    {"name": "huber", "objective": "huber", "metric": "rmse", "alpha": HUBER_DELTA},
]

results = []

for config in model_configs:
    params = lgb_params_base.copy()
    params["objective"] = config["objective"]
    params["metric"] = config["metric"]
    if "alpha" in config:
        params["alpha"] = config["alpha"]

    dtrain = lgb.Dataset(
        X_train_proc,
        label=y_train.values,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val_proc, label=y_val.values, feature_name=feature_cols, free_raw_data=False
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    val_pred = model.predict(X_val_proc, num_iteration=model.best_iteration)
    val_pred = np.clip(val_pred, 0, None)
    val_rmse = np.sqrt(mean_squared_error(y_val.values, val_pred))

    results.append(
        {
            "name": config["name"],
            "rmse": val_rmse,
            "model": model,
            "best_iter": model.best_iteration,
        }
    )
    print(f"  {config['name']}: RMSE={val_rmse:.4f}, best_iter={model.best_iteration}")

# Re-select best model strictly by validation RMSE (official metric)
# This ensures the model chosen is the one that truly minimizes RMSE,
# regardless of internal training objective differences.
best = min(results, key=lambda x: x["rmse"])
print(f"\nBest model (by RMSE): {best['name']} with val RMSE={best['rmse']:.4f}")

# ============================================================
# 6. GENERATE PREDICTIONS FROM BEST MODEL
# ============================================================
test_pred = best["model"].predict(X_test_proc, num_iteration=best["best_iter"])
test_pred = np.clip(test_pred, 0, None)

# ============================================================
# 7. SAVE SUBMISSION
# ============================================================
if "id" in test.columns:
    test_ids = test["id"].values
elif "Id" in test.columns:
    test_ids = test["Id"].values
else:
    test_ids = np.arange(len(test))

submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Contract Award Amount (USD)": test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 8. FINAL VALIDATION SCORE (official metric: RMSE on original scale)
# ============================================================
score = best["rmse"]
print(f"Final Validation Score: {score}")