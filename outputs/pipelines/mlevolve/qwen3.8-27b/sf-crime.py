import pandas as pd
import numpy as np
import os
import re
import json
import lightgbm as lgb
from sklearn.metrics import log_loss

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# ============================================================
# 2. CLEAN DATA
# ============================================================
def clean_coordinates(df):
    df = df.copy()
    df["Y"] = df["Y"].clip(37.65, 37.90)
    df["X"] = df["X"].clip(-122.60, -122.30)
    return df


train = clean_coordinates(train)
test = clean_coordinates(test)

train["Dates"] = pd.to_datetime(train["Dates"])
test["Dates"] = pd.to_datetime(test["Dates"])


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def parse_address(address):
    if pd.isna(address):
        return np.nan, np.nan
    match = re.match(r"(\d+)\s*Block of\s*(.+)", str(address))
    if match:
        return int(match.group(1)), match.group(2).strip()
    match = re.match(r"(\d+)\s*Block\s*(.+)", str(address))
    if match:
        return int(match.group(1)), match.group(2).strip()
    return np.nan, np.nan


def engineer_features(df):
    df = df.copy()

    # Temporal features
    df["hour"] = df["Dates"].dt.hour
    df["day_of_week"] = df["Dates"].dt.dayofweek
    df["month"] = df["Dates"].dt.month
    df["year"] = df["Dates"].dt.year
    df["week_of_year"] = df["Dates"].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df["Dates"].dt.dayofyear
    df["quarter"] = df["Dates"].dt.quarter

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Binary temporal
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_evening"] = ((df["hour"] >= 17) & (df["hour"] < 22)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 17)).astype(int)
    df["is_morning"] = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)

    # Spatial features
    sf_center_lat, sf_center_lon = 37.7749, -122.4194
    df["dist_from_center"] = np.sqrt(
        (df["Y"] - sf_center_lat) ** 2 + (df["X"] - sf_center_lon) ** 2
    )
    df["grid_lat"] = (df["Y"] * 100).round().astype(int)
    df["grid_lon"] = (df["X"] * 100).round().astype(int)
    df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)
    df["fine_grid_lat"] = (df["Y"] * 200).round().astype(int)
    df["fine_grid_lon"] = (df["X"] * 200).round().astype(int)
    df["fine_grid_cell"] = (
        df["fine_grid_lat"].astype(str) + "_" + df["fine_grid_lon"].astype(str)
    )

    # Address features
    if "Address" in df.columns:
        parsed = df["Address"].apply(parse_address)
        df["block_number"] = parsed.apply(lambda x: x[0])
        df["street_name"] = parsed.apply(lambda x: x[1])
        df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce")
        df["block_log"] = np.log1p(df["block_number"].fillna(0))
    else:
        df["block_number"] = np.nan
        df["street_name"] = np.nan
        df["block_log"] = 0.0

    # Categorical cleanup
    for col in ["PdDistrict", "Descript", "Resolution", "DayOfWeek"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # Interaction features
    df["hour_x_district"] = df["hour"].astype(str) + "_" + df["PdDistrict"].astype(str)
    df["month_x_district"] = (
        df["month"].astype(str) + "_" + df["PdDistrict"].astype(str)
    )
    df["dow_x_district"] = (
        df["day_of_week"].astype(str) + "_" + df["PdDistrict"].astype(str)
    )
    df["hour_x_grid"] = df["hour"].astype(str) + "_" + df["grid_cell"]

    return df


print("Engineering features...")
train = engineer_features(train)
test = engineer_features(test)

# ============================================================
# 5. TIME-BASED SPLIT
# ============================================================
train = train.sort_values("Dates").reset_index(drop=True)
split_idx = int(len(train) * 0.9)
train_split = train.iloc[:split_idx].copy()
val_split = train.iloc[split_idx:].copy()

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 4. FREQUENCY ENCODING (fit on train_split only)
# ============================================================
freq_cols = [
    "Descript",
    "Resolution",
    "PdDistrict",
    "street_name",
    "grid_cell",
    "fine_grid_cell",
]
freq_maps = {}

for col in freq_cols:
    if col in train_split.columns:
        freq_maps[col] = train_split[col].value_counts(normalize=True)


def apply_freq_encoding(df, freq_maps):
    df = df.copy()
    for col, freq in freq_maps.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)
    return df


train_split = apply_freq_encoding(train_split, freq_maps)
val_split = apply_freq_encoding(val_split, freq_maps)
test = apply_freq_encoding(test, freq_maps)

# Interaction frequency encodings
for col in ["hour_x_district", "month_x_district", "dow_x_district"]:
    if col in train_split.columns:
        freq = train_split[col].value_counts(normalize=True)
        train_split[f"{col}_freq"] = train_split[col].map(freq).fillna(0)
        val_split[f"{col}_freq"] = val_split[col].map(freq).fillna(0)
        test[f"{col}_freq"] = test[col].map(freq).fillna(0)

# ============================================================
# 6. DEFINE FEATURE COLUMNS
# ============================================================
numeric_features = [
    "hour",
    "day_of_week",
    "month",
    "year",
    "week_of_year",
    "day_of_year",
    "quarter",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_night",
    "is_evening",
    "is_afternoon",
    "is_morning",
    "X",
    "Y",
    "dist_from_center",
    "block_number",
    "block_log",
    "grid_lat",
    "grid_lon",
    "fine_grid_lat",
    "fine_grid_lon",
]

freq_feature_names = [
    f"{col}_freq" for col in freq_cols if f"{col}_freq" in train.columns
]
freq_feature_names += [
    f"{col}_freq"
    for col in ["hour_x_district", "month_x_district", "dow_x_district"]
    if f"{col}_freq" in train.columns
]

categorical_features = [
    "PdDistrict",
    "Descript",
    "Resolution",
    "DayOfWeek",
    "street_name",
]

all_features = (
    [c for c in numeric_features if c in train.columns]
    + [c for c in freq_feature_names if c in train.columns]
    + [c for c in categorical_features if c in train.columns]
)

categories = sorted(train["Category"].unique().tolist())
n_categories = len(categories)

print(f"Total features: {len(all_features)}, Classes: {n_categories}")

# ============================================================
# 7. PREPARE LABELS AND FEATURE MATRICES
# ============================================================
cat_to_idx = {c: i for i, c in enumerate(categories)}
train_labels = train_split["Category"].map(cat_to_idx).values
val_labels = val_split["Category"].map(cat_to_idx).values

assert not np.isnan(train_labels).any(), "Unmapped labels in train!"
assert not np.isnan(val_labels).any(), "Unmapped labels in val!"

# Convert categorical columns to category dtype (required by LightGBM sklearn API)
for col in categorical_features:
    if col in train_split.columns:
        train_split[col] = train_split[col].astype(str)
        val_split[col] = val_split[col].astype(str)
        test[col] = test[col].astype(str)
        # Fit categories on train only to avoid leakage
        cats = train_split[col].astype("category")
        all_cats = list(cats.cat.categories)
        # Add any new categories from val/test
        for df_tmp in [val_split, test]:
            new_cats = set(df_tmp[col].unique()) - set(all_cats)
            if new_cats:
                all_cats.extend(sorted(new_cats))
        cats = sorted(all_cats)
        train_split[col] = train_split[col].astype("category").cat.set_categories(cats)
        val_split[col] = val_split[col].astype("category").cat.set_categories(cats)
        test[col] = test[col].astype("category").cat.set_categories(cats)

X_train = train_split[all_features].copy()
X_val = val_split[all_features].copy()
X_test = test[all_features].copy()

# Fill NaN in numeric features (fit on train only)
for col in all_features:
    if col not in categorical_features:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# ============================================================
# 8. TRAIN LIGHTGBM WITH EARLY STOPPING
# ============================================================
model_params = {
    "objective": "multiclass",
    "num_class": n_categories,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "min_child_weight": 0.01,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMClassifier(**model_params)
categorical_feature_list = [c for c in categorical_features if c in all_features]

model.fit(
    X_train,
    train_labels,
    eval_set=[(X_val, val_labels)],
    eval_metric="multi_logloss",
    categorical_feature=categorical_feature_list,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

best_iteration = model.best_iteration_
print(f"Training complete. Best iteration: {best_iteration}")

# ============================================================
# 9. VALIDATION - Compute log_loss (official metric)
# ============================================================
val_probs = model.predict_proba(X_val)
val_probs_clipped = np.clip(val_probs, 1e-15, 1.0)
val_probs_clipped = val_probs_clipped / val_probs_clipped.sum(axis=1, keepdims=True)

val_score = log_loss(val_labels, val_probs_clipped, labels=list(range(n_categories)))
print(f"Validation log_loss: {val_score:.6f}")

# ============================================================
# 10. TEST INFERENCE & SUBMISSION
# ============================================================
test_probs = model.predict_proba(X_test)
test_probs_clipped = np.clip(test_probs, 1e-15, 1.0)
test_probs_clipped = test_probs_clipped / test_probs_clipped.sum(axis=1, keepdims=True)

os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame({"id": range(len(test))})
predicted_indices = test_probs_clipped.argmax(axis=1)
submission["Category"] = [categories[i] for i in predicted_indices]

submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

# ============================================================
# 11. FINAL OUTPUT
# ============================================================
print(f"\n--- SUMMARY ---")
print(f"Model: LightGBM multiclass (best_iter={best_iteration})")
print(f"Validation log_loss: {val_score:.6f}")
print(f"Submission rows: {len(submission)}")

score = val_score
print(f"Final Validation Score: {score}")