import os
import numpy as np
import pandas as pd
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

TARGET = "price_doc"
ID_COL = "id"
TIME_COL = "timestamp"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.drop(columns=[ID_COL], errors="ignore").copy()
test = test.drop(columns=[ID_COL], errors="ignore").copy()

y_full = train.pop(TARGET)

if TIME_COL in train.columns:
    train[TIME_COL] = pd.to_datetime(train[TIME_COL], errors="coerce")
    if TIME_COL in test.columns:
        test[TIME_COL] = pd.to_datetime(test[TIME_COL], errors="coerce")
    else:
        test[TIME_COL] = pd.NaT
    time_min = train[TIME_COL].min()
else:
    time_min = None

# Identify categorical columns.
cat_cols = [
    c
    for c in train.columns
    if train[c].dtype == object or str(train[c].dtype) == "category"
]

known_categorical = [
    "material",
    "product_type",
    "state",
    "sub_area",
    "area_m",
    "raion_state",
    "raion_material",
    "raion_apartment_type",
    "raion_product_type",
    "raion_area_m",
    "raion_sub_area",
    "raion_raion_state",
    "raion_raion_material",
    "raion_raion_apartment_type",
    "raion_raion_product_type",
    "raion_raion_area_m",
    "raion_raion_sub_area",
]

for c in known_categorical:
    if (
        c in train.columns
        and c not in cat_cols
        and train[c].nunique(dropna=True) < 1000
    ):
        cat_cols.append(c)

continuous_names = {
    "full_sq",
    "life_sq",
    "floor",
    "max_floor",
    "build_year",
    "num_room",
    "kitch_sq",
    "raion_popul",
    "raion_build_cnt",
    "raion_build_years",
    "raion_life_sq",
    "raion_kitch_sq",
    "raion_full_sq",
    "raion_num_room",
    "raion_build_year",
    "raion_floor",
    "raion_max_floor",
    "year",
    "month",
    "day",
    "dayofweek",
    "days_since_min",
}

for c in train.columns:
    if c in cat_cols or c == TIME_COL or c in continuous_names:
        continue
    if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True) <= 50:
        cat_cols.append(c)

cat_cols = list(dict.fromkeys([c for c in cat_cols if c in train.columns]))

num_cols = [
    c
    for c in train.columns
    if c not in cat_cols and c != TIME_COL and pd.api.types.is_numeric_dtype(train[c])
]

# Convert object columns that are mostly numeric to numeric.
for c in train.columns:
    if c in cat_cols or c == TIME_COL:
        continue
    if not pd.api.types.is_numeric_dtype(train[c]):
        converted = pd.to_numeric(train[c], errors="coerce")
        if converted.notna().sum() > 0.8 * train[c].notna().sum():
            train[c] = converted
            if c in test.columns:
                test[c] = pd.to_numeric(test[c], errors="coerce")
            if c not in num_cols:
                num_cols.append(c)
        else:
            if c not in cat_cols:
                cat_cols.append(c)


# Add time features.
def add_time_features(df):
    if TIME_COL in df.columns:
        df["year"] = df[TIME_COL].dt.year
        df["month"] = df[TIME_COL].dt.month
        df["day"] = df[TIME_COL].dt.day
        df["dayofweek"] = df[TIME_COL].dt.dayofweek
        if time_min is not None:
            df["days_since_min"] = (df[TIME_COL] - time_min).dt.days
    return df


train = add_time_features(train)
test = add_time_features(test)


# Basic feature engineering.
def safe_div(a, b):
    return a / b.replace(0, np.nan)


def add_features(df):
    if "full_sq" in df.columns and "life_sq" in df.columns:
        df["life_sq_ratio"] = safe_div(df["life_sq"], df["full_sq"])
    if "full_sq" in df.columns and "kitch_sq" in df.columns:
        df["kitch_sq_ratio"] = safe_div(df["kitch_sq"], df["full_sq"])
    if "floor" in df.columns and "max_floor" in df.columns:
        df["floor_ratio"] = safe_div(df["floor"], df["max_floor"])
    if "full_sq" in df.columns and "num_room" in df.columns:
        df["full_sq_per_room"] = safe_div(df["full_sq"], df["num_room"])
    if "life_sq" in df.columns and "num_room" in df.columns:
        df["life_sq_per_room"] = safe_div(df["life_sq"], df["num_room"])
    if "kitch_sq" in df.columns and "num_room" in df.columns:
        df["kitch_sq_per_room"] = safe_div(df["kitch_sq"], df["num_room"])
    if "build_year" in df.columns and "year" in df.columns:
        df["building_age"] = df["year"] - df["build_year"]
    return df


train = add_features(train)
test = add_features(test)

# Align columns between train and test.
all_cols = list(dict.fromkeys(list(train.columns) + list(test.columns)))
for c in all_cols:
    if c not in train.columns:
        train[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

# Finalize categorical and numeric feature lists.
cat_cols = [c for c in cat_cols if c in train.columns]
num_cols = [c for c in num_cols if c in train.columns]

# Add engineered numeric columns.
for c in [
    "life_sq_ratio",
    "kitch_sq_ratio",
    "floor_ratio",
    "full_sq_per_room",
    "life_sq_per_room",
    "kitch_sq_per_room",
    "building_age",
    "year",
    "month",
    "day",
    "dayofweek",
    "days_since_min",
]:
    if c in train.columns and c not in num_cols and c not in cat_cols:
        num_cols.append(c)

# Ensure all feature columns exist in both frames.
feature_cols = list(dict.fromkeys(num_cols + cat_cols))
for c in feature_cols:
    if c not in train.columns:
        train[c] = np.nan
    if c not in test.columns:
        test[c] = np.nan

# Clean numeric columns.
for c in num_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c] = pd.to_numeric(test[c], errors="coerce")
    # Replace infinite values with NaN.
    train[c] = train[c].replace([np.inf, -np.inf], np.nan)
    test[c] = test[c].replace([np.inf, -np.inf], np.nan)

# Clean categorical columns and align categories using train only.
for c in cat_cols:
    train[c] = train[c].astype("object")
    test[c] = test[c].astype("object")
    train[c] = train[c].fillna("unknown")
    test[c] = test[c].fillna("unknown")

    train_cats = set(train[c].unique())
    test[c] = test[c].where(test[c].isin(train_cats), "unknown")

    combined_cats = list(train[c].unique())
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")
    test[c] = test[c].cat.set_categories(combined_cats)

# Time-based validation split.
if TIME_COL in train.columns:
    train_sorted = train.sort_values(TIME_COL, kind="mergesort")
    val_size = max(1, int(len(train_sorted) * 0.10))
    val_idx = train_sorted.index[-val_size:]
    train_idx = train_sorted.index[:-val_size]
else:
    # Fallback to random holdout if timestamp is unavailable.
    train_shuffled = train.sample(frac=1.0, random_state=42)
    val_size = max(1, int(len(train_shuffled) * 0.10))
    val_idx = train_shuffled.index[-val_size:]
    train_idx = train_shuffled.index[:-val_size]

X_train = train.loc[train_idx, feature_cols]
y_train = np.log1p(y_full.loc[train_idx])

X_val = train.loc[val_idx, feature_cols]
y_val = np.log1p(y_full.loc[val_idx])

X_full = train[feature_cols]
y_full_log = np.log1p(y_full)

X_test = test[feature_cols]

base_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
}

seeds = [42, 123, 456]

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data = lgb.Dataset(
    X_val, label=y_val, categorical_feature=cat_cols, reference=train_data
)

val_pred_log = np.zeros(len(X_val), dtype=np.float64)
best_iters = []

for seed in seeds:
    params = base_params.copy()
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    bi = getattr(model, "best_iteration", 0)
    if bi is None or bi <= 0:
        bi = 5000
    bi = int(bi)
    best_iters.append(bi)

    val_pred_log += model.predict(X_val, num_iteration=bi)

val_pred_log /= len(seeds)
rmsle = float(np.sqrt(np.mean((val_pred_log - y_val) ** 2)))
print(f"Validation RMSLE: {rmsle:.6f}")

final_rounds = int(np.median(best_iters)) if best_iters else 1000

full_data = lgb.Dataset(X_full, label=y_full_log, categorical_feature=cat_cols)
test_pred_log = np.zeros(len(X_test), dtype=np.float64)

for seed in seeds:
    params = base_params.copy()
    params["seed"] = seed
    params["bagging_seed"] = seed
    params["feature_fraction_seed"] = seed

    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred_log += final_model.predict(X_test)

test_pred_log /= len(seeds)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "price_doc": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv")
