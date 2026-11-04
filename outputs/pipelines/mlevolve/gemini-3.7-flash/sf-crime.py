import os
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 0. Global Setup & Seed Configuration
# -------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------
# 1. Load Raw Datasets & Stratified Split
# -------------------------------------------------------------
print("Loading raw datasets from ./input/...")
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

if "Id" in test_raw.columns:
    test_id = test_raw["Id"].values
elif "id" in test_raw.columns:
    test_id = test_raw["id"].values
else:
    test_id = np.arange(len(test_raw))

# Target label encoding
target_encoder = LabelEncoder()
y_train_full = target_encoder.fit_transform(train_raw["Category"].values)
classes = target_encoder.classes_.tolist()
num_classes = len(classes)
print(f"Target classes count: {num_classes}")

# Stratified Train-Validation Split (85% Train, 15% Hold-out Validation)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
train_idx, val_idx = next(sss.split(train_raw, y_train_full))

df_train = train_raw.iloc[train_idx].copy().reset_index(drop=True)
y_train = y_train_full[train_idx]

df_val = train_raw.iloc[val_idx].copy().reset_index(drop=True)
y_val = y_train_full[val_idx]

df_test = test_raw.copy().reset_index(drop=True)

print(
    f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(df_test)}"
)

# -------------------------------------------------------------
# 2. Spatial Coordinate Outlier Cleaning (Fit strictly on Train)
# -------------------------------------------------------------
valid_mask = (df_train["Y"] < 80.0) & (df_train["X"] < -121.0)
district_coords = (
    df_train[valid_mask].groupby("PdDistrict")[["X", "Y"]].median().to_dict("index")
)
global_median_x = df_train.loc[valid_mask, "X"].median()
global_median_y = df_train.loc[valid_mask, "Y"].median()


def clean_coordinates(df):
    outlier_mask = (df["Y"] >= 80.0) | (df["X"] >= -121.0)
    for district, coords in district_coords.items():
        mask = outlier_mask & (df["PdDistrict"] == district)
        df.loc[mask, "X"] = coords["X"]
        df.loc[mask, "Y"] = coords["Y"]
    remaining = (df["Y"] >= 80.0) | (df["X"] >= -121.0)
    df.loc[remaining, "X"] = global_median_x
    df.loc[remaining, "Y"] = global_median_y
    return df


df_train = clean_coordinates(df_train)
df_val = clean_coordinates(df_val)
df_test = clean_coordinates(df_test)

# -------------------------------------------------------------
# 3. Address & Street Parsing
# -------------------------------------------------------------
common_street_types = [
    "ST",
    "AV",
    "WY",
    "BL",
    "DR",
    "RD",
    "CT",
    "LN",
    "PL",
    "PZ",
    "TR",
    "WAY",
    "AL",
    "HWY",
    "HY",
]


def parse_address(df):
    is_intersection = df["Address"].str.contains("/", regex=False).astype(np.int32)
    is_block = (
        df["Address"].str.contains("Block", case=False, regex=False).astype(np.int32)
    )
    block_num = (
        df["Address"]
        .str.extract(r"(\d+)\s+Block", expand=False)
        .fillna("-1")
        .astype(np.float32)
    )
    street_type = (
        df["Address"]
        .str.extract(r"\b(" + "|".join(common_street_types) + r")\b", expand=False)
        .fillna("OTHER")
    )
    primary_street = df["Address"].str.replace(
        r"^\d+\s+Block\s+of\s+", "", case=False, regex=True
    )
    primary_street = primary_street.str.split("/").str[0].str.strip()
    return is_intersection, is_block, block_num, street_type, primary_street


for df in [df_train, df_val, df_test]:
    is_inter, is_blk, blk_no, st_type, prim_st = parse_address(df)
    df["is_intersection"] = is_inter
    df["is_block"] = is_blk
    df["block_num"] = blk_no
    df["street_type"] = st_type
    df["primary_street"] = prim_st

# -------------------------------------------------------------
# 4. Frequency Encodings (Fit strictly on Train)
# -------------------------------------------------------------
addr_counts = df_train["Address"].value_counts()
street_counts = df_train["primary_street"].value_counts()
district_counts = df_train["PdDistrict"].value_counts()


def apply_frequency_encodings(df):
    df["address_freq"] = df["Address"].map(addr_counts).fillna(1).astype(np.float32)
    df["street_freq"] = (
        df["primary_street"].map(street_counts).fillna(1).astype(np.float32)
    )
    df["district_freq"] = (
        df["PdDistrict"].map(district_counts).fillna(1).astype(np.float32)
    )
    return df


df_train = apply_frequency_encodings(df_train)
df_val = apply_frequency_encodings(df_val)
df_test = apply_frequency_encodings(df_test)

# -------------------------------------------------------------
# 5. Spatiotemporal & Spatial Transformations
# -------------------------------------------------------------
SF_CENTER_X, SF_CENTER_Y = -122.4194, 37.7749
FINANCIAL_X, FINANCIAL_Y = -122.4000, 37.7946
TENDERLOIN_X, TENDERLOIN_Y = -122.4140, 37.7847
MISSION_X, MISSION_Y = -122.4190, 37.7599
SFO_DIR_X, SFO_DIR_Y = -122.3800, 37.7100
GGP_X, GGP_Y = -122.4835, 37.7694

min_date = pd.to_datetime(df_train["Dates"]).min()


def engineer_features(df):
    dates = pd.to_datetime(df["Dates"])

    # Temporal features
    df["year"] = dates.dt.year.astype(np.int32)
    df["month"] = dates.dt.month.astype(np.int32)
    df["day"] = dates.dt.day.astype(np.int32)
    df["hour"] = dates.dt.hour.astype(np.int32)
    df["minute"] = dates.dt.minute.astype(np.int32)
    df["dayofweek"] = dates.dt.dayofweek.astype(np.int32)
    df["dayofyear"] = dates.dt.dayofyear.astype(np.int32)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int32)

    df["time_trend_days"] = ((dates - min_date).dt.total_seconds() / 86400.0).astype(
        np.float32
    )
    df["fractional_year"] = (df["year"] + (df["dayofyear"] - 1) / 365.25).astype(
        np.float32
    )

    # Cyclical encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25).astype(
        np.float32
    )
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25).astype(
        np.float32
    )

    # Time of day segment
    df["time_of_day"] = pd.cut(
        df["hour"], bins=[-1, 5, 11, 16, 21, 24], labels=[0, 1, 2, 3, 0], ordered=False
    ).astype(np.int32)

    # Spatial Rotations
    rad30, rad45, rad60 = np.radians(30), np.radians(45), np.radians(60)
    df["rot30_X"] = (df["X"] * np.cos(rad30) - df["Y"] * np.sin(rad30)).astype(
        np.float32
    )
    df["rot30_Y"] = (df["X"] * np.sin(rad30) + df["Y"] * np.cos(rad30)).astype(
        np.float32
    )
    df["rot45_X"] = (df["X"] * np.cos(rad45) - df["Y"] * np.sin(rad45)).astype(
        np.float32
    )
    df["rot45_Y"] = (df["X"] * np.sin(rad45) + df["Y"] * np.cos(rad45)).astype(
        np.float32
    )
    df["rot60_X"] = (df["X"] * np.cos(rad60) - df["Y"] * np.sin(rad60)).astype(
        np.float32
    )
    df["rot60_Y"] = (df["X"] * np.sin(rad60) + df["Y"] * np.cos(rad60)).astype(
        np.float32
    )

    # Polar coordinates and distances to landmarks
    dx_center = df["X"] - SF_CENTER_X
    dy_center = df["Y"] - SF_CENTER_Y
    df["dist_center"] = np.sqrt(dx_center**2 + dy_center**2).astype(np.float32)
    df["angle_center"] = np.arctan2(dy_center, dx_center).astype(np.float32)
    df["manhattan_center"] = (np.abs(dx_center) + np.abs(dy_center)).astype(np.float32)

    df["dist_financial"] = np.sqrt(
        (df["X"] - FINANCIAL_X) ** 2 + (df["Y"] - FINANCIAL_Y) ** 2
    ).astype(np.float32)
    df["dist_tenderloin"] = np.sqrt(
        (df["X"] - TENDERLOIN_X) ** 2 + (df["Y"] - TENDERLOIN_Y) ** 2
    ).astype(np.float32)
    df["dist_mission"] = np.sqrt(
        (df["X"] - MISSION_X) ** 2 + (df["Y"] - MISSION_Y) ** 2
    ).astype(np.float32)
    df["dist_sfo_dir"] = np.sqrt(
        (df["X"] - SFO_DIR_X) ** 2 + (df["Y"] - SFO_DIR_Y) ** 2
    ).astype(np.float32)
    df["dist_ggp"] = np.sqrt((df["X"] - GGP_X) ** 2 + (df["Y"] - GGP_Y) ** 2).astype(
        np.float32
    )

    # Spatial binning
    df["grid_x"] = np.floor((df["X"] - (-122.52)) / 0.002).clip(0, 99).astype(np.int32)
    df["grid_y"] = np.floor((df["Y"] - 37.70) / 0.0015).clip(0, 99).astype(np.int32)
    df["grid_id"] = (df["grid_x"] * 100 + df["grid_y"]).astype(np.int32)

    return df


print("Engineering spatial, temporal, and interaction features...")
df_train = engineer_features(df_train)
df_val = engineer_features(df_val)
df_test = engineer_features(df_test)

# -------------------------------------------------------------
# 6. Spatiotemporal Cross Interaction Counts (from Train)
# -------------------------------------------------------------
dist_hour_counts = df_train.groupby(["PdDistrict", "hour"]).size().to_dict()
grid_hour_counts = df_train.groupby(["grid_id", "hour"]).size().to_dict()


def apply_cross_interactions(df):
    dist_hour_key = list(zip(df["PdDistrict"], df["hour"]))
    grid_hour_key = list(zip(df["grid_id"], df["hour"]))

    df["dist_hour_freq"] = [dist_hour_counts.get(k, 1) for k in dist_hour_key]
    df["dist_hour_freq"] = df["dist_hour_freq"].astype(np.float32)

    df["grid_hour_freq"] = [grid_hour_counts.get(k, 1) for k in grid_hour_key]
    df["grid_hour_freq"] = df["grid_hour_freq"].astype(np.float32)
    return df


df_train = apply_cross_interactions(df_train)
df_val = apply_cross_interactions(df_val)
df_test = apply_cross_interactions(df_test)

# -------------------------------------------------------------
# 7. Categorical Variable Encodings
# -------------------------------------------------------------
cat_cols = ["PdDistrict", "street_type", "primary_street"]

for col in cat_cols:
    unique_train_vals = df_train[col].astype(str).unique().tolist()
    mapping = {val: i for i, val in enumerate(unique_train_vals)}
    unknown_val = len(mapping)

    df_train[col + "_code"] = (
        df_train[col].astype(str).map(mapping).fillna(unknown_val).astype(np.int32)
    )
    df_val[col + "_code"] = (
        df_val[col].astype(str).map(mapping).fillna(unknown_val).astype(np.int32)
    )
    df_test[col + "_code"] = (
        df_test[col].astype(str).map(mapping).fillna(unknown_val).astype(np.int32)
    )

# Select final feature columns
feature_cols = [
    "X",
    "Y",
    "rot30_X",
    "rot30_Y",
    "rot45_X",
    "rot45_Y",
    "rot60_X",
    "rot60_Y",
    "dist_center",
    "angle_center",
    "manhattan_center",
    "dist_financial",
    "dist_tenderloin",
    "dist_mission",
    "dist_sfo_dir",
    "dist_ggp",
    "grid_x",
    "grid_y",
    "grid_id",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "time_trend_days",
    "fractional_year",
    "time_of_day",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    "is_intersection",
    "is_block",
    "block_num",
    "address_freq",
    "street_freq",
    "district_freq",
    "dist_hour_freq",
    "grid_hour_freq",
    "PdDistrict_code",
    "street_type_code",
    "primary_street_code",
]

X_train = df_train[feature_cols]
X_val = df_val[feature_cols]
X_test = df_test[feature_cols]

print(f"Features ready. Total features: {len(feature_cols)}")

# -------------------------------------------------------------
# 8. Model Training: LightGBM Multiclass Log-Loss
# -------------------------------------------------------------
print("Configuring and training LightGBM multiclass classifier...")
lgbm_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=num_classes,
    metric="multi_logloss",
    boosting_type="gbdt",
    n_estimators=600,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=8,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    max_bin=255,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=-1,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=35, verbose=False),
    lgb.log_evaluation(period=0),
]

lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_names=["val"],
    callbacks=callbacks,
)

# -------------------------------------------------------------
# 9. Validation Inference & Evaluation
# -------------------------------------------------------------
print("Evaluating model performance on hold-out validation set...")
val_preds_raw = lgbm_model.predict_proba(X_val)

eps = 1e-15
val_preds_clipped = np.clip(val_preds_raw, eps, 1.0 - eps)
val_preds_prob = val_preds_clipped / val_preds_clipped.sum(axis=1, keepdims=True)

val_score = float(
    log_loss(y_val, val_preds_prob, labels=list(range(num_classes)))
)
print(f"Validation Multi-class Log Loss: {val_score:.5f}")

# -------------------------------------------------------------
# 10. Test Inference & Submission Generation
# -------------------------------------------------------------
print("Generating predictions for test set...")
test_preds_raw = lgbm_model.predict_proba(X_test)
test_preds_clipped = np.clip(test_preds_raw, eps, 1.0 - eps)
test_preds_prob = test_preds_clipped / test_preds_clipped.sum(axis=1, keepdims=True)

test_pred_indices = np.argmax(test_preds_prob, axis=1)
test_pred_categories = target_encoder.inverse_transform(test_pred_indices)

submission_df = pd.DataFrame({"id": test_id, "Category": test_pred_categories})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}."
)

# Required validation score output
print(f"Final Validation Score: {val_score}")
