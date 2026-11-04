import os
import numpy as np
import pandas as pd
import lightgbm as lgb

os.makedirs("./submission", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_parquet(
    "./input/train.parquet",
    columns=["building_id", "meter", "meter_reading", "timestamp"],
)
test_df = pd.read_parquet(
    "./input/test.parquet",
    columns=["building_id", "meter", "timestamp"],
)

train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

# -----------------------------------------------------------------------------
# 2. Chronological validation split (fit/transform/split first to avoid leakage)
# -----------------------------------------------------------------------------
print("Creating chronological split...")
unique_times = sorted(train_df["timestamp"].unique())
split_time = unique_times[int(len(unique_times) * 0.9)]

train_raw = train_df[train_df["timestamp"] < split_time].copy()
val_raw = train_df[train_df["timestamp"] >= split_time].copy()
del train_df

train_raw = train_raw.dropna(subset=["meter_reading"])
val_raw = val_raw.dropna(subset=["meter_reading"])

# -----------------------------------------------------------------------------
# 3. Feature engineering
# -----------------------------------------------------------------------------
print("Engineering features...")


def make_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Compact categorical encoding of the same building_id x meter interaction.
    # meter values are small integers, so this is a unique categorical key.
    df["building_meter"] = (df["building_id"] * 100 + df["meter"]).astype("int32")

    return df


train_feat = make_features(train_raw)
val_feat = make_features(val_raw)
test_feat = make_features(test_df)
del train_raw, val_raw, test_df

# Memory optimization: downcast integer columns.
for df in [train_feat, val_feat, test_feat]:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

# Target transformation: log1p aligns optimization with RMSLE.
train_feat["target_log"] = np.log1p(train_feat["meter_reading"])
val_feat["target_log"] = np.log1p(val_feat["meter_reading"])
test_feat["target_log"] = 0.0  # structural placeholder only

# Drop columns not needed for model input.
train_feat.drop(columns=["timestamp", "meter_reading"], inplace=True)
val_feat.drop(columns=["timestamp"], inplace=True)
test_feat.drop(columns=["timestamp"], inplace=True)

feature_cols = [c for c in train_feat.columns if c != "target_log"]
cat_cols = ["building_id", "meter", "building_meter"]

# Sanity checks.
for _df, _name in [(train_feat, "train"), (val_feat, "val"), (test_feat, "test")]:
    for _c in feature_cols + ["target_log"]:
        assert _c in _df.columns, f"{_name} missing column {_c}"
assert "meter_reading" in val_feat.columns, "val_feat missing meter_reading for metric"


# -----------------------------------------------------------------------------
# 4. Official metric: RMSLE
#    RMSLE = sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
# -----------------------------------------------------------------------------
def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


# -----------------------------------------------------------------------------
# 5. Seed-averaged LightGBM training
# -----------------------------------------------------------------------------
SEEDS = [42, 7, 123]
models = []
val_pred_logs = []

for seed in SEEDS:
    print(f"Training seed {seed} ...")
    params = {
        "objective": "huber",
        "alpha": 0.9,
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X=train_feat[feature_cols],
        y=train_feat["target_log"],
        eval_set=[(val_feat[feature_cols], val_feat["target_log"])],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    models.append(model)
    val_pred_logs.append(model.predict(val_feat[feature_cols]))
    print(f"  seed {seed} best_iteration={model.best_iteration_}")

# -----------------------------------------------------------------------------
# 6. Validation inference using identical post-processing as test
# -----------------------------------------------------------------------------
val_pred_log = np.mean(val_pred_logs, axis=0)
val_pred = np.maximum(np.expm1(val_pred_log), 0.0)
val_score = rmsle(val_feat["meter_reading"], val_pred)
print(f"Validation RMSLE (chronological hold-out): {val_score:.6f}")

# -----------------------------------------------------------------------------
# 7. Test inference using identical post-processing as validation
# -----------------------------------------------------------------------------
test_pred_logs = [m.predict(test_feat[feature_cols]) for m in models]
test_pred_log = np.mean(test_pred_logs, axis=0)
test_pred = np.maximum(np.expm1(test_pred_log), 0.0)

# -----------------------------------------------------------------------------
# 8. Save submission
# -----------------------------------------------------------------------------
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "meter_reading": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")

print(f"Final Validation Score: {val_score}")