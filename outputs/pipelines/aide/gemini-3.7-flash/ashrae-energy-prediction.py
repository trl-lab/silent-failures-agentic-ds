import gc
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# 1. Load Data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


# 2. Feature Engineering Function
def preprocess_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day"] = df["timestamp"].dt.day.astype(np.int8)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["month"] = df["timestamp"].dt.month.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # Cyclical temporal features
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12).astype(np.float32)

    # Optimize numeric datatypes
    df["building_id"] = df["building_id"].astype(np.int16)
    df["meter"] = df["meter"].astype(np.int8)
    df["building_meter"] = (df["building_id"] * 10 + df["meter"]).astype(np.int32)

    return df


print("Preprocessing features...")
train_df = preprocess_features(train_df)
test_df = preprocess_features(test_df)

base_feature_cols = [
    "building_id",
    "meter",
    "building_meter",
    "hour",
    "day",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
]

categorical_features = [
    "building_id",
    "meter",
    "building_meter",
    "dayofweek",
    "month",
    "is_weekend",
]

# Log1p transform of target
y_train = np.log1p(np.clip(train_df["meter_reading"].values, 0, None))
X_train_base = train_df[base_feature_cols].copy()
X_test_base = test_df[base_feature_cols].copy()

del train_df
gc.collect()

# 3. Model Training with 5-Fold Cross Validation and Multi-Level Target Encoding
print("Starting 5-Fold Cross Validation with Schedule-Aware Target Statistics...")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_base), dtype=np.float32)
test_preds = np.zeros(len(X_test_base), dtype=np.float32)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def add_hierarchical_target_encodings(df_tr, y_tr, df_va, df_te):
    stats_df = df_tr[["building_id", "meter", "is_weekend", "hour", "month"]].copy()
    stats_df["target"] = y_tr

    # Group 1: building_id + meter
    bm_stats = (
        stats_df.groupby(["building_id", "meter"])["target"]
        .agg(
            bm_mean="mean",
            bm_std="std",
            bm_median="median",
            bm_min="min",
            bm_max="max",
        )
        .reset_index()
    )

    # Group 2: building_id + meter + is_weekend + hour (schedule-aware diurnal pattern)
    bmwh_stats = (
        stats_df.groupby(["building_id", "meter", "is_weekend", "hour"])["target"]
        .agg(bmwh_mean="mean", bmwh_std="std")
        .reset_index()
    )

    # Group 3: building_id + meter + month (seasonal profile)
    bmm_stats = (
        stats_df.groupby(["building_id", "meter", "month"])["target"]
        .agg(bmm_mean="mean")
        .reset_index()
    )

    global_mean = y_tr.mean()

    def merge_and_fill(df):
        merged = df.merge(bm_stats, on=["building_id", "meter"], how="left")
        merged = merged.merge(
            bmwh_stats,
            on=["building_id", "meter", "is_weekend", "hour"],
            how="left",
        )
        merged = merged.merge(
            bmm_stats, on=["building_id", "meter", "month"], how="left"
        )

        merged["bm_mean"] = merged["bm_mean"].fillna(global_mean).astype(np.float32)
        merged["bm_median"] = merged["bm_median"].fillna(global_mean).astype(np.float32)
        merged["bm_std"] = merged["bm_std"].fillna(0.0).astype(np.float32)
        merged["bm_min"] = merged["bm_min"].fillna(0.0).astype(np.float32)
        merged["bm_max"] = merged["bm_max"].fillna(global_mean).astype(np.float32)
        merged["bmwh_mean"] = (
            merged["bmwh_mean"].fillna(merged["bm_mean"]).astype(np.float32)
        )
        merged["bmwh_std"] = merged["bmwh_std"].fillna(0.0).astype(np.float32)
        merged["bmm_mean"] = (
            merged["bmm_mean"].fillna(merged["bm_mean"]).astype(np.float32)
        )
        return merged

    return merge_and_fill(df_tr), merge_and_fill(df_va), merge_and_fill(df_te)


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_base)):
    print(f"--- Fold {fold + 1} ---")
    X_tr_fold = X_train_base.iloc[train_idx].copy()
    y_tr_fold = y_train[train_idx]
    X_va_fold = X_train_base.iloc[val_idx].copy()
    y_va_fold = y_train[val_idx]
    X_te_fold = X_test_base.copy()

    X_tr_enc, X_va_enc, X_te_enc = add_hierarchical_target_encodings(
        X_tr_fold, y_tr_fold, X_va_fold, X_te_fold
    )

    for col in categorical_features:
        X_tr_enc[col] = X_tr_enc[col].astype("category")
        X_va_enc[col] = X_va_enc[col].astype("category")
        X_te_enc[col] = X_te_enc[col].astype("category")

    trn_data = lgb.Dataset(
        X_tr_enc, label=y_tr_fold, categorical_feature=categorical_features
    )
    val_data = lgb.Dataset(
        X_va_enc,
        label=y_va_fold,
        categorical_feature=categorical_features,
        reference=trn_data,
    )

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict(X_va_enc, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    fold_rmsle = root_mean_squared_error(y_va_fold, val_pred)
    print(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}")

    test_preds += model.predict(X_te_enc, num_iteration=model.best_iteration) / n_splits

    del (
        trn_data,
        val_data,
        X_tr_fold,
        X_va_fold,
        X_te_fold,
        X_tr_enc,
        X_va_enc,
        X_te_enc,
        model,
    )
    gc.collect()

# 4. Overall Out-Of-Fold Evaluation
cv_rmsle = root_mean_squared_error(y_train, oof_preds)
print(f"\n==========================================")
print(f"Overall Out-of-Fold RMSLE: {cv_rmsle:.5f}")
print(f"==========================================")

# 5. Generate Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=np.int64),
        "meter_reading": np.expm1(np.clip(test_preds, 0, None)),
    }
)

submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with {len(submission)} rows."
)
