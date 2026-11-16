import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

# 1. Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


# 2. Extract temporal and interaction features
def prepare_features(df):
    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["building_id"] = df["building_id"].astype(np.int16)
    df["meter"] = df["meter"].astype(np.int8)
    df["building_meter"] = (
        df["building_id"].astype(np.int32) * 10 + df["meter"].astype(np.int32)
    ).astype(np.int16)
    return df


train = prepare_features(train)
test = prepare_features(test)

train["target_log"] = np.log1p(np.maximum(0, train["meter_reading"].values))

cat_cols = ["building_id", "meter", "building_meter"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

# 3. Setup 5-Fold CV and out-of-fold prediction setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train), dtype=np.float32)
test_preds = np.zeros(len(test), dtype=np.float32)

feature_cols = [
    "building_id",
    "meter",
    "building_meter",
    "hour",
    "day",
    "month",
    "dayofweek",
    "is_weekend",
    "bm_h_mean",
    "bm_h_std",
    "bm_dow_mean",
    "bm_wh_mean",
    "bm_wh_std",
    "bm_m_mean",
    "bm_m_std",
    "bm_mean",
    "bm_std",
]

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    tr_df = train.iloc[tr_idx].copy()
    val_df = train.iloc[val_idx].copy()
    test_fold = test.copy()

    # Target encoding mean and std statistics computed strictly on the training fold
    bm_h = (
        tr_df.groupby(["building_id", "meter", "hour"], observed=True)["target_log"]
        .agg(bm_h_mean="mean", bm_h_std="std")
        .reset_index()
    )
    bm_dow = (
        tr_df.groupby(["building_id", "meter", "dayofweek"], observed=True)[
            "target_log"
        ]
        .mean()
        .rename("bm_dow_mean")
        .reset_index()
    )
    bm_wh = (
        tr_df.groupby(["building_id", "meter", "is_weekend", "hour"], observed=True)[
            "target_log"
        ]
        .agg(bm_wh_mean="mean", bm_wh_std="std")
        .reset_index()
    )
    bm_m = (
        tr_df.groupby(["building_id", "meter", "month"], observed=True)["target_log"]
        .agg(bm_m_mean="mean", bm_m_std="std")
        .reset_index()
    )
    bm_stats = (
        tr_df.groupby(["building_id", "meter"], observed=True)["target_log"]
        .agg(bm_mean="mean", bm_std="std")
        .reset_index()
    )

    g_mean = float(tr_df["target_log"].mean())
    g_std = float(tr_df["target_log"].std())

    def merge_encodings(df):
        df = df.merge(bm_h, on=["building_id", "meter", "hour"], how="left")
        df = df.merge(bm_dow, on=["building_id", "meter", "dayofweek"], how="left")
        df = df.merge(
            bm_wh, on=["building_id", "meter", "is_weekend", "hour"], how="left"
        )
        df = df.merge(bm_m, on=["building_id", "meter", "month"], how="left")
        df = df.merge(bm_stats, on=["building_id", "meter"], how="left")
        df["bm_mean"] = df["bm_mean"].fillna(g_mean).astype(np.float32)
        df["bm_std"] = df["bm_std"].fillna(g_std).astype(np.float32)
        df["bm_h_mean"] = df["bm_h_mean"].fillna(df["bm_mean"]).astype(np.float32)
        df["bm_h_std"] = df["bm_h_std"].fillna(df["bm_std"]).astype(np.float32)
        df["bm_dow_mean"] = df["bm_dow_mean"].fillna(df["bm_mean"]).astype(np.float32)
        df["bm_wh_mean"] = df["bm_wh_mean"].fillna(df["bm_h_mean"]).astype(np.float32)
        df["bm_wh_std"] = df["bm_wh_std"].fillna(df["bm_h_std"]).astype(np.float32)
        df["bm_m_mean"] = df["bm_m_mean"].fillna(df["bm_mean"]).astype(np.float32)
        df["bm_m_std"] = df["bm_m_std"].fillna(df["bm_std"]).astype(np.float32)
        return df

    tr_df = merge_encodings(tr_df)
    val_df = merge_encodings(val_df)
    test_fold = merge_encodings(test_fold)

    lgb_tr = lgb.Dataset(tr_df[feature_cols], label=tr_df["target_log"])
    lgb_val = lgb.Dataset(
        val_df[feature_cols], label=val_df["target_log"], reference=lgb_tr
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "seed": 42 + fold,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        lgb_tr,
        num_boost_round=1000,
        valid_sets=[lgb_tr, lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(val_df[feature_cols])
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test_fold[feature_cols]) / kf.n_splits

    del tr_df, val_df, test_fold, lgb_tr, lgb_val, model
    gc.collect()

# 4. Evaluation metric computation
oof_rmsle = np.sqrt(
    np.mean((np.maximum(0, oof_preds) - train["target_log"].values) ** 2)
)
print(f"Validation RMSLE: {oof_rmsle:.4f}")

# 5. Save submission file
sub = pd.DataFrame(
    {"id": np.arange(len(test)), "meter_reading": np.expm1(np.maximum(0, test_preds))}
)
sub.to_csv("./working/submission.csv", index=False)
