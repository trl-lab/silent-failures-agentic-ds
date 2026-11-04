import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

target_col = "price_doc"
y_train_orig = train_df[target_col].values
y_train_log = np.log1p(y_train_orig)

train_df = train_df.drop(columns=[target_col], errors="ignore")
df_all = pd.concat([train_df, test_df], ignore_index=True)

# 2. Datetime Features
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
df_all["year"] = df_all["timestamp"].dt.year
df_all["month"] = df_all["timestamp"].dt.month
df_all["day"] = df_all["timestamp"].dt.day
df_all["dayofweek"] = df_all["timestamp"].dt.dayofweek
df_all["dayofyear"] = df_all["timestamp"].dt.dayofyear
df_all["year_month"] = df_all["year"] * 100 + df_all["month"]
df_all["quarter"] = df_all["timestamp"].dt.quarter
df_all["days_since_start"] = (
    df_all["timestamp"] - pd.to_datetime("2011-01-01")
).dt.days
df_all = df_all.drop(columns=["timestamp"])

# 3. Clean Physical Property Anomalies
df_all.loc[df_all["full_sq"] == 0, "full_sq"] = np.nan
df_all.loc[df_all["full_sq"] > 1000, "full_sq"] = np.nan

mask_life = df_all["life_sq"] > df_all["full_sq"]
df_all.loc[mask_life, "life_sq"] = np.nan

mask_build = (df_all["build_year"] < 1800) | (df_all["build_year"] > 2020)
df_all.loc[mask_build, "build_year"] = np.nan

mask_floor = df_all["floor"] > df_all["max_floor"]
df_all.loc[mask_floor, "max_floor"] = np.nan

# 4. Expanded Domain Feature Engineering
df_all["life_ratio"] = df_all["life_sq"] / (df_all["full_sq"] + 1e-5)
df_all["kitch_ratio"] = df_all["kitch_sq"] / (df_all["full_sq"] + 1e-5)
df_all["floor_ratio"] = df_all["floor"] / (df_all["max_floor"] + 1e-5)
df_all["sq_per_room"] = df_all["full_sq"] / (df_all["num_room"] + 1e-5)
df_all["life_sq_per_room"] = df_all["life_sq"] / (df_all["num_room"] + 1e-5)
df_all["extra_sq"] = (
    df_all["full_sq"]
    - df_all["life_sq"].clip(lower=0)
    - df_all["kitch_sq"].clip(lower=0)
)
df_all["age"] = df_all["year"] - df_all["build_year"]
df_all["is_top_floor"] = (
    (df_all["floor"] == df_all["max_floor"]) & (df_all["max_floor"] > 1)
).astype(int)
df_all["is_first_floor"] = (df_all["floor"] == 1).astype(int)
df_all["floor_from_top"] = df_all["max_floor"] - df_all["floor"]

# 5. Multi-scale Spatial & Temporal Aggregations
agg_cols = ["full_sq", "life_sq", "kitch_sq", "num_room", "age"]
if "sub_area" in df_all.columns:
    for col in agg_cols:
        if col in df_all.columns:
            grp_mean = df_all.groupby("sub_area")[col].transform("mean")
            grp_std = df_all.groupby("sub_area")[col].transform("std")
            df_all[f"sub_area_{col}_mean"] = grp_mean
            df_all[f"sub_area_{col}_std"] = grp_std
            df_all[f"sub_area_{col}_ratio"] = df_all[col] / (grp_mean + 1e-5)

if "year_month" in df_all.columns:
    for col in ["full_sq", "life_sq"]:
        if col in df_all.columns:
            df_all[f"ym_{col}_mean"] = df_all.groupby("year_month")[col].transform(
                "mean"
            )

# Handle Categoricals
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("missing")

# Separate train and test sets
X_train = df_all.iloc[: len(train_df)].copy()
X_test = df_all.iloc[len(train_df) :].copy()

feature_cols = [c for c in df_all.columns if c != "id"]

# 6. 5-Fold Cross Validation with Tuned CatBoost Regressor
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr = X_train.iloc[train_idx][feature_cols], y_train_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx][feature_cols], y_train_log[val_idx]

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    val_preds = model.predict(X_va)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test[feature_cols]) / kf.n_splits

# 7. Validation Score Calculation
oof_rmsle = np.sqrt(np.mean((oof_preds - y_train_log) ** 2))
print(f"5-Fold CV RMSLE: {oof_rmsle:.5f}")

# 8. Save Final Predictions
final_test_preds = np.expm1(test_preds)
final_test_preds = np.clip(final_test_preds, a_min=0, a_max=None)

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "price_doc": final_test_preds})
sub.to_csv("./working/submission.csv", index=False)
