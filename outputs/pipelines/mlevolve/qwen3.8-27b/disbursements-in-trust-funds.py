import pandas as pd
import numpy as np
import json
import os
import warnings
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
TARGET = "Cash Disbursements (US$)"

id_col = "id" if "id" in train.columns else None
if id_col is None:
    for col in train.columns:
        if "id" in col.lower() or "index" in col.lower():
            id_col = col
            break
    if id_col is None:
        id_col = train.columns[0]

print(f"ID column: {id_col}")
print(f"Target column: {TARGET}")

feature_cols_raw = [c for c in train.columns if c not in [TARGET, id_col]]

numeric_cols = (
    train[feature_cols_raw].select_dtypes(include=[np.number]).columns.tolist()
)
categorical_cols = (
    train[feature_cols_raw]
    .select_dtypes(include=["object", "category"])
    .columns.tolist()
)
for col in numeric_cols:
    if train[col].nunique() < 20:
        categorical_cols.append(col)
        numeric_cols.remove(col)

datetime_cols = (
    train[feature_cols_raw].select_dtypes(include=["datetime64"]).columns.tolist()
)

print(f"Numeric cols: {numeric_cols}")
print(f"Categorical cols: {categorical_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (stratified random)
# ============================================================
temporal_col = None
for col in train.columns:
    if (
        train[col].dtype == "datetime64[ns]"
        or "date" in col.lower()
        or "year" in col.lower()
        or "period" in col.lower()
    ):
        temporal_col = col
        break

# Use stratified random split on target quantiles for representative validation
target_quantiles = pd.qcut(train[TARGET], q=10, duplicates='drop')
train_base, val = train_test_split(
    train, test_size=0.2, random_state=42, stratify=target_quantiles
)
train_base = train_base.reset_index(drop=True)
val = val.reset_index(drop=True)
print(f"Stratified random split: train={len(train_base)}, val={len(val)}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(
    train_df,
    val_df,
    test_df,
    target_col,
    id_col,
    numeric_cols,
    categorical_cols,
    temporal_col,
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Signed log transform target: sign(x) * log1p(|x|)
    train_df["target_signed_log"] = np.sign(train_df[target_col]) * np.log1p(np.abs(train_df[target_col]))
    val_df["target_signed_log"] = np.sign(val_df[target_col]) * np.log1p(np.abs(val_df[target_col]))

    # Magnitude target: log1p(|x|) for the magnitude regressor
    train_df["target_mag_log"] = np.log1p(np.abs(train_df[target_col]))
    val_df["target_mag_log"] = np.log1p(np.abs(val_df[target_col]))

    # Sign class target: 0=negative, 1=zero, 2=positive
    train_df["target_sign_class"] = np.where(
        train_df[target_col] < 0, 0, np.where(train_df[target_col] == 0, 1, 2)
    )
    val_df["target_sign_class"] = np.where(
        val_df[target_col] < 0, 0, np.where(val_df[target_col] == 0, 1, 2)
    )

    # is_negative binary feature
    train_df["is_negative"] = (train_df[target_col] < 0).astype(int)
    val_df["is_negative"] = (val_df[target_col] < 0).astype(int)
    test_df["is_negative"] = 0  # placeholder for test

    # abs_target related feature (use magnitude as a feature proxy)
    train_df["abs_target_log"] = np.log1p(np.abs(train_df[target_col]))
    val_df["abs_target_log"] = np.log1p(np.abs(val_df[target_col]))
    test_df["abs_target_log"] = 0  # placeholder for test

    target_vals = train_df[target_col].dropna()
    print(
        f"Target stats: mean={target_vals.mean():.2f}, median={target_vals.median():.2f}, "
        f"min={target_vals.min():.2f}, max={target_vals.max():.2f}, skew={target_vals.skew():.2f}"
    )
    print(f"Sign distribution: neg={sum(train_df[target_col] < 0)}, zero={sum(train_df[target_col] == 0)}, pos={sum(train_df[target_col] > 0)}")

    # Target encoding + frequency + count
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        global_mean = train_df["target_signed_log"].mean()
        agg = train_df.groupby(col)["target_signed_log"].agg(["mean", "count"])
        smoothing = 10
        smoothed_enc = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
            agg["count"] + smoothing
        )

        train_df[f"{col}_target_enc"] = (
            train_df[col].map(smoothed_enc).fillna(global_mean)
        )
        val_df[f"{col}_target_enc"] = val_df[col].map(smoothed_enc).fillna(global_mean)
        test_df[f"{col}_target_enc"] = (
            test_df[col].map(smoothed_enc).fillna(global_mean)
        )

        freq = train_df[col].value_counts(normalize=True)
        train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
        val_df[f"{col}_freq"] = val_df[col].map(freq).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

        count_enc = train_df[col].value_counts()
        train_df[f"{col}_count"] = train_df[col].map(count_enc).fillna(0)
        val_df[f"{col}_count"] = val_df[col].map(count_enc).fillna(0)
        test_df[f"{col}_count"] = test_df[col].map(count_enc).fillna(0)

    # Group-level aggregations
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        agg_stats = train_df.groupby(col)["target_signed_log"].agg(
            ["mean", "std", "min", "max", "median"]
        )
        agg_stats.columns = [f"{col}_grp_{c}" for c in agg_stats.columns]

        train_df = train_df.merge(agg_stats, on=col, how="left")
        val_df = val_df.merge(agg_stats, on=col, how="left")
        test_df = test_df.merge(agg_stats, on=col, how="left")

        for c in agg_stats.columns:
            fill_val = (
                train_df["target_signed_log"].mean() if "mean" in c or "median" in c else 0
            )
            train_df[c] = train_df[c].fillna(fill_val)
            val_df[c] = val_df[c].fillna(fill_val)
            test_df[c] = test_df[c].fillna(fill_val)

    # Interaction features
    if len(categorical_cols) >= 2:
        top_cats = categorical_cols[:3]
        for i in range(len(top_cats)):
            for j in range(i + 1, len(top_cats)):
                inter_col = f"{top_cats[i]}__{top_cats[j]}"
                train_df[inter_col] = (
                    train_df[top_cats[i]].astype(str)
                    + "__"
                    + train_df[top_cats[j]].astype(str)
                )
                val_df[inter_col] = (
                    val_df[top_cats[i]].astype(str)
                    + "__"
                    + val_df[top_cats[j]].astype(str)
                )
                test_df[inter_col] = (
                    test_df[top_cats[i]].astype(str)
                    + "__"
                    + test_df[top_cats[j]].astype(str)
                )

                global_mean = train_df["target_signed_log"].mean()
                agg = train_df.groupby(inter_col)["target_signed_log"].agg(["mean", "count"])
                smoothing = 20
                smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
                    agg["count"] + smoothing
                )

                train_df[f"{inter_col}_enc"] = (
                    train_df[inter_col].map(smoothed).fillna(global_mean)
                )
                val_df[f"{inter_col}_enc"] = (
                    val_df[inter_col].map(smoothed).fillna(global_mean)
                )
                test_df[f"{inter_col}_enc"] = (
                    test_df[inter_col].map(smoothed).fillna(global_mean)
                )

    # Temporal features
    if temporal_col is not None and temporal_col in train_df.columns:
        for df in [train_df, val_df, test_df]:
            if temporal_col in df.columns:
                if df[temporal_col].dtype == "datetime64[ns]":
                    df[f"{temporal_col}_year"] = df[temporal_col].dt.year
                    df[f"{temporal_col}_month"] = df[temporal_col].dt.month
                    df[f"{temporal_col}_quarter"] = df[temporal_col].dt.quarter
                    df[f"{temporal_col}_dayofyear"] = df[temporal_col].dt.dayofyear
                    df[f"{temporal_col}_dayofweek"] = df[temporal_col].dt.dayofweek
                    df[f"{temporal_col}_month_sin"] = np.sin(
                        2 * np.pi * df[f"{temporal_col}_month"] / 12
                    )
                    df[f"{temporal_col}_month_cos"] = np.cos(
                        2 * np.pi * df[f"{temporal_col}_month"] / 12
                    )
                elif df[temporal_col].dtype in [np.int64, np.int32, np.float64]:
                    df[f"{temporal_col}_year_sin"] = np.sin(
                        2 * np.pi * df[temporal_col] / 100
                    )
                    df[f"{temporal_col}_year_cos"] = np.cos(
                        2 * np.pi * df[temporal_col] / 100
                    )

    # Numeric feature engineering
    for col in numeric_cols:
        if col not in train_df.columns:
            continue
        if train_df[col].min() >= 0:
            train_df[f"{col}_log"] = np.log1p(train_df[col].clip(lower=0))
            val_df[f"{col}_log"] = np.log1p(val_df[col].clip(lower=0))
            test_df[f"{col}_log"] = np.log1p(test_df[col].clip(lower=0))

        train_min = train_df[col].min()
        train_max = train_df[col].max()
        train_range = train_max - train_min if train_max > train_min else 1
        train_df[f"{col}_norm"] = (train_df[col] - train_min) / train_range
        val_df[f"{col}_norm"] = ((val_df[col] - train_min) / train_range).clip(0, 1)
        test_df[f"{col}_norm"] = ((test_df[col] - train_min) / train_range).clip(0, 1)

    # Handle NaNs
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col in [id_col, target_col, "target_signed_log", "target_mag_log", "target_sign_class"]:
            continue
        train_median = train_df[col].median()
        train_df[col] = train_df[col].fillna(train_median)
        if col in val_df.columns:
            val_df[col] = val_df[col].fillna(train_median)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(train_median)

    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("UNKNOWN").astype(str)
            if col in val_df.columns:
                val_df[col] = val_df[col].fillna("UNKNOWN").astype(str)
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna("UNKNOWN").astype(str)

    return train_df, val_df, test_df


train_proc, val_proc, test_proc = engineer_features(
    train_base, val, test, TARGET, id_col, numeric_cols, categorical_cols, temporal_col
)

# Define final feature set (exclude target-related columns)
final_features = [
    c for c in train_proc.columns
    if c not in [id_col, TARGET, "target_signed_log", "target_mag_log", "target_sign_class"]
]
print(f"Final feature count: {len(final_features)}")

# ============================================================
# 5. PREPARE FEATURES FOR XGBOOST
# ============================================================
cat_feature_cols = [
    c
    for c in final_features
    if train_proc[c].dtype == "object" or train_proc[c].dtype.name == "category"
]
num_feature_cols = [c for c in final_features if c not in cat_feature_cols]
print(
    f"Categorical features: {len(cat_feature_cols)}, Numeric features: {len(num_feature_cols)}"
)

for col in cat_feature_cols:
    train_proc[col] = train_proc[col].astype("category")
    val_proc[col] = val_proc[col].astype("category")
    test_proc[col] = test_proc[col].astype("category")
    all_cats = train_proc[col].cat.categories
    val_proc[col] = val_proc[col].cat.set_categories(all_cats)
    test_proc[col] = test_proc[col].cat.set_categories(all_cats)

for col in num_feature_cols:
    if train_proc[col].isna().any():
        fill_val = train_proc[col].median()
        train_proc[col] = train_proc[col].fillna(fill_val)
        val_proc[col] = val_proc[col].fillna(fill_val)
        test_proc[col] = test_proc[col].fillna(fill_val)

# ============================================================
# 6. DEFINE TARGETS
# ============================================================
y_train_orig = train_proc[TARGET].values
y_val_orig = val_proc[TARGET].values
y_train_mag_log = train_proc["target_mag_log"].values
y_val_mag_log = val_proc["target_mag_log"].values
y_train_sign_class = train_proc["target_sign_class"].values
y_val_sign_class = val_proc["target_sign_class"].values

print(f"Target orig-space: mean={y_train_orig.mean():.2f}, std={y_train_orig.std():.2f}")
print(f"Magnitude log-space: mean={y_train_mag_log.mean():.4f}, std={y_train_mag_log.std():.4f}")
print(f"Sign class distribution (train): neg={sum(y_train_sign_class==0)}, zero={sum(y_train_sign_class==1)}, pos={sum(y_train_sign_class==2)}")

# ============================================================
# 7. MODEL 1: Sign Classifier (3-class: negative, zero, positive)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: Sign Classifier (XGBoost 3-class)")
print("=" * 60)

params_classifier = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "nthread": -1,
    "seed": 42,
}

dtrain_cls = xgb.DMatrix(train_proc[final_features], label=y_train_sign_class, enable_categorical=True)
dval_cls = xgb.DMatrix(val_proc[final_features], label=y_val_sign_class, enable_categorical=True)
dtest_cls = xgb.DMatrix(test_proc[final_features], enable_categorical=True)

model_classifier = xgb.train(
    params_classifier,
    dtrain_cls,
    num_boost_round=3000,
    evals=[(dtrain_cls, "train"), (dval_cls, "val")],
    early_stopping_rounds=100,
    verbose_eval=False,
)

val_cls_probs = model_classifier.predict(dval_cls)
test_cls_probs = model_classifier.predict(dtest_cls)

# Convert class probabilities to sign predictions
# Class 0 = negative, Class 1 = zero, Class 2 = positive
val_sign_pred = np.argmax(val_cls_probs, axis=1)
test_sign_pred = np.argmax(test_cls_probs, axis=1)

# Map to actual sign values: -1, 0, +1
val_sign_values = np.where(val_sign_pred == 0, -1, np.where(val_sign_pred == 1, 0, 1))
test_sign_values = np.where(test_sign_pred == 0, -1, np.where(test_sign_pred == 1, 0, 1))

acc_cls = (val_sign_pred == y_val_sign_class).mean()
print(f"Classifier accuracy: {acc_cls:.4f} | Best iter: {model_classifier.best_iteration}")

# ============================================================
# 8. MODEL 2: Magnitude Regressor (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: Magnitude Regressor (XGBoost)")
print("=" * 60)

params_mag_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "nthread": -1,
    "seed": 42,
}

dtrain_mag = xgb.DMatrix(train_proc[final_features], label=y_train_mag_log, enable_categorical=True)
dval_mag = xgb.DMatrix(val_proc[final_features], label=y_val_mag_log, enable_categorical=True)
dtest_mag = xgb.DMatrix(test_proc[final_features], enable_categorical=True)

model_mag_xgb = xgb.train(
    params_mag_xgb,
    dtrain_mag,
    num_boost_round=8000,
    evals=[(dtrain_mag, "train"), (dval_mag, "val")],
    early_stopping_rounds=200,
    verbose_eval=False,
)

val_mag_pred_xgb = model_mag_xgb.predict(dval_mag)
test_mag_pred_xgb = model_mag_xgb.predict(dtest_mag)

# Evaluate in original space
val_mag_orig_xgb = np.expm1(val_mag_pred_xgb)
val_full_pred_xgb = val_sign_values * val_mag_orig_xgb
rmse_mag_xgb = np.sqrt(mean_squared_error(y_val_orig, val_full_pred_xgb))
print(f"XGBoost magnitude RMSE (original space): {rmse_mag_xgb:.4f} | Best iter: {model_mag_xgb.best_iteration}")

# ============================================================
# 8b. MODEL 2b: Magnitude Regressor (XGBoost Huber Loss)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2b: Magnitude Regressor (XGBoost Pseudo-Huber)")
print("=" * 60)

params_mag_xgb_huber = {
    "objective": "reg:pseudohubererror",
    "huber_slope": 1.0,
    "eval_metric": "rmse",
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "nthread": -1,
    "seed": 42,
}

model_mag_xgb_huber = xgb.train(
    params_mag_xgb_huber,
    dtrain_mag,
    num_boost_round=8000,
    evals=[(dtrain_mag, "train"), (dval_mag, "val")],
    early_stopping_rounds=200,
    verbose_eval=False,
)

val_mag_pred_xgb_huber = model_mag_xgb_huber.predict(dval_mag)
test_mag_pred_xgb_huber = model_mag_xgb_huber.predict(dtest_mag)

# Evaluate in original space
val_mag_orig_xgb_huber = np.expm1(val_mag_pred_xgb_huber)
val_full_pred_xgb_huber = val_sign_values * val_mag_orig_xgb_huber
rmse_mag_xgb_huber = np.sqrt(mean_squared_error(y_val_orig, val_full_pred_xgb_huber))
print(f"XGBoost Huber magnitude RMSE (original space): {rmse_mag_xgb_huber:.4f} | Best iter: {model_mag_xgb_huber.best_iteration}")

# ============================================================
# 9. MODEL 3: Magnitude Regressor (LightGBM)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 3: Magnitude Regressor (LightGBM)")
print("=" * 60)

# Prepare LightGBM datasets
lgb_train = lgb.Dataset(train_proc[final_features], label=y_train_mag_log)
lgb_val = lgb.Dataset(val_proc[final_features], label=y_val_mag_log, reference=lgb_train)
lgb_test = lgb.Dataset(test_proc[final_features], reference=lgb_train)

params_mag_lgb = {
    "objective": "regression",
    "metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "num_leaves": 63,
    "verbosity": -1,
    "nthread": -1,
    "seed": 42,
}

model_mag_lgb = lgb.train(
    params_mag_lgb,
    lgb_train,
    num_boost_round=8000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(period=0)],
)

val_mag_pred_lgb = model_mag_lgb.predict(val_proc[final_features], num_iteration=model_mag_lgb.best_iteration)
test_mag_pred_lgb = model_mag_lgb.predict(test_proc[final_features], num_iteration=model_mag_lgb.best_iteration)

# Evaluate in original space
val_mag_orig_lgb = np.expm1(val_mag_pred_lgb)
val_full_pred_lgb = val_sign_values * val_mag_orig_lgb
rmse_mag_lgb = np.sqrt(mean_squared_error(y_val_orig, val_full_pred_lgb))
print(f"LightGBM magnitude RMSE (original space): {rmse_mag_lgb:.4f} | Best iter: {model_mag_lgb.best_iteration}")

# ============================================================
# 10. ENSEMBLE: Inverse-RMSE Weighted Ensemble
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE: Inverse-RMSE Weighted (XGBoost + XGBoost-Huber + LightGBM)")
print("=" * 60)

# Collect all model predictions and their validation RMSEs
model_preds_val = {
    "xgb_sqr": val_mag_pred_xgb,
    "xgb_huber": val_mag_pred_xgb_huber,
    "lgb": val_mag_pred_lgb,
}
model_preds_test = {
    "xgb_sqr": test_mag_pred_xgb,
    "xgb_huber": test_mag_pred_xgb_huber,
    "lgb": test_mag_pred_lgb,
}
model_rmses = {
    "xgb_sqr": rmse_mag_xgb,
    "xgb_huber": rmse_mag_xgb_huber,
    "lgb": rmse_mag_lgb,
}

# Compute inverse-RMSE weights
inv_rmses = {k: 1.0 / v for k, v in model_rmses.items()}
weight_sum = sum(inv_rmses.values())
weights = {k: v / weight_sum for k, v in inv_rmses.items()}

print(f"Weights: {weights}")
print(f"Individual RMSEs: {model_rmses}")

# Weighted ensemble
val_mag_pred_ens = sum(weights[k] * model_preds_val[k] for k in model_preds_val)
test_mag_pred_ens = sum(weights[k] * model_preds_test[k] for k in model_preds_test)

val_mag_orig_ens = np.expm1(val_mag_pred_ens)
val_full_pred_ens = val_sign_values * val_mag_orig_ens
rmse_ens = np.sqrt(mean_squared_error(y_val_orig, val_full_pred_ens))
print(f"Ensemble RMSE (original space): {rmse_ens:.4f}")

# ============================================================
# 11. SELECT BEST MODEL
# ============================================================
print("\n" + "=" * 60)
print("MODEL SELECTION")
print("=" * 60)

results = {
    "xgb_magnitude": (rmse_mag_xgb, val_full_pred_xgb, test_mag_pred_xgb, test_sign_values),
    "xgb_huber": (rmse_mag_xgb_huber, val_full_pred_xgb_huber, test_mag_pred_xgb_huber, test_sign_values),
    "lgb_magnitude": (rmse_mag_lgb, val_full_pred_lgb, test_mag_pred_lgb, test_sign_values),
    "ensemble": (rmse_ens, val_full_pred_ens, test_mag_pred_ens, test_sign_values),
}

for name, (rmse, _, _, _) in results.items():
    print(f"  {name}: RMSE = {rmse:.4f}")

best_name = min(results, key=lambda k: results[k][0])
best_rmse = results[best_name][0]
best_test_mag_pred = results[best_name][2]
best_test_sign = results[best_name][3]

print(f"\nSelected: {best_name} (RMSE = {best_rmse:.4f})")

# ============================================================
# 12. GENERATE SUBMISSION
# ============================================================
# Final prediction: sign * expm1(magnitude_log_pred)
final_test_pred = best_test_sign * np.expm1(best_test_mag_pred)

submission = pd.DataFrame(
    {"id": range(len(final_test_pred)), "Cash Disbursements (US$)": final_test_pred}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Predictions: mean={final_test_pred.mean():.2f}, std={final_test_pred.std():.2f}, "
    f"min={final_test_pred.min():.2f}, max={final_test_pred.max():.2f}"
)

# ============================================================
# 13. FINAL VALIDATION SCORE
# ============================================================
final_rmse = np.sqrt(mean_squared_error(y_val_orig, results[best_name][1]))
print(f"\n--- Validation Summary ---")
print(f"Split method: Stratified random (20% validation)")
print(f"Metric: RMSE in original space (official competition metric)")
print(f"Best model: {best_name}")
print(f"Final Validation Score: {final_rmse}")
