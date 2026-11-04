import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import os
import pickle

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget stats:\n{train['NADAC Rate'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
target_col = "NADAC Rate"
id_col = "id" if "id" in train.columns else None

y_full = train[target_col].copy()
train = train.drop(columns=[target_col])

feature_cols = [c for c in train.columns if c != id_col]
cat_cols = (
    train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
)
num_cols = [c for c in feature_cols if c not in cat_cols]

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
train_data, val_data = train_test_split(
    pd.concat([train, y_full], axis=1), test_size=0.15, random_state=42
)
train = train_data.drop(columns=[target_col])
val = val_data.drop(columns=[target_col])
y_train = train_data[target_col].values
y_val = val_data[target_col].values

print(f"Train split: {train.shape[0]} samples")
print(f"Validation split: {val.shape[0]} samples")
print(f"Test: {test.shape[0]} samples")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(
    df, cat_cols, num_cols, target_enc_dict=None, freq_enc_dict=None, fit=False
):
    """Apply feature engineering. Fit target/frequency encoding only when fit=True."""
    df = df.copy()

    # --- Target encoding for categoricals ---
    if target_enc_dict is not None and not fit:
        for col in cat_cols:
            if col in df.columns and col in target_enc_dict:
                df[col + "_te"] = (
                    df[col]
                    .map(target_enc_dict[col])
                    .fillna(target_enc_dict[col].get("__global__", 0))
                )

    # --- Basic numerical features ---
    for col in num_cols:
        if col in df.columns:
            if df[col].min() > 0:
                df[col + "_log"] = np.log1p(df[col])
            df[col + "_sq"] = df[col] ** 2

    # --- Interaction features between top numerical features ---
    if len(num_cols) >= 2:
        top_num = num_cols[:5]
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                c1, c2 = top_num[i], top_num[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # --- Categorical frequency encoding ---
    if fit:
        for col in cat_cols:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True)
                df[col + "_freq"] = df[col].map(freq)
    else:
        if freq_enc_dict is not None:
            for col in cat_cols:
                if col in df.columns and col in freq_enc_dict:
                    df[col + "_freq"] = df[col].map(freq_enc_dict[col]).fillna(0)

    return df


# Compute target encoding and frequency encoding from TRAIN only
target_enc_dict = {}
freq_enc_dict = {}
global_mean = y_train.mean()

for col in cat_cols:
    if col in train.columns:
        agg = pd.DataFrame({"cat": train[col].values, "target": y_train})
        stats = agg.groupby("cat")["target"].agg(["mean", "count"])
        smoothing = 10
        stats["te"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        target_enc_dict[col] = stats["te"].to_dict()
        target_enc_dict[col]["__global__"] = global_mean

        freq = train[col].value_counts(normalize=True).to_dict()
        freq_enc_dict[col] = freq

# Apply feature engineering
train_fe = engineer_features(
    train, cat_cols, num_cols, target_enc_dict, freq_enc_dict, fit=True
)
val_fe = engineer_features(
    val, cat_cols, num_cols, target_enc_dict, freq_enc_dict, fit=False
)
test_fe = engineer_features(
    test, cat_cols, num_cols, target_enc_dict, freq_enc_dict, fit=False
)

# ============================================================
# 5. HANDLE CATEGORICALS (label encode for tree-based models)
# ============================================================
label_encoders = {}
all_cat_cols_in_fe = [
    c
    for c in train_fe.columns
    if train_fe[c].dtype == "object" or train_fe[c].dtype == "category"
]

for col in all_cat_cols_in_fe:
    train_vals = train_fe[col].astype(str)
    unique_vals = sorted(train_vals.unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    unknown_code = len(unique_vals)  # code assigned to unseen categories

    train_fe[col] = train_vals.map(mapping).fillna(unknown_code).astype(int)
    val_fe[col] = val_fe[col].astype(str).map(mapping).fillna(unknown_code).astype(int)
    test_fe[col] = test_fe[col].astype(str).map(mapping).fillna(unknown_code).astype(int)

    label_encoders[col] = mapping

# ============================================================
# 6. FINALIZE FEATURE SET
# ============================================================
final_features = train_fe.columns.tolist()

for df_name, df in [("train", train_fe), ("val", val_fe), ("test", test_fe)]:
    df[final_features] = df[final_features].fillna(-999)

val_fe = val_fe[final_features]
test_fe = test_fe[final_features]

print(f"Final feature count: {len(final_features)}")

# ============================================================
# 7. MODEL CONFIGURATIONS: 3 LightGBM ensemble
# ============================================================
# Target contains negative values (min ≈ -5.77), so log1p is invalid.
# Train directly on the original target values.
y_train_log = y_train
y_val_log = y_val

model_configs = [
    {
        "name": "conservative",
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "min_child_samples": 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "name": "moderate",
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 2000,
        "min_child_samples": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 2,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "random_state": 123,
        "verbose": -1,
    },
    {
        "name": "aggressive",
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.02,
        "n_estimators": 3000,
        "min_child_samples": 3,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 7,
        "verbose": -1,
    },
]

# ============================================================
# 8. TRAINING LOOP WITH VALIDATION TRACKING
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

models = []
val_preds_list = []
train_preds_list = []
training_history = []

print("\n" + "=" * 60)
print("TRAINING LIGHTGBM ENSEMBLE")
print("=" * 60)

for i, config in enumerate(model_configs):
    model = lgb.LGBMRegressor(**config)
    early_stop = lgb.early_stopping(stopping_rounds=100, verbose=False)

    model.fit(
        train_fe[final_features],
        y_train_log,
        eval_set=[(val_fe[final_features], y_val_log)],
        callbacks=[early_stop],
        eval_metric="rmse",
    )

    best_iter = model.best_iteration_

    # Validation predictions (original space — no transform needed)
    val_pred = model.predict(val_fe[final_features], num_iteration=best_iter)

    # Training predictions (for overfitting check)
    train_pred = model.predict(train_fe[final_features], num_iteration=best_iter)

    # Compute metrics in ORIGINAL space (official metric: RMSE)
    val_rmse = root_mean_squared_error(y_val, val_pred)
    train_rmse = root_mean_squared_error(y_train, train_pred)

    print(
        f"  Model {i+1} ({config['name']}): best_iter={best_iter}, "
        f"train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}, "
        f"overfit_gap={val_rmse - train_rmse:.4f}"
    )

    models.append(model)
    val_preds_list.append(val_pred)
    train_preds_list.append(train_pred)
    training_history.append(
        {
            "name": config["name"],
            "best_iteration": best_iter,
            "val_rmse": val_rmse,
            "train_rmse": train_rmse,
        }
    )

# ============================================================
# 9. ENSEMBLE WEIGHTS (inverse validation RMSE)
# ============================================================
val_rmses = np.array([h["val_rmse"] for h in training_history])
weights = 1.0 / val_rmses
weights = weights / weights.sum()
print(
    f"\nEnsemble weights: {dict(zip([h['name'] for h in training_history], weights.round(4)))}"
)

# ============================================================
# 10. ENSEMBLE VALIDATION PREDICTION
# ============================================================
val_pred_ensemble = np.average(val_preds_list, axis=0, weights=weights)
val_rmse_ensemble = root_mean_squared_error(y_val, val_pred_ensemble)
print(f"Ensemble validation RMSE: {val_rmse_ensemble:.4f}")

# ============================================================
# 11. TEST PREDICTIONS
# ============================================================
test_preds_list = []
for model in models:
    best_iter = model.best_iteration_
    test_pred = model.predict(test_fe[final_features], num_iteration=best_iter)
    test_preds_list.append(test_pred)

test_pred_ensemble = np.average(test_preds_list, axis=0, weights=weights)

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
test_ids = test.index if id_col is None else test[id_col].values
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "NADAC Rate": test_pred_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================
best_model_idx = int(np.argmin(val_rmses))
best_model = models[best_model_idx]
fi = pd.Series(best_model.feature_importances_, index=final_features).sort_values(
    ascending=False
)
print(f"\nTop 10 features (from '{training_history[best_model_idx]['name']}'):")
print(fi.head(10))

# ============================================================
# 14. FINAL VALIDATION SCORE
# ============================================================
score = val_rmse_ensemble
print(f"\n{'=' * 60}")
print(f"VALIDATION SUMMARY")
print(f"{'=' * 60}")
print(f"  Split: 15% hold-out (random_state=42)")
print(f"  Metric: RMSE (official competition metric)")
print(f"  Target: NADAC Rate (original space)")
print(f"  Models: 3 LightGBM (conservative/moderate/aggressive)")
print(f"  Ensemble: weighted average (weights ∝ 1/val_RMSE)")
print(f"  Post-processing: none (target can be negative)")
print(f"  Individual val RMSEs: {val_rmses.round(4)}")
print(f"  Ensemble val RMSE: {val_rmse_ensemble:.4f}")
print(f"{'=' * 60}")
print(f"Final Validation Score: {score}")
