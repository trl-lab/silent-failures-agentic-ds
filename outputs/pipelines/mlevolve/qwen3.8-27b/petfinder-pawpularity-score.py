import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(
    f"Target stats: mean={train_df['Pawpularity'].mean():.2f}, std={train_df['Pawpularity'].std():.2f}"
)

# Separate target
y_full = train_df["Pawpularity"].values
train_df = train_df.drop(columns=["Pawpularity"])

# Drop Id column from train and test
if "Id" in train_df.columns:
    train_df = train_df.drop(columns=["Id"])
if "Id" in test_df.columns:
    test_df = test_df.drop(columns=["Id"])

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT (before any feature engineering)
# ============================================================
train_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=y_full
)
y_train = y_full[train_df.index]
y_val = y_full[val_df.index]

print(f"Train split: {train_df.shape}, Val split: {val_df.shape}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
# Actual feature columns (all binary metadata flags)
feature_cols = [
    "Subject Focus", "Eyes", "Face", "Near", "Action", "Accessory",
    "Group", "Collage", "Human", "Occlusion", "Info", "Blur"
]


def engineer_features(df):
    df = df.copy()

    # Drop Id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # --- Key positive indicators ---
    positive_cols = ["Subject Focus", "Eyes", "Face", "Near"]
    df["PositiveSum"] = df[positive_cols].sum(axis=1)

    # --- Key negative indicators ---
    negative_cols = ["Collage", "Human", "Occlusion", "Info", "Blur"]
    df["NegativeSum"] = df[negative_cols].sum(axis=1)

    # --- Pos/Neg difference ---
    df["PosNegDiff"] = df["PositiveSum"] - df["NegativeSum"]

    # --- High-value pairwise interactions ---
    df["Eyes_Face"] = df["Eyes"] * df["Face"]
    df["Eyes_Near"] = df["Eyes"] * df["Near"]
    df["Face_Near"] = df["Face"] * df["Near"]
    df["SubjectFocus_Eyes"] = df["Subject Focus"] * df["Eyes"]

    return df


train_feat = engineer_features(train_df)
val_feat = engineer_features(val_df)
test_feat = engineer_features(test_df)

# ============================================================
# 5. CONVERT TO NUMPY
# ============================================================
X_train = train_feat.values.astype(np.float32)
X_val = val_feat.values.astype(np.float32)
X_test = test_feat.values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

n_features = X_train.shape[1]
print(
    f"Features: {n_features}, Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
)

test_ids = np.arange(len(test_feat))


# ============================================================
# 7. TRAINING CONFIGURATION - Two LightGBM Models
# ============================================================
lgb_params_A = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.01,
    "min_child_samples": 50,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 10000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

lgb_params_B = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.005,
    "min_child_samples": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 3,
    "n_estimators": 10000,
    "verbose": -1,
    "random_state": 123,
    "n_jobs": -1,
}

# ============================================================
# 8. TRAIN LIGHTGBM MODEL A
# ============================================================
print("\n--- Training LightGBM Model A ---")
lgb_model_A = lgb.LGBMRegressor(**lgb_params_A)
lgb_model_A.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=300, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
lgb_A_best_iter = lgb_model_A.best_iteration_
lgb_A_val_pred = lgb_model_A.predict(X_val)
lgb_A_val_rmse = float(np.sqrt(mean_squared_error(y_val, lgb_A_val_pred)))
print(f"LightGBM A | best_iter={lgb_A_best_iter} | Val RMSE={lgb_A_val_rmse:.4f}")

# ============================================================
# 9. TRAIN LIGHTGBM MODEL B
# ============================================================
print("\n--- Training LightGBM Model B ---")
lgb_model_B = lgb.LGBMRegressor(**lgb_params_B)
lgb_model_B.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=300, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
lgb_B_best_iter = lgb_model_B.best_iteration_
lgb_B_val_pred = lgb_model_B.predict(X_val)
lgb_B_val_rmse = float(np.sqrt(mean_squared_error(y_val, lgb_B_val_pred)))
print(f"LightGBM B | best_iter={lgb_B_best_iter} | Val RMSE={lgb_B_val_rmse:.4f}")

# ============================================================
# 10. ENSEMBLE (simple 0.5/0.5 average)
# ============================================================
val_ensemble_pred = 0.5 * lgb_A_val_pred + 0.5 * lgb_B_val_pred
val_ensemble_pred = np.clip(val_ensemble_pred, 0, 100)

final_val_rmse = float(np.sqrt(mean_squared_error(y_val, val_ensemble_pred)))
print(f"\nEnsemble Val RMSE: {final_val_rmse:.4f}")

# ============================================================
# 11. TEST INFERENCE
# ============================================================
lgb_A_test_pred = lgb_model_A.predict(X_test)
lgb_B_test_pred = lgb_model_B.predict(X_test)

test_ensemble_pred = 0.5 * lgb_A_test_pred + 0.5 * lgb_B_test_pred
test_ensemble_pred = np.clip(test_ensemble_pred, 0, 100)

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Pawpularity": test_ensemble_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Pred stats: mean={test_ensemble_pred.mean():.2f}, std={test_ensemble_pred.std():.2f}, min={test_ensemble_pred.min():.2f}, max={test_ensemble_pred.max():.2f}"
)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================
print(f"\n{'='*50}")
print(f"  LightGBM A Val RMSE: {lgb_A_val_rmse:.4f}")
print(f"  LightGBM B Val RMSE: {lgb_B_val_rmse:.4f}")
print(f"  Ensemble Val RMSE:   {final_val_rmse:.4f}")
print(f"{'='*50}")

print(f"Final Validation Score: {final_val_rmse:.4f}")
