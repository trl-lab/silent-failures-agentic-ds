import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target_col = "Class_number_of_rings"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "ring" in c.lower() or "class" in c.lower()
    ]
    if possible_targets:
        target_col = possible_targets[0]

# Fix: explicitly cast target values to numeric float32
y = pd.to_numeric(train_df[target_col], errors="coerce").astype(np.float32).values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()


# 2. Feature Engineering
def preprocess_features(df):
    data = df.copy()

    # Standardize column names
    rename_dict = {}
    for col in data.columns:
        c_clean = col.lower().replace(" ", "_").replace("-", "_")
        rename_dict[col] = c_clean
    data.rename(columns=rename_dict, inplace=True)

    eps = 1e-6
    # Engineered physical and geometric features
    data["volume"] = data["length"] * data["diameter"] * data["height"]
    data["surface_area"] = 2 * (
        data["length"] * data["diameter"]
        + data["length"] * data["height"]
        + data["diameter"] * data["height"]
    )
    data["density"] = data["whole_weight"] / (data["volume"] + eps)
    data["weight_sum_parts"] = (
        data["shucked_weight"] + data["viscera_weight"] + data["shell_weight"]
    )
    data["water_weight"] = data["whole_weight"] - data["weight_sum_parts"]

    # Weight and dimension ratios
    data["ratio_shucked_whole"] = data["shucked_weight"] / (data["whole_weight"] + eps)
    data["ratio_viscera_whole"] = data["viscera_weight"] / (data["whole_weight"] + eps)
    data["ratio_shell_whole"] = data["shell_weight"] / (data["whole_weight"] + eps)
    data["ratio_shucked_shell"] = data["shucked_weight"] / (data["shell_weight"] + eps)
    data["ratio_length_diameter"] = data["length"] / (data["diameter"] + eps)
    data["ratio_height_length"] = data["height"] / (data["length"] + eps)
    data["ratio_height_diameter"] = data["height"] / (data["diameter"] + eps)

    # One-hot encode categorical features like 'sex'
    if "sex" in data.columns:
        data = pd.get_dummies(data, columns=["sex"], drop_first=False)

    return data


X_train_proc = preprocess_features(X_train)
X_test_proc = preprocess_features(X_test)

# Align train and test feature columns
X_train_proc, X_test_proc = X_train_proc.align(
    X_test_proc, join="left", axis=1, fill_value=0
)

# Convert all features to float32
X_train_proc = X_train_proc.astype(np.float32)
X_test_proc = X_test_proc.astype(np.float32)

# 3. Cross-Validation Setup & Modeling
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(X_train_proc))
oof_preds_ridge = np.zeros(len(X_train_proc))
test_preds_lgb = np.zeros(len(X_test_proc))
test_preds_ridge = np.zeros(len(X_test_proc))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    # LightGBM model
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)

    val_pred_lgb = model_lgb.predict(X_va)
    oof_preds_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += model_lgb.predict(X_test_proc) / n_splits

    # Linear Ridge model for ensembling
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(X_tr, y_tr)
    val_pred_ridge = ridge.predict(X_va)
    oof_preds_ridge[val_idx] = val_pred_ridge
    test_preds_ridge += ridge.predict(X_test_proc) / n_splits

    fold_rmse = root_mean_squared_error(
        y_va, 0.85 * val_pred_lgb + 0.15 * val_pred_ridge
    )
    print(f"Fold {fold + 1} Blend RMSE: {fold_rmse:.4f}")

oof_preds = 0.85 * oof_preds_lgb + 0.15 * oof_preds_ridge
test_preds = 0.85 * test_preds_lgb + 0.15 * test_preds_ridge

overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall OOF RMSE: {overall_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Class_number_of_rings": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
