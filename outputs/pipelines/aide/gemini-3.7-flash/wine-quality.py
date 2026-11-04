import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


def engineer_features(df):
    data = df.copy()

    # Binary encoding for color if present
    if "color" in data.columns:
        data["is_white"] = (data["color"] == "white").astype(float)
        data = data.drop(columns=["color"])

    # 1. Acidity interactions and balance
    data["total_acidity"] = data["fixed_acidity"] + data["volatile_acidity"]
    data["fixed_to_volatile_acidity"] = data["fixed_acidity"] / (
        data["volatile_acidity"] + 1e-6
    )
    data["citric_to_total_acidity"] = data["citric_acid"] / (
        data["total_acidity"] + 1e-6
    )
    data["volatile_to_total_acidity"] = data["volatile_acidity"] / (
        data["total_acidity"] + 1e-6
    )

    # 2. Sulfur dioxide and enological molecular SO2 equilibrium
    data["bound_sulfur_dioxide"] = (
        data["total_sulfur_dioxide"] - data["free_sulfur_dioxide"]
    )
    data["free_to_total_sulfur_dioxide"] = data["free_sulfur_dioxide"] / (
        data["total_sulfur_dioxide"] + 1e-6
    )
    # Molecular SO2 (effective active SO2 based on Henderson-Hasselbalch equation)
    data["molecular_so2"] = data["free_sulfur_dioxide"] / (
        1.0 + 10.0 ** (data["pH"] - 1.81)
    )

    # 3. Sugar, Alcohol, and Density balance
    data["sugar_to_total_acidity"] = data["residual_sugar"] / (
        data["total_acidity"] + 1e-6
    )
    data["sugar_to_density"] = data["residual_sugar"] / (data["density"] + 1e-6)
    data["alcohol_to_density"] = data["alcohol"] / (data["density"] + 1e-6)
    data["sugar_to_alcohol"] = data["residual_sugar"] / (data["alcohol"] + 1e-6)
    data["density_offset"] = (data["density"] - 1.0) * 1000.0

    # 4. Salts, minerals, and pH interactions
    data["hydrogen_ion_conc"] = 10.0 ** (-data["pH"])
    data["sulphates_to_chlorides"] = data["sulphates"] / (data["chlorides"] + 1e-6)
    data["chlorides_to_acidity"] = data["chlorides"] / (data["total_acidity"] + 1e-6)
    data["sulphates_to_acidity"] = data["sulphates"] / (data["total_acidity"] + 1e-6)
    data["alcohol_pH_interaction"] = data["alcohol"] * data["pH"]

    return data


# Apply feature engineering
train_processed = engineer_features(train_df)
test_processed = engineer_features(test_df)

feature_cols = [col for col in train_processed.columns if col != "quality"]
target_col = "quality"

X = train_processed[feature_cols].values
y = train_processed[target_col].values
X_test = test_processed[feature_cols].values

# Multi-seed 5-fold cross-validation
SEEDS = [42, 123, 456]
N_SPLITS = 5

oof_seed_predictions = np.zeros((len(SEEDS), len(train_processed)))
test_seed_predictions = np.zeros((len(SEEDS), len(test_processed)))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "n_estimators": 2000,
}

for seed_idx, seed in enumerate(SEEDS):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    params = xgb_params.copy()
    params["random_state"] = seed

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=60)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_preds = model.predict(X_va)
        oof_seed_predictions[seed_idx, val_idx] = val_preds
        test_seed_predictions[seed_idx] += model.predict(X_test) / N_SPLITS

    seed_rmse = root_mean_squared_error(y, oof_seed_predictions[seed_idx])
    print(f"Seed {seed} Out-of-Fold RMSE: {seed_rmse:.5f}")

# Final blended predictions
oof_final = np.mean(oof_seed_predictions, axis=0)
test_final = np.mean(test_seed_predictions, axis=0)

overall_rmse = root_mean_squared_error(y, oof_final)
print(f"\nFinal Ensembled Out-of-Fold RMSE: {overall_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_final)), "quality": test_final})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
