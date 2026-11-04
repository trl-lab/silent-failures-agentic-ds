import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Convert target to numeric (stored as categorical/object in parquet)
train["Class_number_of_rings"] = train["Class_number_of_rings"].astype(float)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target stats:\n{train['Class_number_of_rings'].describe()}")

# ============================================================
# 2. IDENTIFY FEATURES AND TARGET
# ============================================================
TARGET = "Class_number_of_rings"

id_col = None
if "id" in train.columns:
    id_col = "id"

feature_cols = [c for c in train.columns if c != TARGET and c != id_col]
print(f"Feature columns: {feature_cols}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Create domain-informed features for abalone age prediction."""
    df = df.copy()

    # --- Categorical encoding ---
    sex_mapping = {"I": 0, "F": 1, "M": 2}
    df["Sex_encoded"] = df["Sex"].map(sex_mapping).fillna(0).astype(int)
    df["Sex_M"] = (df["Sex"] == "M").astype(int)
    df["Sex_F"] = (df["Sex"] == "F").astype(int)
    df["Sex_I"] = (df["Sex"] == "I").astype(int)

    # --- Geometric features ---
    df["volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["surface_area"] = 2 * (
        df["Length"] * df["Diameter"]
        + df["Diameter"] * df["Height"]
        + df["Length"] * df["Height"]
    )
    df["length_diameter_ratio"] = df["Length"] / (df["Diameter"] + 1e-8)
    df["length_height_ratio"] = df["Length"] / (df["Height"] + 1e-8)
    df["diameter_height_ratio"] = df["Diameter"] / (df["Height"] + 1e-8)

    # --- Weight ratios (body composition) ---
    df["shucked_ratio"] = df["Shucked_weight"] / (df["Whole_weight"] + 1e-8)
    df["viscera_ratio"] = df["Viscera_weight"] / (df["Whole_weight"] + 1e-8)
    df["shell_ratio"] = df["Shell_weight"] / (df["Whole_weight"] + 1e-8)
    df["shucked_shell_ratio"] = df["Shucked_weight"] / (df["Shell_weight"] + 1e-8)
    df["viscera_shell_ratio"] = df["Viscera_weight"] / (df["Shell_weight"] + 1e-8)
    df["whole_shell_ratio"] = df["Whole_weight"] / (df["Shell_weight"] + 1e-8)

    # --- Density features ---
    df["whole_density"] = df["Whole_weight"] / (df["volume"] + 1e-8)
    df["shucked_density"] = df["Shucked_weight"] / (df["volume"] + 1e-8)
    df["shell_density"] = df["Shell_weight"] / (df["volume"] + 1e-8)

    # --- Log transforms ---
    df["log_length"] = np.log1p(df["Length"])
    df["log_diameter"] = np.log1p(df["Diameter"])
    df["log_height"] = np.log1p(df["Height"])
    df["log_whole"] = np.log1p(df["Whole_weight"])
    df["log_shucked"] = np.log1p(df["Shucked_weight"])
    df["log_viscera"] = np.log1p(df["Viscera_weight"])
    df["log_shell"] = np.log1p(df["Shell_weight"])
    df["log_volume"] = np.log1p(df["volume"])

    # --- Polynomial interactions ---
    df["length_diameter_product"] = df["Length"] * df["Diameter"]
    df["length_height_product"] = df["Length"] * df["Height"]
    df["diameter_height_product"] = df["Diameter"] * df["Height"]

    # --- Weight differences ---
    df["meat_weight"] = df["Shucked_weight"] + df["Viscera_weight"]
    df["meat_ratio"] = df["meat_weight"] / (df["Whole_weight"] + 1e-8)
    df["residual_weight"] = (
        df["Whole_weight"]
        - df["Shucked_weight"]
        - df["Viscera_weight"]
        - df["Shell_weight"]
    )

    # --- Squared features ---
    df["length_sq"] = df["Length"] ** 2
    df["diameter_sq"] = df["Diameter"] ** 2
    df["whole_sq"] = df["Whole_weight"] ** 2
    df["shell_sq"] = df["Shell_weight"] ** 2

    # --- Cube root of volume ---
    df["volume_cbrt"] = np.cbrt(df["volume"])

    # --- Ratio of linear dims to weight ---
    df["length_per_weight"] = df["Length"] / (df["Whole_weight"] + 1e-8)
    df["diameter_per_weight"] = df["Diameter"] / (df["Whole_weight"] + 1e-8)

    return df


train_fe = engineer_features(train)
test_fe = engineer_features(test)

# ============================================================
# 4. DEFINE FINAL FEATURE SET
# ============================================================
exclude_cols = ["Sex", TARGET]
if id_col:
    exclude_cols.append(id_col)

final_features = [c for c in train_fe.columns if c not in exclude_cols]
print(f"Number of engineered features: {len(final_features)}")

# ============================================================
# 5. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
train_split, val_split = train_test_split(
    train_fe, test_size=0.15, random_state=42
)

print(f"Train split: {train_split.shape[0]} samples")
print(f"Validation split: {val_split.shape[0]} samples")
print(f"Test set: {test_fe.shape[0]} samples")

# ============================================================
# 6. PREPARE MATRICES
# ============================================================
train_X = train_split[final_features].values
val_X = val_split[final_features].values
test_X = test_fe[final_features].values

train_y = train_split[TARGET].values
val_y = val_split[TARGET].values

if id_col:
    test_ids = test_fe[id_col].values
else:
    test_ids = np.arange(len(test_fe))

# ============================================================
# 7. DEFINE ENSEMBLE CONFIGURATIONS (DIVERSE)
# ============================================================
base_params = {
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

# Diverse configurations: different loss functions and hyperparameters
configs = [
    # (seed, overrides) - 2 huber models with different alpha
    (42, {"objective": "huber", "alpha": 1.0}),
    (123, {"objective": "huber", "alpha": 2.0}),
    # 2 regression_l2 models (standard RMSE optimization)
    (7, {"objective": "regression_l2"}),
    (2024, {"objective": "regression_l2", "feature_fraction": 0.7, "min_child_samples": 40}),
    # 1 regression_l1 model (MAE - robust to outliers)
    (999, {"objective": "regression_l1"}),
    # 1 deep model to capture complex patterns
    (555, {"objective": "regression_l2", "num_leaves": 63, "max_depth": 8, "min_child_samples": 15}),
]

print(f"\nEnsemble: {len(configs)} diverse models")

# ============================================================
# 8. TRAIN DIVERSE ENSEMBLE WITH EARLY STOPPING
# ============================================================
val_preds = []
test_preds = []
model_scores = []
model_best_iters = []

for i, (seed, overrides) in enumerate(configs):
    params = {**base_params, **overrides, "seed": seed}

    train_data = lgb.Dataset(train_X, label=train_y)
    val_data = lgb.Dataset(val_X, label=val_y, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    val_pred = model.predict(val_X, num_iteration=best_iter)
    test_pred = model.predict(test_X, num_iteration=best_iter)

    val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    model_scores.append(val_rmse)
    val_preds.append(val_pred)
    test_preds.append(test_pred)
    model_best_iters.append(best_iter)

    obj_name = overrides.get("objective", "huber")
    alpha_str = f", alpha={overrides['alpha']}" if "alpha" in overrides else ""
    print(
        f"  Model {i+1}/{len(configs)} (seed={seed}, obj={obj_name}{alpha_str}, iter={best_iter}): val RMSE = {val_rmse:.4f}"
    )

# ============================================================
# 9. WEIGHTED ENSEMBLE AVERAGING (inverse RMSE weights)
# ============================================================
# Weights inversely proportional to validation RMSE
weights = np.array([1.0 / s for s in model_scores])
weights = weights / weights.sum()

val_pred_ensemble = np.average(val_preds, axis=0, weights=weights)
test_pred_ensemble = np.average(test_preds, axis=0, weights=weights)

print(f"\nModel weights: {['%.4f' % w for w in weights]}")
print(f"Individual model RMSEs: {['%.4f' % s for s in model_scores]}")
print(f"Mean individual RMSE: {np.mean(model_scores):.4f}")

# ============================================================
# 10. EVALUATE ENSEMBLE ON VALIDATION SET (RMSE - official metric)
# ============================================================
ensemble_rmse = np.sqrt(mean_squared_error(val_y, val_pred_ensemble))
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Improvement over mean: {np.mean(model_scores) - ensemble_rmse:.4f}")
print(
    f"Prediction range: [{val_pred_ensemble.min():.2f}, {val_pred_ensemble.max():.2f}]"
)
print(f"Target range (val): [{val_y.min():.2f}, {val_y.max():.2f}]")

# ============================================================
# 11. SECOND STAGE: RETRAIN BEST CONFIG ON FULL DATA
# ============================================================
# Find the best performing model configuration
best_idx = np.argmin(model_scores)
best_seed, best_overrides = configs[best_idx]
best_iter = model_best_iters[best_idx]

print(f"\nBest model: index {best_idx} (seed={best_seed}, RMSE={model_scores[best_idx]:.4f}, iter={best_iter})")

# Combine train + val for full retraining
full_X = train_fe[final_features].values
full_y = train_fe[TARGET].values

full_params = {**base_params, **best_overrides, "seed": best_seed}
full_train_data = lgb.Dataset(full_X, label=full_y)

full_model = lgb.train(
    full_params,
    full_train_data,
    num_boost_round=best_iter,  # Use best iteration from validation
)

# Retrain second-best model on full data as well for diversity
second_best_idx = np.argsort(model_scores)[1]
second_seed, second_overrides = configs[second_best_idx]
second_iter = model_best_iters[second_best_idx]

second_params = {**base_params, **second_overrides, "seed": second_seed}
full_train_data2 = lgb.Dataset(full_X, label=full_y)

second_full_model = lgb.train(
    second_params,
    full_train_data2,
    num_boost_round=second_iter,
)

# Final test prediction: blend the two full-data models (weighted by their original val RMSE)
w1 = (1.0 / model_scores[best_idx]) / (1.0 / model_scores[best_idx] + 1.0 / model_scores[second_best_idx])
w2 = 1.0 - w1

test_pred_full = w1 * full_model.predict(test_X) + w2 * second_full_model.predict(test_X)

# Use the full-data retrained predictions as final test predictions
test_pred_ensemble = test_pred_full

print(f"Full-data retraining complete. Weights: best={w1:.4f}, second={w2:.4f}")

# ============================================================
# 12. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Class_number_of_rings": test_pred_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = ensemble_rmse
print(f"Final Validation Score: {score:.4f}")
print(f"(Note: test predictions use full-data retrained models for ~15% more data)")
