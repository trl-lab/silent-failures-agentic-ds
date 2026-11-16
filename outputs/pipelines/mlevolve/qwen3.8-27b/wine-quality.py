import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import pickle

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(
    f"Target distribution (train):\n{train_df['quality'].value_counts().sort_index()}"
)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================


def engineer_features(df):
    """Apply domain-informed feature engineering for wine quality prediction."""
    df = df.copy()

    # --- Sulfur dioxide family interactions ---
    df["free_so2_ratio"] = df["free_sulfur_dioxide"] / (
        df["total_sulfur_dioxide"] + 1e-8
    )
    df["bound_so2"] = df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"]
    df["so2_per_alcohol"] = df["total_sulfur_dioxide"] / (df["alcohol"] + 1e-8)

    # --- Acidity family interactions ---
    df["volatile_to_fixed_ratio"] = df["volatile_acidity"] / (
        df["fixed_acidity"] + 1e-8
    )
    df["citric_to_fixed_ratio"] = df["citric_acid"] / (df["fixed_acidity"] + 1e-8)
    df["total_acidity"] = (
        df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    )
    df["volatile_acidity_sq"] = df["volatile_acidity"] ** 2

    # --- Alcohol-Density relationship (known inverse correlation) ---
    df["alcohol_density_product"] = df["alcohol"] * df["density"]
    df["alcohol_minus_density"] = df["alcohol"] - (df["density"] * 1000)
    df["alcohol_sq"] = df["alcohol"] ** 2
    df["alcohol_cu"] = df["alcohol"] ** 3

    # --- Sugar-Density interaction ---
    df["sugar_density_ratio"] = df["residual_sugar"] / (df["density"] + 1e-8)
    df["sugar_per_alcohol"] = df["residual_sugar"] / (df["alcohol"] + 1e-8)

    # --- Chlorides interactions ---
    df["chlorides_per_so2"] = df["chlorides"] / (df["total_sulfur_dioxide"] + 1e-8)
    df["chlorides_sq"] = df["chlorides"] ** 2

    # --- pH interactions ---
    df["ph_sq"] = df["pH"] ** 2
    df["ph_alcohol"] = df["pH"] * df["alcohol"]
    df["ph_volatile_acid"] = df["pH"] * df["volatile_acidity"]

    # --- Sulphates interactions ---
    df["sulphates_per_alcohol"] = df["sulphates"] / (df["alcohol"] + 1e-8)
    df["sulphates_sq"] = df["sulphates"] ** 2

    # --- Log transforms for right-skewed features ---
    df["log_residual_sugar"] = np.log1p(df["residual_sugar"])
    df["log_chlorides"] = np.log1p(df["chlorides"])
    df["log_free_so2"] = np.log1p(df["free_sulfur_dioxide"])
    df["log_total_so2"] = np.log1p(df["total_sulfur_dioxide"])
    df["log_volatile_acidity"] = np.log1p(df["volatile_acidity"])

    # --- Density normalization (density is ~0.99-1.0, normalize to useful range) ---
    df["density_scaled"] = (df["density"] - 0.99) * 1000

    # --- Composite quality indicators ---
    df["quality_proxy_1"] = (
        df["alcohol"] - 2 * df["volatile_acidity"] + 0.5 * df["sulphates"]
    )
    # NOTE: quality_proxy_2 requires a global max; handled separately after split

    return df


# ============================================================
# 4. TRAIN/VALIDATION SPLIT (Stratified by quality)
# ============================================================
train_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["quality"]
)

print(f"\nTrain split shape: {train_df.shape}")
print(f"Validation split shape: {val_df.shape}")
print(
    f"Train quality dist: {train_df['quality'].value_counts().sort_index().to_dict()}"
)
print(f"Val quality dist: {val_df['quality'].value_counts().sort_index().to_dict()}")

# Apply feature engineering AFTER split to avoid leakage
vol_max_train = train_df["volatile_acidity"].max()

def engineer_features(df, vol_max):
    """Apply domain-informed feature engineering for wine quality prediction."""
    df = df.copy()

    # --- Sulfur dioxide family interactions ---
    df["free_so2_ratio"] = df["free_sulfur_dioxide"] / (
        df["total_sulfur_dioxide"] + 1e-8
    )
    df["bound_so2"] = df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"]
    df["so2_per_alcohol"] = df["total_sulfur_dioxide"] / (df["alcohol"] + 1e-8)

    # --- Acidity family interactions ---
    df["volatile_to_fixed_ratio"] = df["volatile_acidity"] / (
        df["fixed_acidity"] + 1e-8
    )
    df["citric_to_fixed_ratio"] = df["citric_acid"] / (
        df["fixed_acidity"] + 1e-8
    )
    df["total_acidity"] = (
        df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    )
    df["volatile_acidity_sq"] = df["volatile_acidity"] ** 2

    # --- Alcohol-Density relationship (known inverse correlation) ---
    df["alcohol_density_product"] = df["alcohol"] * df["density"]
    df["alcohol_minus_density"] = df["alcohol"] - (df["density"] * 1000)
    df["alcohol_sq"] = df["alcohol"] ** 2
    df["alcohol_cu"] = df["alcohol"] ** 3

    # --- Sugar-Density interaction ---
    df["sugar_density_ratio"] = df["residual_sugar"] / (df["density"] + 1e-8)
    df["sugar_per_alcohol"] = df["residual_sugar"] / (df["alcohol"] + 1e-8)

    # --- Chlorides interactions ---
    df["chlorides_per_so2"] = df["chlorides"] / (df["total_sulfur_dioxide"] + 1e-8)
    df["chlorides_sq"] = df["chlorides"] ** 2

    # --- pH interactions ---
    df["ph_sq"] = df["pH"] ** 2
    df["ph_alcohol"] = df["pH"] * df["alcohol"]
    df["ph_volatile_acid"] = df["pH"] * df["volatile_acidity"]

    # --- Sulphates interactions ---
    df["sulphates_per_alcohol"] = df["sulphates"] / (df["alcohol"] + 1e-8)
    df["sulphates_sq"] = df["sulphates"] ** 2

    # --- Log transforms for right-skewed features ---
    df["log_residual_sugar"] = np.log1p(df["residual_sugar"])
    df["log_chlorides"] = np.log1p(df["chlorides"])
    df["log_free_so2"] = np.log1p(df["free_sulfur_dioxide"])
    df["log_total_so2"] = np.log1p(df["total_sulfur_dioxide"])
    df["log_volatile_acidity"] = np.log1p(df["volatile_acidity"])

    # --- Density normalization (density is ~0.99-1.0, normalize to useful range) ---
    df["density_scaled"] = (df["density"] - 0.99) * 1000

    # --- Composite quality indicators ---
    df["quality_proxy_1"] = (
        df["alcohol"] - 2 * df["volatile_acidity"] + 0.5 * df["sulphates"]
    )
    df["quality_proxy_2"] = df["alcohol"] * (
        1 - df["volatile_acidity"] / (vol_max + 1e-8)
    )

    return df


# Apply feature engineering with train-only max
train_df = engineer_features(train_df, vol_max_train)
val_df = engineer_features(val_df, vol_max_train)
test_df = engineer_features(test_df, vol_max_train)

# ============================================================
# 3. DEFINE FEATURE COLUMNS
# ============================================================
feature_cols = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    # Engineered features
    "free_so2_ratio",
    "bound_so2",
    "so2_per_alcohol",
    "volatile_to_fixed_ratio",
    "citric_to_fixed_ratio",
    "total_acidity",
    "volatile_acidity_sq",
    "alcohol_density_product",
    "alcohol_minus_density",
    "alcohol_sq",
    "alcohol_cu",
    "sugar_density_ratio",
    "sugar_per_alcohol",
    "chlorides_per_so2",
    "chlorides_sq",
    "ph_sq",
    "ph_alcohol",
    "ph_volatile_acid",
    "sulphates_per_alcohol",
    "sulphates_sq",
    "log_residual_sugar",
    "log_chlorides",
    "log_free_so2",
    "log_total_so2",
    "log_volatile_acidity",
    "density_scaled",
    "quality_proxy_1",
    "quality_proxy_2",
]

print(f"\nTotal features: {len(feature_cols)}")

# ============================================================
# 5. SCALE FEATURES (Fit on train only - prevent leakage)
# ============================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(train_df[feature_cols].values)
X_val = scaler.transform(val_df[feature_cols].values)
X_test = scaler.transform(test_df[feature_cols].values)

y_train = train_df["quality"].values
y_val = val_df["quality"].values

# Handle any potential inf/nan from division
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nX_train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
print(f"X_val stats: mean={X_val.mean():.4f}, std={X_val.std():.4f}")
print(f"X_test stats: mean={X_test.mean():.4f}, std={X_test.std():.4f}")

test_ids = np.arange(len(test_df))

print(f"\n✅ Data processing and feature engineering complete.")
print(f"   Features: {len(feature_cols)}")
print(f"   Train samples: {X_train.shape[0]}")
print(f"   Val samples: {X_val.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Target range: [{y_train.min()}, {y_train.max()}]")
print(f"   Target mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

# ============================================================
# 6. MULTI-SEED ENSEMBLE TRAINING
# ============================================================
# Strategy: Train multiple LightGBM models with different random seeds
# and slightly varied hyperparameters to create diverse predictions.
# Averaging reduces variance and smooths predictions for the discrete target.

# Base hyperparameters (tuned for small tabular data)
base_params = {
    "objective": "huber",
    "alpha": 0.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_gain_to_split": 0.05,
    "verbose": -1,
}

# Define multiple configurations for diversity
configs = [
    # Config 0: Base
    {
        "seed": 42,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "learning_rate": 0.02,
    },
    # Config 1: Slightly more leaves, less regularization
    {
        "seed": 123,
        "num_leaves": 40,
        "max_depth": 7,
        "min_data_in_leaf": 15,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "learning_rate": 0.02,
    },
    # Config 2: More regularized, fewer leaves
    {
        "seed": 7,
        "num_leaves": 24,
        "max_depth": 5,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "learning_rate": 0.015,
    },
    # Config 3: Different split, more aggressive
    {
        "seed": 2024,
        "num_leaves": 35,
        "max_depth": 6,
        "min_data_in_leaf": 18,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "learning_rate": 0.025,
    },
    # Config 4: Conservative
    {
        "seed": 999,
        "num_leaves": 28,
        "max_depth": 5,
        "min_data_in_leaf": 22,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "learning_rate": 0.018,
    },
    # Config 5: Another variation
    {
        "seed": 555,
        "num_leaves": 32,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "learning_rate": 0.022,
    },
]

num_models = len(configs)
print(f"\nTraining {num_models} LightGBM models with diverse configurations...")

# Store all models and their predictions
models = []
val_preds_list = []
test_preds_list = []
seed_rmses = []

for i, config in enumerate(configs):
    # Build params for this config
    params = base_params.copy()
    params.update(config)

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name="auto")
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name="auto", reference=train_data
    )

    # Train with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[val_data],
        num_boost_round=5000,
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    models.append(model)

    # Predictions
    val_pred = model.predict(X_val, num_iteration=best_iter)
    test_pred = model.predict(X_test, num_iteration=best_iter)

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

    seed_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    seed_rmses.append(seed_rmse)

    print(
        f"  Model {i}: seed={config['seed']}, best_iter={best_iter}, val_RMSE={seed_rmse:.4f}"
    )

# ============================================================
# 7. ENSEMBLE: Weighted average (weight by inverse RMSE)
# ============================================================
weights = np.array([1.0 / rmse for rmse in seed_rmses])
weights = weights / weights.sum()  # Normalize

print(f"\nEnsemble weights: {weights.round(4)}")

# Weighted ensemble predictions
val_preds_ensemble = np.average(val_preds_list, axis=0, weights=weights)
test_preds_ensemble = np.average(test_preds_list, axis=0, weights=weights)

# Also compute simple average for comparison
val_preds_simple = np.mean(val_preds_list, axis=0)
test_preds_simple = np.mean(test_preds_list, axis=0)

# ============================================================
# 8. EVALUATION
# ============================================================
print(f"\nIndividual model RMSEs: {[f'{r:.4f}' for r in seed_rmses]}")
print(f"Mean individual RMSE: {np.mean(seed_rmses):.4f}")
print(f"Std individual RMSE:  {np.std(seed_rmses):.4f}")

val_rmse_weighted = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))
val_rmse_simple = np.sqrt(mean_squared_error(y_val, val_preds_simple))

print(f"\nEnsemble RMSE (weighted): {val_rmse_weighted:.4f}")
print(f"Ensemble RMSE (simple):   {val_rmse_simple:.4f}")

# Use the better ensemble
if val_rmse_weighted <= val_rmse_simple:
    final_val_preds = val_preds_ensemble
    final_test_preds = test_preds_ensemble
    ensemble_type = "weighted"
else:
    final_val_preds = val_preds_simple
    final_test_preds = test_preds_simple
    ensemble_type = "simple"

print(f"Using {ensemble_type} ensemble")

# ============================================================
# 9. POST-PROCESSING: Clip predictions to valid range
# ============================================================
q_min = y_train.min()
q_max = y_train.max()

final_val_preds = np.clip(final_val_preds, q_min, q_max)
final_test_preds = np.clip(final_test_preds, q_min, q_max)

# Recompute after clipping
val_rmse_final = np.sqrt(mean_squared_error(y_val, final_val_preds))
print(f"Final validation RMSE (after clipping): {val_rmse_final:.4f}")

# ============================================================
# 10. FEATURE IMPORTANCE (from first model for insight)
# ============================================================
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({"feature": feature_cols, "gain": importance}).sort_values(
    "gain", ascending=False
)
print(f"\nTop 10 features by gain (model 0):")
print(feat_imp.head(10).to_string(index=False))

# ============================================================
# 11. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

submission = pd.DataFrame({"id": test_ids.astype(int), "quality": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction range: [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")
print(
    f"Prediction mean: {final_test_preds.mean():.4f}, std: {final_test_preds.std():.4f}"
)
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 12. FINAL VALIDATION SCORE (Official metric: RMSE)
# ============================================================
score = val_rmse_final
print(f"\n{'='*50}")
print(f"VALIDATION SUMMARY")
print(f"{'='*50}")
print(f"Split: 85/15 stratified by quality")
print(f"Metric: RMSE (official)")
print(f"Model: {num_models}-model LightGBM ensemble ({ensemble_type} averaging)")
print(f"Individual RMSEs: {seed_rmses}")
print(f"Ensemble improvement: {np.mean(seed_rmses) - val_rmse_final:.4f}")
print(f"{'='*50}")
print(f"Final Validation Score: {score}")