import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Set working directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Load data
print("Loading data...")
train_data = pd.read_parquet("./input/train.parquet")
test_data = pd.read_parquet("./input/test.parquet")

# Feature engineering
print("Preprocessing data...")

# Prepare base features
train_X = train_data[
    [
        "longitude",
        "latitude",
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]
]
train_y = train_data["medianHouseValue"]

test_X = test_data[
    [
        "longitude",
        "latitude",
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]
]

# Log transformation
train_X = np.log1p(train_X)
test_X = np.log1p(test_X)

# Add interaction features
train_X["totalRooms_totalBedrooms"] = train_X["totalRooms"] * train_X["totalBedrooms"]
train_X["medianIncome_housingMedianAge"] = train_X["medianIncome"] * train_X["housingMedianAge"]

# Add polynomial features
train_X["totalRooms_sq"] = train_X["totalRooms"] ** 2
train_X["totalBedrooms_sq"] = train_X["totalBedrooms"] ** 2

# Add derived features
train_X["population_totalRooms"] = train_X["population"] / train_X["totalRooms"]
train_X["households_totalBedrooms"] = train_X["households"] / train_X["totalBedrooms"]

test_X["totalRooms_totalBedrooms"] = test_X["totalRooms"] * test_X["totalBedrooms"]
test_X["medianIncome_housingMedianAge"] = test_X["medianIncome"] * test_X["housingMedianAge"]
test_X["totalRooms_sq"] = test_X["totalRooms"] ** 2
test_X["totalBedrooms_sq"] = test_X["totalBedrooms"] ** 2
test_X["population_totalRooms"] = test_X["population"] / test_X["totalRooms"]
test_X["households_totalBedrooms"] = test_X["households"] / test_X["totalBedrooms"]

# Define all feature columns
feature_columns = [
    "longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms",
    "population", "households", "medianIncome",
    "totalRooms_totalBedrooms", "medianIncome_housingMedianAge",
    "totalRooms_sq", "totalBedrooms_sq",
    "population_totalRooms", "households_totalBedrooms"
]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_X[feature_columns], train_y, test_size=0.2, random_state=42
)

# Standardize features using train data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_X_scaled = scaler.transform(test_X[feature_columns])

# Save train features for reference
pd.DataFrame(X_train_scaled).to_csv("./working/train_features.csv", index=False)
print("Train features saved to working/train_features.csv")

# Hyperparameter grid search
print("Starting hyperparameter grid search...")
num_leaves_options = [15, 25, 31, 37, 43]
learning_rate_options = [0.01, 0.05, 0.1]
max_depth_options = [3, 4, 5]
n_estimators_options = [100, 200, 500]

base_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

results = []

for num_leaves in num_leaves_options:
    for learning_rate in learning_rate_options:
        for max_depth in max_depth_options:
            for n_estimators in n_estimators_options:
                params = base_params.copy()
                params["num_leaves"] = num_leaves
                params["learning_rate"] = learning_rate
                params["max_depth"] = max_depth
                params["n_estimators"] = n_estimators

                train_data_lgb = lgb.Dataset(X_train_scaled, label=y_train)
                val_data_lgb = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data_lgb)

                model = lgb.train(
                    params,
                    train_data_lgb,
                    num_boost_round=n_estimators,
                    valid_sets=[train_data_lgb, val_data_lgb],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )

                y_pred_val = model.predict(X_val_scaled, num_iteration=model.best_iteration)
                val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)

                results.append({
                    "num_leaves": num_leaves,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "best_iteration": model.best_iteration,
                    "val_rmse": val_rmse,
                    "model": model
                })
                print(f"Config: num_leaves={num_leaves}, lr={learning_rate}, max_depth={max_depth}, n_est={n_estimators}, val_rmse={val_rmse:.4f}")

# Sort by validation RMSE and select top 5
results.sort(key=lambda x: x["val_rmse"])
top_5 = results[:5]

print(f"\nTop 5 configurations by validation RMSE:")
for i, r in enumerate(top_5):
    print(f"{i+1}. val_rmse={r['val_rmse']:.4f}")

# Ensemble predictions
print("\nCreating ensemble predictions...")
ensemble_preds = []
for r in top_5:
    test_pred = r["model"].predict(test_X_scaled, num_iteration=r["best_iteration"])
    ensemble_preds.append(test_pred)

# Average predictions from top 5 models
final_test_pred = np.mean(ensemble_preds, axis=0)

# Validation metric for ensemble
ensemble_val_pred = np.mean([r["model"].predict(X_val_scaled, num_iteration=r["best_iteration"]) for r in top_5], axis=0)
ensemble_val_rmse = mean_squared_error(y_val, ensemble_val_pred, squared=False)
print(f"Ensemble Validation Score: {ensemble_val_rmse:.4f}")

# Test predictions
print("Making test predictions...")

# Create submission
submission = pd.DataFrame({
    "id": range(len(test_data)),
    "medianHouseValue": final_test_pred
})
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to submission/submission.csv")