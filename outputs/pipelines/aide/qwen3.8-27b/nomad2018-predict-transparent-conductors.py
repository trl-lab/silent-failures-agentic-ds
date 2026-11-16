import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget stats:\n{train['formation_energy_ev_natom'].describe()}")

# Separate features and target
target_col = "formation_energy_ev_natom"
y = train[target_col].values
X = train.drop(columns=[target_col])
X_test = test.copy()

# Drop id column if present in features
if "id" in X.columns:
    id_col = X["id"]
    X = X.drop(columns=["id"])
    X_test = X_test.drop(columns=["id"])

if "id" in X_test.columns:
    test_ids = X_test["id"]
    X_test = X_test.drop(columns=["id"])
else:
    test_ids = np.arange(len(X_test))

# Handle any remaining non-numeric columns
for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.factorize(X[col])[0]
        X_test[col] = pd.factorize(X_test[col])[0]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")


# Define rmsle metric
def rmsle(y_true, y_pred):
    # Clip predictions to avoid log of non-positive values
    y_pred = np.clip(y_pred, 1e-10, None)
    return np.sqrt(mean_squared_error(np.log(y_true + 1e-10), np.log(y_pred + 1e-10)))


# LightGBM parameters - modest for small dataset
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

print("\n--- 5-Fold Cross-Validation ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_folds

    fold_rmsle = rmsle(y_val, val_pred)
    print(f"  Fold {fold+1}: best_iter={model.best_iteration}, rmsle={fold_rmsle:.6f}")

# Overall OOF rmsle
y_safe = np.clip(y, 1e-10, None)
oof_safe = np.clip(oof_preds, 1e-10, None)
overall_rmsle = np.sqrt(mean_squared_error(np.log(y_safe), np.log(oof_safe)))
print(f"\nOverall OOF RMSLE: {overall_rmsle:.6f}")

# Retrain on full data for final submission
dtrain_full = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=int(np.mean([500])),  # reasonable number of rounds
    callbacks=[lgb.log_evaluation(-1)],
)

final_preds = final_model.predict(X_test)

# Ensure predictions are positive for rmsle
final_preds = np.clip(final_preds, 1e-10, None)

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(final_preds)), "formation_energy_ev_natom": final_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction stats:\n{submission['formation_energy_ev_natom'].describe()}")
