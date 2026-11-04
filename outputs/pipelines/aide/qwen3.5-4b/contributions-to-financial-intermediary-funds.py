import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [col for col in train_df.columns if col not in ["id", "Amount in USD"]]
X_train = train_df[feature_cols].values
y_train = train_df["Amount in USD"].values
X_test = test_df[feature_cols].values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_score = np.inf
best_params = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # LightGBM parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, feature_name=feature_cols
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"Fold {fold+1}: RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_params = params.copy()

# Final model training on all training data
final_model = lgb.train(
    best_params,
    lgb.Dataset(X_train_scaled, label=y_train),
    num_boost_round=1000,
    verbose_eval=-1,
)

# Generate predictions for test set
test_pred = final_model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({"id": test_df["id"], "Amount in USD": test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Best 5-fold CV RMSE: {best_val_score:.4f}")
