import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify target and features
target_col = "usdprice"
feature_cols = [col for col in train.columns if col != target_col]

# Handle missing values
train[feature_cols] = train[feature_cols].fillna(train[feature_cols].median())
test[feature_cols] = test[feature_cols].fillna(train[feature_cols].median())

# Create features
X_train = train[feature_cols].values
y_train = train[target_col].values
X_test = test[feature_cols].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define LightGBM model with early stopping
model = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation for RMSE evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1}: RMSE = {rmse:.4f}")

print(f"CV RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

# Train final model on all data
model.fit(X_train_scaled, y_train, callbacks=[lgb.early_stopping(50, verbose=False)])

# Generate predictions
y_pred = model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(test)), "usdprice": y_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
