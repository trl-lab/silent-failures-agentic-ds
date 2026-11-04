import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
train = pd.read_parquet("./input/train.parquet")
test  = pd.read_parquet("./input/test.parquet")

target_col = "IMU Score"
y = train[target_col].values
X_full = train.drop(columns=[target_col])

# Keep only numeric feature columns present in both sets
num_cols = [c for c in X_full.select_dtypes(include=[np.number]).columns
            if c in test.columns]
X_full = X_full[num_cols]
X_test = test[num_cols]

# ── 2. Split FIRST, then preprocess (no leakage) ─────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Impute using train-only statistics
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_val   = X_val.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

# Scale using train-only statistics
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── 3. Train model ───────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
model.fit(X_train_s, y_train)

# ── 4. Validate with official metric (RMSE) ──────────────────────────────────
val_pred = model.predict(X_val_s)
rmse     = float(np.sqrt(mean_squared_error(y_val, val_pred)))
print(f"Final Validation Score: {rmse:.6f}")

# ── 5. Predict on test & write submission ────────────────────────────────────
test_pred = model.predict(X_test_s)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({
    "id":        np.arange(len(test_pred), dtype=int),
    "IMU Score": test_pred
})
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")