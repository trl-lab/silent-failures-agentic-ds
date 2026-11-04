import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import os

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# Step 1: Data Loading & Preprocessing
# ==============================================================================
print("Loading data...")

# Load datasets
train_data = pd.read_parquet("./input/train.parquet")
test_data = pd.read_parquet("./input/test.parquet")

# Identify target column
target_column = "Bus age"


# Preprocessing function
def preprocess_data(df, target_col):
    # Drop target column if it exists (only for train data)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Select features (exclude non-numeric columns if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols]

    # Handle missing values (simple mean fill for numeric)
    X = X.fillna(X.mean())

    return X


# Preprocess train and test
X_raw_train = preprocess_data(train_data, target_column)
X_raw_test = preprocess_data(test_data, target_column)

# Split train into train and validation FIRST (before any scaling)
X_train_full, X_val, y_train, y_val = train_test_split(
    X_raw_train, train_data[target_column], test_size=0.2, random_state=42
)

# Scale features only on training data to prevent leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_raw_test)


# ==============================================================================
# Step 2: Model Definition
# ==============================================================================
class LightGBMRegressor:
    def __init__(self, n_estimators=1000, learning_rate=0.05, verbose=0):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            verbose=verbose,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        # Use callbacks instead of early_stopping_rounds parameter
        # Need to provide eval_set and eval_metric for early stopping
        self.model.fit(X, y,
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      eval_set=[(X, y)],
                      eval_metric='rmse')
        return self

    def predict(self, X):
        return self.model.predict(X)


# ==============================================================================
# Step 3: Training & Evaluation
# ==============================================================================
print("Training model...")
model = LightGBMRegressor()
model.fit(X_train_scaled, y_train)

# ==============================================================================
# Step 4: Validation & Submission
# ==============================================================================
# Validation predictions
y_val_pred = model.predict(X_val_scaled)

# Calculate RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"Final Validation Score: {rmse:.6f}")

# Test predictions
y_test_pred = model.predict(X_test_scaled)

# Ensure submission format matches train target column
submission_df = pd.DataFrame({"Bus age": y_test_pred})

# Save submission
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")