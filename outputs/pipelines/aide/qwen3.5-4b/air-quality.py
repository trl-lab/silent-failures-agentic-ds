import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target = "CO(GT)"
feature_cols = [
    "Date",
    "Time",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

# Encode categorical features
categorical_cols = [
    "Date",
    "Time",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]
ordinal_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    ordinal_encoders[col] = le

# Create feature matrix
X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create validation split (80/20)
from sklearn.model_selection import train_test_split

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=False
)

# Train LightGBM model
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

lgb_model.fit(
    X_train_final, y_train_final, eval_set=[(X_val, y_val)], callbacks=callbacks
)

# Calculate validation RMSE
val_pred = lgb_model.predict(X_val)
val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
print(f"Validation RMSE: {val_rmse:.4f}")

# Make predictions on test set
test_pred = lgb_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "CO(GT)": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
