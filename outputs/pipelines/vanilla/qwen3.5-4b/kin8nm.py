import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# Define file paths relative to the current working directory
train_path = 'train.parquet'
test_path = 'test.parquet'
output_path = 'predictions.csv'
target_col = 'y'

# Load data from parquet files
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify feature columns (exclude target)
feature_cols = [col for col in df_train.columns if col != target_col]

# Prepare dataframes
X_train = df_train[feature_cols]
y_train = df_train[target_col]
X_test = df_test[feature_cols]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_processed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_processed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

# Handle categorical columns automatically
# While specifications indicate 0 nominal columns, ensure robustness against object dtypes
for col in X_train_processed.columns:
    if X_train_processed[col].dtype == object:
        X_train_processed[col] = X_train_processed[col].astype('category')
        X_test_processed[col] = X_test_processed[col].astype('category')

# Train LightGBM Regressor
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train_processed, y_train.values)

# Generate predictions
y_pred = model.predict(X_test_processed)

# Create output dataframe with id and y
predictions_df = pd.DataFrame({
    'id': range(len(X_test)),
    'y': y_pred
})

# Save predictions to CSV
predictions_df.to_csv(output_path, index=False)