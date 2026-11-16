import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import catboost as cb

# Data paths
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Define target and features
target_col = 'Damage Description (total)'
feature_columns = [col for col in df_train.columns if col != target_col]

# Separate features and target
X_train = df_train[feature_columns]
y_train = df_train[target_col]
X_test = df_test[feature_columns]

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train model with CatBoost (handles missing values and categorical features automatically)
model = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=False,
    random_seed=42
)

model.fit(X_train, y_train_encoded)

# Generate predictions
y_pred_encoded = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Create output dataframe
result_df = pd.DataFrame({
    'id': list(range(len(df_test))),
    'Damage Description (total)': y_pred_labels
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)