import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Define file paths as specified in the prompt
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/test.parquet'
output_file = 'predictions.csv'

# Load data from parquet files
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify columns
target_col = 'Diagnosis'
id_col = 'ID'
feature_cols = [col for col in train_df.columns if col not in [id_col, target_col]]

# Verify feature columns exist in test set
test_feature_cols = [col for col in test_df.columns if col not in [id_col, target_col]]
assert feature_cols == test_feature_cols, "Feature columns mismatch between train and test."

# Prepare data
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train_raw = train_df[target_col].values
y_test_raw = test_df[target_col].values

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)

# Train XGBoost model
model = xgb.XGBClassifier(eval_metric='auc', random_state=42)
model.fit(X_train_imputed, y_train_encoded)

# Generate predictions
# model.predict returns class labels (integers corresponding to encoder classes)
predicted_classes_idx = model.predict(X_test_imputed)
predicted_classes_str = label_encoder.inverse_transform(predicted_classes_idx)

# Create output dataframe
output_df = pd.DataFrame({
    'id': np.arange(len(predicted_classes_str)),
    'Diagnosis': predicted_classes_str
})

# Save to CSV
output_df.to_csv(output_file, index=False)