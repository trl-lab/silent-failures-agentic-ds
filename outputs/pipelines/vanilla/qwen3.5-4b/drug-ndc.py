import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Imputer, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from pathlib import Path

# Set working environment variables
data_dir = "/path/to/user/workspace/ds_failures/datasets/strable/drug-ndc/data/"
train_path = Path(data_dir) / "train.parquet"
test_path = Path(data_dir) / "test.parquet"
output_path = "predictions.csv"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Prepare data
# Identify features and target
target_column = 'dea_schedule'
feature_columns = [col for col in test_df.columns if col != target_column]

# Handle missing values and categorical encoding
# Strategy: Use sklearn pipelines to ensure consistency between train and test
numeric_features = []
categorical_features = []

for col in feature_columns:
    dtype = train_df[col].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        numeric_features.append(col)
    else:
        categorical_features.append(col)

if len(feature_columns) == 0:
    raise ValueError("No feature columns found.")

preprocessor = ColumnTransformer(transformers=[
    ('num', Imputer(strategy='median'), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

target_preprocessor = LabelEncoder()

# Process Train Data
# Fit Preprocessors on train
X_train = train_df[feature_columns].copy()
X_train_encoded = preprocessor.fit_transform(X_train.astype(object))
y_train = train_df[target_column].values

# Process Test Data
# Transform Test using fitted preprocessors
X_test = test_df[feature_columns].copy()
X_test_encoded = preprocessor.transform(X_test.astype(object))
# Ensure y_test is handled later or predicted directly
ids = np.arange(len(test_df))

# Train Model
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

model.fit(X_train_encoded, y_train)

# Predict
proba_pred = model.predict_proba(X_test_encoded)
pred_indices = np.argmax(proba_pred, axis=1)
pred_labels = target_preprocessor.inverse_transform(pred_indices)

# Create Output DataFrame
results_df = pd.DataFrame({
    'id': ids,
    'dea_schedule': pred_labels
})

# Save to CSV
results_df.to_csv(output_path, index=False)