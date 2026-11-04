import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/test.parquet'
output_path = 'predictions.csv'

# Load datasets
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Identify target and feature columns
target_column = 'class'
feature_columns = [col for col in df_train.columns if col != target_column]

# Initialize transformers
le = LabelEncoder()
num_imputer = SimpleImputer(strategy='median')

# Extract target variables
y_train = df_train[target_column]
y_test = df_test[target_column]

# Encode target variable
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Extract features
X_train = df_train[feature_columns]
X_test = df_test[feature_columns]

# Convert to numeric to handle potential categorical columns in features
# While dataset is mostly numeric, ensure robustness by handling object types as specified
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        # Treat object columns by converting to numeric (mean replacement if fails)
        try:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        except:
            X_train[col] = 'nan'
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Impute missing values
X_train = num_imputer.fit_transform(X_train)
X_test = num_imputer.transform(X_test)

# Train model
model = XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.05, 
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False,
    tree_method='hist',
    verbosity=0
)

model.fit(X_train, y_train_enc)

# Predict
y_pred_enc = model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

# Prepare output dataframe with exact columns requested
result_df = pd.DataFrame({
    'id': range(len(df_test)),
    'class': y_pred
})

# Save predictions
result_df.to_csv(output_path, index=False)