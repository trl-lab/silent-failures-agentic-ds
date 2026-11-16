import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss

# Define file paths as specified
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify columns
target_col = 'Category'
feature_cols = [col for col in train_df.columns if col != target_col]

# Split features and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].values

X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Identify categorical columns (string or category dtype)
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Perform One-Hot Encoding on categorical columns
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    # Use get_dummies to avoid conflicts and handle missing strings cleanly
    # We iterate only to avoid duplicating columns, using transform logic effectively via assignment
    pass

# Convert categorical columns to dummy variables
# This is done efficiently to ensure compatibility with LightGBM which prefers dense arrays
X_train_obj = X_train[categorical_cols].copy()
X_test_obj = X_test[categorical_cols].copy()

# Create encoder for categorical columns dynamically
try:
    # Try sklearn approach
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    # Initialize Encoder
    transformer = ColumnTransformer([('categorical', 'passthrough', categorical_cols)])
    # However, standard fit is better
    
    # Simpler approach: Direct Dummy transformation
    # We reconstruct the dataframe ensuring all numeric/float remain and cat become dummies
    # First, split train into cat and num parts
    X_train_cat = X_train[categorical_cols].copy()
    X_train_num = X_train[~X_train.columns.intersection(categorical_cols)].copy()
    
    X_test_cat = X_test[categorical_cols].copy()
    X_test_num = X_test[~X_test.columns.intersection(categorical_cols)].copy()
    
    # One Hot Encode Categoricals
    # Using pandas get_dummies
    X_train_cats_encoded = pd.get_dummies(X_train_cat, drop_first=True)
    X_test_cats_encoded = pd.get_dummies(X_test_cat, drop_first=True)
    
    # Combine Numerics and Encoded Categoricals
    X_train_processed = pd.concat([X_train_num, X_train_cats_encoded], axis=1)
    X_test_processed = pd.concat([X_test_num, X_test_cats_encoded], axis=1)
    
except Exception:
    # Fallback if get_dummies encounters issues (rare)
    X_train_processed = X_train
    X_test_processed = X_test

# Create and Train Model
model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=10000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

# Fit model
model.fit(X_train_processed, y_train)

# Predict
predictions_proba = model.predict_proba(X_test_processed)
# Log-Loss calculation is automatic for evaluation, here we need category labels
# Get class probabilities to match class names
classes = model.classes_
predicted_classes = model.predict(X_test_processed)

# Construct output dataframe
# id starts from 0, corresponding to row index in test set
result_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Category': predicted_classes.astype(str)
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)