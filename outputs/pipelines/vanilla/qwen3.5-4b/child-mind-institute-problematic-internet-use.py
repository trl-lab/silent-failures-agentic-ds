import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define file paths as per specification
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'sii'

# Separate features and target
feature_cols = [col for col in train_df.columns if col != target_col]
y_train = train_df[target_col].values
X_train_raw = train_df[feature_cols]
X_test_raw = test_df[feature_cols]

# Identify categorical and numerical columns
numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()

# Initialize transformers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', num_imputer),
            ('encoder', None) # We will stack numbers directly
        ])),
        ('cat', Pipeline(steps=[
            ('imputer', cat_imputer),
            ('encoder', onehot_encoder)
        ]))
    ],
    remainder='passthrough' # Catch leftovers if any (should not be used with fixed list above)
)

# Better approach: Split into parts manually to avoid pipeline complexity with ColumnTransformer
# We need to ensure consistent shape

# Preprocess training data
X_train_num = num_imputer.fit_transform(X_train_raw[numeric_cols])
X_train_cat_encoded = onehot_encoder.fit_transform(X_train_raw[categorical_cols])

# Combine
X_train_final = np.hstack([X_train_num, X_train_cat_encoded])

# Preprocess testing data
X_test_num = num_imputer.transform(X_test_raw[numeric_cols])
X_test_cat_encoded = onehot_encoder.transform(X_test_raw[categorical_cols])

# Combine
X_test_final = np.hstack([X_test_num, X_test_cat_encoded])

# Train LightGBM model
model = lgb.LGBMClassifier(
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    n_estimators=500,
    random_state=42,
    verbosity=-1
)

model.fit(X_train_final, y_train)

# Make predictions
y_prob = model.predict_proba(X_test_final)[:, 1]
# Binarize predictions based on threshold 0.5
y_pred_binary = (y_prob >= 0.5).astype(int)

# Create output dataframe with 'id' and 'sii'
result_df = pd.DataFrame({
    'id': range(len(X_test_final)),
    'sii': y_pred_binary
})

# Save predictions to CSV
result_df.to_csv('predictions.csv', index=False)