import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

# Paths as specified in the prompt
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Identify features and target
target_col = 'Class'
feature_cols = [col for col in train_df.columns if col != target_col]

# Extract X and y
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Initialize Imputer for missing values (ShapeFactor etc.)
imputer = SimpleImputer(strategy='median')

# Apply imputer to both sets
X_train_processed = imputer.fit_transform(X_train)
X_test_processed = imputer.transform(X_test)

# Initialize and train LightGBM model
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    verbose=-1,
    random_state=42
)

model.fit(X_train_processed, y_train.values)

# Predict on test set
y_pred = model.predict(X_test_processed)

# Prepare output dataframe
# Ensure 'Class' values are consistent type (string) and 'id' starts from 0
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'Class': y_pred
})

# Save to CSV
output_df.to_csv(OUTPUT_PATH, index=False)