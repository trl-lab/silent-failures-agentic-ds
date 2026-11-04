import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import lightgbm as lgbm

# Define paths as specified in the prompt
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

# Load data
print("Loading train data...")
train_df = pd.read_parquet(TRAIN_PATH)
print("Loading test data...")
test_df = pd.read_parquet(TEST_PATH)

# Identify columns
target_col = 'Cover_Type'
X_cols = [col for col in train_df.columns if col != target_col]
y = train_df[target_col].values
X_train = train_df[X_cols].copy()
X_test = test_df[X_cols].copy().copy()

# Separate categorical and numerical columns for preprocessing
# We assume object/string types are categorical
numeric_features = []
categorical_features = []

for col in X_cols:
    if train_df[col].dtype == 'object':
        categorical_features.append(col)
    else:
        numeric_features.append(col)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create model pipeline
model = lgbm.LGBMClassifier(n_estimators=500, random_state=42)

# Create combined pipeline
clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])

# Fit model
print("Training model...")
clf.fit(X_train, y)

# Predict on test set
print("Predicting...")
y_pred = clf.predict(X_test)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'Cover_Type': y_pred
})

# Save predictions
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")