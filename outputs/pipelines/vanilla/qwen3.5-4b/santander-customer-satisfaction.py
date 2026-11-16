import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

# Configuration
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/santander-customer-satisfaction/data/test.parquet"
OUTPUT_FILE = "predictions.csv"

# Load Data
print("Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Ensure target column exists
target_col = "TARGET"
# Filter out rows if necessary, but assuming valid parquet files
y_train = df_train[target_col]
X_train = df_train.drop(columns=[target_col])
X_test = df_test.copy()

# Function to preprocess features automatically
def prepare_features(train_df, test_df, target_col):
    """
    Handles missing values and categorical encoding automatically.
    Uses numeric columns with imputation and object columns with One-Hot Encoding.
    """
    # Define feature columns
    feature_columns = [col for col in df_train.columns if col != target_col]
    
    # Identify data types automatically
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(['object']).columns.tolist()
    
    print(f"Numeric features count: {len(numeric_features)}")
    print(f"Categorical features count: {len(categorical_features)}")
    
    # Preprocessor pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ], remainder='drop'
    )
    
    combined_transformer = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Transform Training Data
    X_train_processed = combined_transformer.fit_transform(X_train)
    # Transform Testing Data (using the fitted transformers)
    X_test_processed = combined_transformer.transform(X_test)
    
    return X_train_processed, X_test_processed

# Prepare Features
X_train_proc, X_test_proc = prepare_features(df_train, df_test, target_col)

# Initialize and Train Model
model = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)

print("Training model...")
model.fit(X_train_proc, y_train)

# Generate Predictions
proba_scores = model.predict_proba(X_test_proc)[:, 1]

# Create DataFrame with required format
results_df = pd.DataFrame({
    'id': range(len(X_test)),
    'TARGET': proba_scores
})

# Save to CSV
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")