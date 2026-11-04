import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

# Define paths as per instructions
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load data
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

# Identify target and features
target_col = 'class'
X_train_raw = train_df.drop(columns=[target_col], errors='ignore')
y_train = train_df[target_col]

X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

# Ensure test columns match train columns (order matters for ColumnTransformer)
X_test_raw = X_test_raw[X_train_raw.columns]

# Separate numeric and categorical columns
# Numeric: np.number
# Categorical: Everything else (Object, Int, Float, String)
numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in X_train_raw.columns if col not in numeric_cols]

# Define Preprocessors
# 1. Imputation and Scaling for Numeric
num_imputer = SimpleImputer(strategy='median')
num_scaler = StandardScaler()
num_transformer = Pipeline(steps=[
    ('imputer', num_imputer),
    ('scaler', num_scaler)
])

# 2. Imputation and Encoding for Categorical
# Missing values replaced by 'most_frequent' (mode) to preserve category meaning
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories='auto')
cat_transformer = Pipeline(steps=[
    ('imputer', cat_imputer),
    ('encoder', cat_encoder)
])

# Combine transformers
ct = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Train pipeline
print("Training model...")
# We use LightGBM classifier
clf = lgb.LGBMClassifier(
    n_estimators=500, 
    max_depth=5, 
    learning_rate=0.1, 
    random_state=42, 
    verbosity=-1
)

# Prepare data for pipeline
try:
    X_train_processed = ct.fit_transform(X_train_raw)
    X_test_processed = ct.transform(X_test_raw)
    
    # If categorical columns were encoded, ensure target is compatible with model
    # LGBMClassifier handles numeric targets generally (0 or 1)
    # We keep y_train as is
    
    clf.fit(X_train_processed, y_train)
except Exception as e:
    print(f"Error during pipeline fitting: {e}")
    exit(1)

# Generate predictions
y_pred = clf.predict(X_test_processed)

# Prepare output dataframe
output_df = pd.DataFrame({
    'id': np.arange(len(y_pred)),
    'class': y_pred
})

# Save predictions
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Predictions saved to {OUTPUT_PATH}")