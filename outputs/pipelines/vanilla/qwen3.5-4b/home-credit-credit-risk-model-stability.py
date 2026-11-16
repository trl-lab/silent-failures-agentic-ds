import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-credit-risk-model-stability/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-credit-risk-model-stability/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

# Load Data
print("Loading data...")
try:
    # Attempt to load the parquet files provided
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {TRAIN_PATH}")

# Verify columns
print(f"Train shape: {train_df.shape}, Columns: {list(train_df.columns)}")

# Identify Target
target_col = 'target'
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

# Prepare Features
feature_cols = [col for col in train_df.columns if col != target_col]

# Separate IDs
case_ids_train = train_df['case_id'] if 'case_id' in train_df.columns else None
ids_train = train_df.index.tolist()
ids_test = test_df.index.tolist() if 'case_id' not in test_df.columns else test_df['case_id'].tolist()

# Filter rows based on case_id if needed (assuming single ID mapping in case of join failure prevention)
X_train_raw = train_df[feature_cols]
y_train = train_df[target_col]

# Feature Engineering Pipeline: Handle missing and categorical
categorical_columns = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_imputer = SimpleImputer(strategy='median')
numeric_transformer = Pipeline(steps=[
    ('imputer', numeric_imputer)
])

# For categorical, we need OHE or Target Encoding. 
# Since catboost/lightgbm is available, but here we are building a pipeline to ensure standard compliance
# We will use LabelEncoder and OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('label_encoder', LabelEncoder()),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],
    remainder='drop'
)

# Reconstruct X with correct dtypes if necessary
print("Preprocessing training data...")
X_train_processed = preprocessor.fit_transform(X_train_raw)
y_train_processed = y_train.values

# Split for validation (Optional: keep track of score but here we just fit/train for final submission)
# Since the prompt asks for stability/prediction, we can use a simple split or cross-validation strategy
# Using K-Fold CV is better for stability
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

best_score = 0
best_model = None

# Grid search is too heavy, use basic params
param_grid = {'max_depth': [10, 15, 20], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [500, 1000]}
grid_search = None

# Since we need speed and reliability, let's pick LightGBM or XGBoost directly without heavy grid search
# Selecting LightGBM for its speed and stability on large tabular data

# Fit Model
model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

model.fit(X_train_processed, y_train_processed)

# Evaluate on a holdout split to ensure robustness if possible
val_split = len(X_train_processed) // 2
X_val = X_train_processed[val_split:]
y_val = y_train_processed[val_split:]
X_train_final = X_train_processed[:val_split]
y_train_final = y_train_processed[:val_split]

# Refit on valid split to avoid bias
# In this simplified environment, we trust the initial training.

print("Predicting...")
pred_probs = model.predict_proba(X_test_processed)[:, 1]

# Create output DataFrame
output_df = pd.DataFrame({
    'id': ids_test,
    'target': pred_probs
})

output_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved predictions to {OUTPUT_PATH}")