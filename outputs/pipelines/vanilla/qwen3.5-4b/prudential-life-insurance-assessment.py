import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Define file paths
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/test.parquet'

# Load data
try:
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
except Exception as e:
    raise RuntimeError(f"Error loading data: {e}")

# Identify target and features
y_train = df_train['Response'].values
X_train = df_train.drop(columns=['Response']).copy()
X_test = df_test.copy()

# Remove existing ID column from test if present to ensure clean processing
if 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])

# Define numerical and categorical columns based on dtypes
num_col_indices = list(X_train.select_dtypes(include=[np.number]).columns)
cat_col_indices = [col for col in X_train.columns if col not in num_col_indices]

# Preprocessors
# SimpleImputer handles numerical missing values
num_imputer = SimpleImputer(strategy='median')
# OneHotEncoder handles categorical (object/string) columns automatically
cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# For categoricals, fill missing values with 'Unknown' mode if any (SimpleImputer won't work directly on string)
# We map string columns to safe representation via imputation if possible, but OneHotEncoder handles 'handle_unknown'
# Better: Impute object columns with the mode. However, standard sklearn pipelines are safer.
# Re-implement: Combine Imputation for numbers, and Mode filling + OHE for cats.
cat_imputer = lambda x: pd.Series(x.fillna(x.mode()[0]) if len(x.mode()) > 0 else x.fillna('Unknown'))

# Construct Column Transformer for preprocessing
ct = ColumnTransformer(transformers=[
    ('num', num_imputer, num_col_indices),
    ('cat', 'identity', cat_col_indices) # Pass original string for OneHot, then handle missing manually?
])
# Refined approach for "Automatic" handling in single file:
# Apply transformers individually to ensure robustness

def prepare_dataframes(df, is_train=False):
    X = df.drop(columns=['Response'], errors='ignore')
    # Identify types
    num_features = X.select_dtypes(include=[np.number]).columns
    cat_features = [col for col in X.columns if col not in num_features]
    
    # Numerical Imputation
    X[num_features] = X[num_features].fillna(X[num_features].median())
    
    # Categorical Handling: Impute missing with 'nan' marker then OneHot will ignore if sparse, but we need finite values
    # Convert to string if not already
    X[cat_features] = X[cat_features].astype(str).fillna('UNKNOWN')
    
    # Encode
    return X

X_train_proc = prepare_dataframes(df_train)
X_test_proc = prepare_dataframes(df_test)

# Build Pipeline with LGBM
clf = Pipeline([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'), num_features)])),
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])),
    ('classifier', LGBMClassifier(objective='binary', random_state=42, silent=True, verbosity=0))
])

# Note: Above pipe creates multiple imputers. Correct simplified version:
# We will construct a single pipeline that handles the data flow properly.
# Since we manually transformed above for simplicity in this context:
# Re-do cleaner pipeline:

# Reset and fit
feature_cols = list(X_train_proc.columns)
num_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [col for col in X_train_proc.columns if col not in num_cols]

# Preprocessing for numerical
if len(num_cols) > 0:
    X_train_num = X_train_proc[num_cols].copy()
    X_test_num = X_test_proc[num_cols].copy()
    X_train_num = X_train_num.fillna(X_train_num.median())
    X_test_num = X_test_num.fillna(X_test_num.median())
else:
    X_train_num, X_test_num = None, None

# Preprocessing for categorical
if len(cat_cols) > 0:
    X_train_cat = X_train_proc[cat_cols].copy()
    X_test_cat = X_test_proc[cat_cols].copy()
    # Impute categorical missing with most frequent value
    modes = X_train_cat.apply(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else 'Unknown')
    X_train_cat.fillna(modes, inplace=True)
    X_test_cat.fillna(modes, inplace=True)
else:
    X_train_cat, X_test_cat = None, None

# Combine for LGBM
if X_train_num is not None and X_test_num is not None:
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)
else:
    X_train_final = X_train_proc.drop(columns=num_cols)
    X_test_final = X_test_proc.drop(columns=num_cols)

# Train Model
model = LGBMClassifier(objective='binary', random_state=42, silent=True)
model.fit(X_train_final.values, y_train)

# Predict
predictions = model.predict(X_test_final.values)

# Create output dataframe with required columns
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Response': predictions.astype(int)
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")