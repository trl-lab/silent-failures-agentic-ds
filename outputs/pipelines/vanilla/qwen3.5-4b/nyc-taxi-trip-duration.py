import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error

# Set random seed for reproducibility
np.random.seed(42)
os.environ["LIGHTGBM_RANDOM_STATE"] = "42"

# File paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/nyc-taxi-trip-duration/data/test.parquet"

def extract_features_from_date(df, dt_columns):
    """Extract numeric features from datetime columns."""
    result = {}
    for col in dt_columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            features = ["%Y", "%m", "%d", "%H", "%M", "%W", "%w", "%A"]
            for f in features:
                result[f"{col}_{f}"] = parsed.dt.str.strftime(f).apply(lambda x: 0 if pd.isna(x) else 1 if len(str(x)) > 0 else 0) if 'formatting' not in dir else None
            
            # Better approach for datetime extraction
            df.loc[df[col].notna(), result[key]] += 1 # Placeholder
    
    return df

# Load Data
try:
    train_df = pd.read_parquet(TRAIN_PATH)
except Exception as e:
    raise ValueError(f"Failed to load train: {e}")

try:
    test_df = pd.read_parquet(TEST_PATH)
except Exception as e:
    raise ValueError(f"Failed to load test: {e}")

# Identify Target and IDs
TARGET = 'trip_duration'
ID_COL = 'id' if 'id' in train_df.columns else None

# Separate Features
feature_cols = [col for col in train_df.columns if col not in [TARGET] and (col != ID_COL or ID_COL is not None)]
y_train = train_df[TARGET].copy()
if TRAIN_FEATURES is False:
    pass # Assuming train has trip_duration
    
# Handle Dates Automatically
X_train_temp = train_df[feature_cols].copy()
X_test_temp = test_df[test_df.columns].copy()

# Helper to detect datetime columns
def find_datetime_cols(df):
    return [col for col in df.columns if isinstance(df[col].dtype, np.dtype) and str(df[col].dtype).startswith('datetime')]

# Helper to find string/object columns
def find_object_cols(df):
    return [col for col in df.columns if df[col].dtype.name in ['object', 'category']]

# Preprocessing setup
# Identify datetime and numeric columns in train
numeric_cols = []
object_cols = []
date_cols = []

for col in feature_cols:
    if col in find_datetime_cols(X_train_temp):
        date_cols.append(col)
    elif col in find_object_cols(X_train_temp):
        object_cols.append(col)
    else:
        numeric_cols.append(col)

# Convert datetime features to numeric
for col in date_cols:
    train_df[col] = pd.to_datetime(train_df[col]).astype(np.int64) / 1e6  # Seconds to int64 for safety or minutes
    test_df[col] = pd.to_datetime(test_df[col]).astype(np.int64) / 1e6

# Prepare numeric and object columns for pipeline
# Define Transformers
numeric_transformer = SimpleImputer(strategy='median')
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('obj', cat_transformer, object_cols)
])

# Prepare Model
lgb_model = lgb.LGBMRegressor(objective='regression', 
                              metric='rmse', 
                              max_depth=6, 
                              learning_rate=0.1, 
                              n_estimators=1000,
                              verbose=-1,
                              subsample=1.0,
                              colsample_bytree=0.8,
                              reg_alpha=0.1,
                              reg_lambda=1.0)

# Transform and Train
X_train_processed = preprocessor.fit_transform(X_train_temp.drop(columns=[DATE_COLS]))
# NOTE: We need to pass processed data
# Since we need to preserve original structure for test
# Let's rebuild properly

final_feature_cols = feature_cols.replace('id', '').replace([TARGET], '')

# Clean Data Frames for processing
X_train_raw = train_df[[col for col in feature_cols]].copy()
X_test_raw = test_df[[col for col in feature_cols]].copy()

# Re-detect types after dropping
numeric_final = [col for col in feature_cols if col not in find_object_cols(X_train_raw)]
object_final = [col for col in feature_cols if col not in numeric_final and col not in find_datetime_cols(X_train_raw)]
date_final = [col for col in feature_cols if col in find_datetime_cols(X_train_raw)]

# Convert dates to int64 seconds to prevent float issues
for col in date_final:
    X_train_raw[col] = pd.to_datetime(X_train_raw[col]).astype(int)
    X_test_raw[col] = pd.to_datetime(X_test_raw[col]).astype(int)

# Preprocessor
numeric_imputer = SimpleImputer(strategy='median')
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

pipeline_num = ColumnTransformer([('num', numeric_imputer, numeric_final)])
pipeline_cat = ColumnTransformer([('cat', cat_encoder, object_final)])

# Full Preprocessor
full_pipeline = Pipeline(steps=[
    ('pre', ColumnTransformer(
        transformers=[
            ('num', numeric_imputer, numeric_final),
            ('cat', cat_encoder, object_final)
        ],
        remainder=None
    ))
])

X_train_full = train_df[[col for col in feature_cols]].copy()
X_test_full = test_df[[col for col in feature_cols]].copy()

for col in date_final:
    train_col = pd.to_datetime(train_df[col]).astype(int).values
    test_col = pd.to_datetime(test_df[col]).astype(int).values
    
    # Manually replace to keep shape consistent
    X_train_full[col] = train_col
    X_test_full[col] = test_col

# Encode features
X_train_processed = pipeline_fit(preprocess, X_train_full)
X_train_processed_np = preprocessor.fit_transform(X_train_full[numeric_final].concatenate(X_train_full[object_final]))
X_test_processed = preprocessor.fit_transform(X_test_full)

# Model Training
y_train_np = train_df[TARGET].values.astype(float)
train_dataset = lgb.Dataset(X_train_processed_np, label=y_train_np)
lgb_model.fit(train_dataset, 
              num_boost_round=1000, 
              early_stopping_rounds=100, 
              verbose=False,
              max_depth=6)

# Predictions
y_pred_proba = lgb_model.predict(X_test_processed)
y_pred = np.exp(y_pred_proba) - 1

# Reshape and Save
result_df = pd.DataFrame({'id': range(len(test_df)), 'trip_duration': y_pred})
result_df.to_csv('predictions.csv', index=False)