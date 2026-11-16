import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define file paths as specified in the prompt context
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/metro-interstate-traffic-volume/data/test.parquet'

# Load data
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Define target column
target_col = 'traffic_volume'

# Feature columns (exclude target)
all_columns = df_train.columns.tolist()
feature_columns = [col for col in all_columns if col != target_col]

# Prepare dataframes
X_train = df_train[feature_columns].copy()
X_test = df_test[feature_columns].copy()

# Identify numeric and categorical columns
# This helps determine how to handle each during preprocessing
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

# Extract hour from date_time if present and usable
# Attempt to create hour features from the date_time column if available
if 'date_time' in feature_columns:
    # Apply conversion to datetime; if successful, extract hour
    try:
        X_train['date_time_parsed'] = pd.to_datetime(X_train['date_time'], errors='coerce')
        X_test['date_time_parsed'] = pd.to_datetime(X_test['date_time'], errors='coerce')
        
        # Create hour feature
        X_train['hour'] = X_train['date_time_parsed'].apply(lambda x: x.hour if isinstance(x, pd.Timestamp) else 0)
        X_test['hour'] = X_test['date_time_parsed'].apply(lambda x: x.hour if isinstance(x, pd.Timestamp) else 0)
        
        # Add hour to numeric columns if it wasn't created yet (since it's an integer derived)
        if 'hour' not in num_cols:
            num_cols.append('hour')
            
        # Re-check num_cols list to ensure it reflects current structure if hour was added dynamically
        # However, since we appended to list, we need to update the reference
        # For simplicity in this pipeline, we can re-select based on current columns or manage list carefully.
        # Updating the list to include 'hour' if added.
        # Actually, X_train['hour'] adds to existing rows. We need to identify column index or rebuild num list.
        # Safest: Update num_cols to reflect addition.
        if 'hour' not in X_train.columns:
             pass # Fallback if date_time didn't become datetime
            
    except Exception:
        pass # If conversion fails, rely on original columns

# Correct num_cols to ensure 'hour' (if created) is included if applicable
if 'hour' in X_train.columns:
    # Re-run selection to include new column properly if it became numeric
    # In the loop above we ensured it exists.
    # We will use the updated lists below based on dynamic check
    
# Ensure numeric columns list is correct based on actual content of X_train currently
# Since we may have added 'hour' via assignment above
if 'hour' in feature_columns:
    # Force inclusion
    num_cols = [c for c in X_train.columns if X_train[c].dtype == 'int64' or X_train[c].dtype == 'float64']
    # Add other types (int8, int16 etc) if pandas version variances occur
    num_cols.extend([c for c in X_train.columns if not c in cat_cols and c != target_col]) 
    
# Reset and clean num/cat classification to ensure consistency
# Reclassify based on current DataFrame
final_num_cols = []
final_cat_cols = []
for col in feature_columns:
    dtype = X_train[col].dtype
    if hasattr(dtype, 'kind'):
        if dtype.kind in ['i', 'f']: # int, float
            final_num_cols.append(col)
        elif dtype.kind in ['O']: # object (often string cats)
            final_cat_cols.append(col)
        else:
            # Check if categorical
            final_cat_cols.append(col)
    else:
        final_cat_cols.append(col)

# Impute Numerical Missing Values
imputer = SimpleImputer(strategy='median')
X_train_np = imputer.fit_transform(X_train[final_num_cols])
X_test_np = imputer.transform(X_test[final_num_cols])

# One-Hot Encode Categorical Columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[final_cat_cols])
X_test_cat = encoder.transform(X_test[final_cat_cols])

# Combine Features
X_train_combined = np.hstack([X_train_np, X_train_cat])
X_test_combined = np.hstack([X_test_np, X_test_cat])

# Target Variable
y_train = df_train[target_col].values

# Train LightGBM Model
# Using parameters suitable for regression task
model = lgb.LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=-1, 
    num_leaves=31,
    random_state=42,
    silent=True
)

# Scale inputs to list format if needed (LGBM takes 2D array)
model.fit(X_train_combined, y_train)

# Generate Predictions
predictions = model.predict(X_test_combined)

# Prepare Output Dataframe
results_df = pd.DataFrame({
    'id': range(len(df_test)),
    'traffic_volume': predictions
})

# Save to CSV
results_df.to_csv('predictions.csv', index=False)