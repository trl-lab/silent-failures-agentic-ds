import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/test.parquet')

# Define target column
target_col = 'LOANTOASSET'
y_train = train_df[target_col]
y_test = None

# Identify feature columns (exclude target)
feature_cols = [col for col in train_df.columns if col != target_col]
# Ensure test columns match train features for consistent preprocessing
test_features = [col for col in feature_cols if col in test_df.columns]
if not test_features:
    raise ValueError("No feature columns found in test set.")
# Add remaining test columns that might exist but were excluded due to missing target mapping, though ideally they align.
# If test has extra non-feature columns, exclude them.
# If test misses features present in train, that would be an error.
test_cols_used = [col for col in test_features if col in test_df.columns]

X_train = train_df[test_cols_used].copy()
X_test = test_df[test_cols_used].copy()

# Preprocessing: Fill missing values and Encode categorical
encoders = {}

def prepare_data(train, test, feat_cols):
    for col in feat_cols:
        original_dtype = train[col].dtype
        is_categorical = np.dtype(original_dtype).kind == 'O' or str(original_dtype) == 'object'
        
        # Impute Missing Values
        if is_categorical:
            # Fill categorical missing with most frequent value (Mode)
            # Ensure we don't crash if all NaN
            mode_val = train[col].mode().iloc[0] if not train[col].isna().all() else 0
            train[col] = train[col].fillna(mode_val)
            test[col] = test[col].fillna(mode_val)
            
            # Encode
            le = LabelEncoder()
            train[col] = train[col].astype(str) # Ensure string for label encoder
            train[col] = train[col].fillna(mode_val.astype(str))
            test[col] = test[col].fillna(mode_val.astype(str))
            train[col_encoded] = train[col].apply(lambda x: le.transform([x])[0]) if isinstance(le, LabelEncoder) else False
            encoders[col] = le
            
            # For numeric categorical, simple replacement
            # Using numeric fallback for non-unique strings
        else:
            # Numeric
            imputer = SimpleImputer(strategy='median')
            train[col] = imputer.fit_transform([[x] for x in train[col]])
            train[col] = train[col].reshape(-1, 1)
            test[col] = imputer.transform([[x] for x in test[col]])
            test[col] = test[col].reshape(-1, 1)

    return X_train, X_test

# Resetting logic for cleaner manual pipeline implementation for broader compatibility
# Re-initializing for clarity in a single script
X_final = train_df[feature_cols].copy()
X_test_final = test_df[feature_cols].copy()

processed_dict = {}

for col in feature_cols:
    # Determine type
    if X_final[col].dtype.kind == 'O':
        # Categorical/Object
        # Check for NA
        mode_value = X_final[col].mode().iloc[0] if not X_final[col].isna().all() else 'UNKNOWN'
        
        # Create Encoders dict per column
        # Note: Using a custom simple encode function to avoid sklearn dependency overhead for this specific logic
        unique_vals = X_final[col].astype(str).unique()
        
        # Create a lookup table for training data
        lookup_map = {val: i for i, val in enumerate(unique_vals)}
        
        # Update test data with same map
        # Replace values not in train with mode_value (assuming leakage prevention is handled or strict fit/transform)
        # Here, to simplify, we use the fit-transform pattern.
        # But since I need to manually manage state, let's do a simpler approach: 
        # 1. LabelEncode on Train.
        # 2. Same Mapping on Test.
        le = LabelEncoder()
        # Handle existing NA
        temp_train = X_final[col].dropna() if X_final[col].notna().any() else []
        # Better: Fill first
        fill_mode = X_final[col].mode().iloc[0] if not X_final[col].isna().all() else 0
        fill_str = str(fill_mode)
        
        X_final[col] = X_final[col].fillna(fill_str)
        X_test_final[col] = X_test_final[col].fillna(fill_str)
        
        # Label Encode
        # Ensure we cast to string to prevent int comparison issues
        temp_train_str = X_final[col].astype(str)
        temp_test_str = X_test_final[col].astype(str)
        
        # Re-initialize Label Encoder fresh for each column for correctness
        le_obj = LabelEncoder()
        le_obj.fit(temp_train_str)
        
        # Apply to Train
        X_final[str(col)] = le_obj.transform(temp_train_str).astype(int)
        
        # Apply to Test (handling unseen categories by mapping to mode if possible, or raising warning/silently handling)
        # In production, unseen categories in test should be mapped to a constant (e.g., -1) or re-fit.
        # For this script, we assume train/test schema consistency.
        try:
            X_test_final[str(col)] = le_obj.transform(temp_test_str).astype(int)
        except ValueError:
            # If unseen category exists in test, map it to the mode value index
            idx_fill = -1 # Default placeholder for unseen in this simplified script
            # Find index of mode in le_obj.classes_
            # This is messy. 
            # Safer: Re-create mapping based on Train only.
            X_test_final[col] = X_test_final[col].fillna(fill_str)
            # If unseen, treat as mode index (0 in this case if mode was first unique)
            # Simplified: Just rely on LabelEncoder fit_transform handling unseen as -1. 
            # We will replace -1 with 0 (representing mode/index 0) for safety
            transformed_test = le_obj.transform(temp_test_str)
            processed_list = list(transformed_test)
            X_test_final[col] = pd.DataFrame({col: [int(i) if i >= 0 else 0 for i in processed_list]})
            X_test_final = X_test_final.reset_index(drop=True) # Align indices
            
            break

# Correct Approach for Categorical Columns using sklearn pipeline components logic manually
X_processed = train_df[[col for col in feature_cols]].copy()
X_test_proc = test_df[[col for col in feature_cols]].copy()

# Define encoders storage
encoders = {}

for col in feature_cols:
    if X_processed[col].dtype in ['object', 'category']:
        # Categorial
        # Impute missing
        imp_mode = X_processed[col].mode().iloc[0] if not X_processed[col].isna().all() else 'UNK'
        X_processed[col] = X_processed[col].fillna(imp_mode)
        X_test_proc[col] = X_test_proc[col].fillna(imp_mode)
        
        # Encode
        le = LabelEncoder()
        le.fit(X_processed[col].astype(str))
        
        # Transforms
        # Convert to array to fit sklearn transform signature if needed, otherwise direct apply
        X_processed.loc[:, f'{col}_enc'] = le.transform(X_processed[col].astype(str))
        X_test_proc.loc[:, f'{col}_enc'] = le.transform(X_test_proc[col].astype(str))
        # Drop original columns
        X_processed.drop(columns=[col], inplace=True)
        X_test_proc.drop(columns=[col], inplace=True)
    elif X_processed[col].dtype.kind in ['i', 'f']:
        # Numeric
        imp_median = X_processed[col].median()
        X_processed[col].fillna(imp_median, inplace=True)
        X_test_proc[col].fillna(imp_median, inplace=True)

# Prepare DataFrames
X_train_clean = X_processed
X_test_clean = X_test_proc

# Ensure target is extracted
y_train = train_df[target_col]

# Train Model
model = lgb.LGBMRegressor(objective='regression', metric='rmse', num_leaves=31, learning_rate=0.05, random_state=42, n_estimators=1000, verbose=-1)
model.fit(X_train_clean, y_train)

# Predict
y_pred = model.predict(X_test_clean)

# Ensure outputs are float
y_pred = np.round(y_pred, 4).tolist()

# Create Output Frame
output_df = pd.DataFrame({
    'id': range(len(y_pred)),
    'LOANTOASSET': y_pred
})

# Save
output_df.to_csv('predictions.csv', index=False)