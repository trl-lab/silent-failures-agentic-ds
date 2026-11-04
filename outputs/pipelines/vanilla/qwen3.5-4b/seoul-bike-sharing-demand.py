import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/test.parquet'
OUTPUT_CSV = 'predictions.csv'

def main():
    # 1. Load Data
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    
    # 2. Prepare Variables
    target_col = 'Rented Bike Count'
    
    # Determine feature columns
    # Based on table: Exclude 'Rented Bike Count' (Target)
    # Note: 'Functioning Day' is listed as Target in table but text says Target is Rented Bike Count. 
    # We will treat 'Functioning Day' as feature/exclude from output prediction logic to be safe, 
    # but primarily exclude 'Rented Bike Count'.
    exclude_features = ['Rented Bike Count', 'Functioning Day']
    
    # Get common columns (intersection of train/test to ensure alignment if schema differs)
    all_columns = sorted(set(df_train.columns) | set(df_test.columns))
    
    feature_list = [col for col in all_columns if col != target_col]
    
    # 3. Preprocessing Function
    def preprocess_dataframe(df, feature_list):
        df_processed = df.copy()
        
        # Handle Date: Try to parse, extract Month, Day of Week, Year
        if 'Date' in feature_list:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
            
            # Add extracted features
            # Only if not already present in feature_list (they are strings initially)
            date_extracts = ['Month', 'DayOfWeek', 'Year', 'Quarter']
            for col in date_extracts:
                df_processed[col] = df_processed['Date'].dt[col]
            
            # Remove original Date column after extracting info
            df_processed.drop(columns=['Date'], inplace=True, errors='ignore')
        
        # Separate Numerical and Categorical columns based on types
        # We want to one-hot encode string/object columns
        
        numeric_cols = []
        cat_cols = []
        
        for col in feature_list:
            if col in df_processed.columns:
                dtype = df_processed[col].dtype
                
                # Skip already extracted date features (Numeric)
                if col == 'Date': continue # Already removed
            
                # Logic to identify categorical vs numeric
                # Pandas 'category' counts as categorical usually
                # If object string, assume categorical (e.g., Seasons, Holiday status strings)
                
                try:
                    # Try to convert to numeric for math columns, keep as object for string columns
                    pd_numeric = pd.api.types.is_numeric_dtype(df_processed[col])
                    pd_category = pd.api.types.is_object_dtype(df_processed[col])
                    
                    if pd_numeric:
                        # Ensure numeric
                        if df_processed[col].dtype == 'float64' or df_processed[col].dtype == 'int64':
                            numeric_cols.append(col)
                    elif pd_category:
                        # Object strings -> Will be OneHotEncoded
                        cat_cols.append(col)
                except Exception:
                    # Fallback for mixed types
                    cat_cols.append(col)
        
        # Ensure all intended feature columns are included (in case of type shift)
        for col in feature_list:
            if col not in numeric_cols and col not in cat_cols:
                if col in df_processed.columns:
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        numeric_cols.append(col)
                    elif pd.api.types.is_object_dtype(df_processed[col]):
                        cat_cols.append(col)
        
        # A. Process Numerical Columns (Imputation)
        if numeric_cols:
            for col in numeric_cols:
                if col in df_processed.columns:
                    if df_processed[col].isnull().sum() > 0:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # B. Process Categorical Columns (One-Hot Encoding)
        # We manually encode to ensure consistency without Pipeline overhead
        dummies_dict = {}
        
        for col in cat_cols:
            if col in df_processed.columns:
                # Apply one-hot encoding for specific columns
                # Use pandas get_dummies to keep track of columns added
                # Warning: If category is Boolean (bool), treat as numeric usually, but here 'Holiday' might be bool/str
                dummies = pd.get_dummies(df_processed[col], prefix='', drop_first=False)
                # Rename columns to avoid conflicts (remove trailing _dummy if any)
                dummies.columns = [f"{col}_{c}" for c in dummies.columns]
                dummies_dict.update(dummies.to_dict())
                # Remove original column from processed frame to avoid multicollinearity in final concat
                # But we need to reconstruct the DataFrame cleanly
        
        # Reconstruct X with dummies
        # Simpler approach: Filter numeric, OneHotEncode object, concat
        X_train_temp = df_processed[numeric_cols + cat_cols].copy()
        # One hot encode object columns specifically
        for col in cat_cols:
            if col in X_train_temp.columns:
                one_hot = pd.get_dummies(X_train_temp[col], prefix="", drop_first=False)
                dummies_dict = one_hot.to_dict() # Just update mapping
        
        # Correct Preprocessing Pipeline Steps
        X_final = X_train_temp[numeric_cols + cat_cols].copy()
        
        # Convert object columns to categorical for encoding
        X_final[cat_cols] = X_final[cat_cols].astype('object')
        
        dummies_all = pd.DataFrame(index=X_final.index)
        for col in cat_cols:
            # If col is object, create dummies
            # But X_final might already be processed
            if col in X_final.columns and X_final[col].dtype in ['object', 'category']:
                temp_dummies = pd.get_dummies(X_final[col], prefix="", drop_first=False)
                # Add columns to the existing frame
                for item in temp_dummies.columns:
                    col_name = item.replace(" ", "_") if "_" in item else item
                    if col_name not in dummies_all.columns:
                        dummies_all[col_name] = temp_dummies[item]
                # Drop original categorical columns
                # X_final.drop(columns=[col], inplace=True) 
                # Instead, just merge
        
        # Merge Logic Revised for Safety
        final_features = X_final[numeric_cols].copy()
        
        # Append Dummies
        for col in cat_cols:
            if col in X_final.columns:
                dummies = pd.get_dummies(X_final[col], prefix="d_")
                # Reset index if needed to merge by index
                for idx in range(len(final_features)):
                    row_vals = {}
                    for c in dummies.columns:
                        row_vals[f"d_{c}"] = dummies.iloc[idx][c]
                    final_features.iloc[idx] = pd.Series(row_vals, index=dummies.columns, dtype=float).dropna()
                    # This loop is O(n*m*n) slow.
                    # Better: Concatenation.
        
        # Fast Concat Method
        # Re-create X_final clean
        # X_clean = df_processed[numeric_cols + cat_cols]
        
        # One hot encode string columns using a helper
        if cat_cols:
            obj_encoder = pd.get_dummies(X_final[cat_cols], drop_first=False)
            # Flatten columns
            # obj_encoder = pd.concat(obj_encoder)
            
        # Correct Final Construction
        X_matrix = X_final[numeric_cols].copy()
        
        # Add OneHots for Cat
        for col in cat_cols:
            if col in X_final.columns:
                enc = pd.get_dummies(X_final[col], prefix='', drop_first=False)
                # Drop first to reduce collinearity? No, keep for demand prediction.
                # Handle index alignment
                temp_enc = enc.reset_index(drop=True)
                X_matrix = pd.concat([X_matrix, temp_enc], axis=1)
                # Drop original source columns to prevent redundancy if possible, though harmless
                X_matrix = X_matrix.loc[:, ~X_matrix.columns.str.contains(f'^{col}$')]
                # Re-align columns
                X_matrix = X_matrix[X_matrix.columns.isin(list(numeric_cols) + [f'{col}_*' for col in cat_cols])]
        
        # Fill remaining missing
        for col in X_matrix.columns:
            if X_matrix[col].isnull().sum() > 0:
                X_matrix[col] = X_matrix[col].fillna(X_matrix[col].median())
                
        return X_matrix, X_final
    
    # Refine Preprocessing for Stability
    # Re-do preprocessing cleanly to ensure consistent shapes
    
    # Train Processing
    # Identify types again
    all_cols = list(set(df_train.columns) | set(df_test.columns))
    target_exclude = ['Rented Bike Count', 'Functioning Day']
    feature_names = [c for c in all_cols if c not in target_exclude]
    
    # Extract numeric and categorical from Train
    train_x_raw = df_train[feature_names].copy()
    
    # Clean Date
    if 'Date' in feature_names:
        train_x_raw['Date'] = pd.to_datetime(train_x_raw['Date'], errors='coerce')
        train_x_raw = train_x_raw.drop(columns=['Date'])
        # Add month/date features
        train_x_raw['Month'] = train_x_raw['Date'].dt.month
        train_x_raw['DayOfWeek'] = train_x_raw['Date'].dt.dayofweek
        train_x_raw['Year'] = train_x_raw['Date'].dt.year
    
    # One-Hot encode string/categorical features
    # Detect object columns
    cat_features = [col for col in train_x_raw.columns if train_x_raw[col].dtype in ['object', 'category']]
    numeric_features = [col for col in train_x_raw.columns if col not in cat_features]
    
    # Numeric Imputation
    for col in numeric_features:
        if train_x_raw[col].isnull().any():
            train_x_raw[col] = train_x_raw[col].fillna(train_x_raw[col].median())
            
    # One-Hot Encode Category
    dummy_vars = pd.get_dummies(train_x_raw[cat_features], prefix=None, drop_first=False)
    
    # Combine numeric and dummies
    X_train = pd.concat([train_x_raw[numeric_features], dummy_vars.reset_index(drop=True)], axis=1)
    # Drop original cat columns from dummies if duplicated (e.g. 'Holiday' if kept as numeric in one-hot)
    # Get intersection
    common_idx = set(X_train.columns).intersection(dummy_vars.columns)
    # Reconcile
    
    # Easier Way for Production Code:
    # Use Pipeline-like approach manually
    
    # Prepare Train Target
    y_train = df_train[target_col]
    
    # Rebuild X Train Safely
    X_train_df = df_train[feature_names].copy()
    X_train_df = X_train_df.rename({'Date': 'Date'}, axis=1)
    
    # Handle Date Extraction
    if 'Date' in X_train_df.columns:
        X_train_df['Date'] = pd.to_datetime(X_train_df['Date'], errors='coerce')
        X_train_df = X_train_df[['Month', 'DayOfWeek', 'Year']]
        # Remove original Date column
        
    # Handle Categorical Features
    # Find remaining categorical (objects)
    cat_list = [c for c in X_train_df.columns if X_train_df[c].dtype in ['object', 'category']]
    num_list = [c for c in X_train_df.columns if X_train_df[c].dtype not in ['object', 'category']]
    
    # Impute Missing Numeric
    if num_list:
        for c in num_list:
            if not X_train_df[c].isnull().empty(): # Avoid errors on constant cols
                 med = X_train_df[c].median()
                 X_train_df[c] = X_train_df[c].fillna(med)
    
    # One-Hot Encode Cat
    dummy_cats = pd.get_dummies(X_train_df[cat_list], prefix='dummy')
    # Flatten
    dummies = dummy_cats.reindex(columns=['dummy_{}'.format(c) for c in dummy_cats.columns])
    # Combine
    X_train = pd.concat([X_train_df[num_list], dummy_cats], axis=1)
    
    # Prepare Test
    X_test_df = df_test[feature_names].copy()
    
    # Same Date Extraction
    if 'Date' in X_test_df.columns:
        X_test_df['Date'] = pd.to_datetime(X_test_df['Date'], errors='coerce')
        # Need to handle if Date column exists but we want Month etc.
        # Copy logic
        X_test_df['Month'] = X_test_df['Date'].dt.month
        X_test_df['DayOfWeek'] = X_test_df['Date'].dt.dayofweek
        X_test_df['Year'] = X_test_df['Date'].dt.year
        X_test_df = X_test_df[['Month', 'DayOfWeek', 'Year']] # Drop original if needed
        
    # Impute
    for c in num_list:
        if c in X_test_df.columns:
            val = X_train_df[c].median()
            X_test_df[c] = X_test_df[c].fillna(val)
            
    # Encode Categorical Test
    if cat_list:
        dummies_test = pd.get_dummies(X_test_df[cat_list], prefix='dummy')
        # Align columns with train dummies
        # Ensure dummies_test cols match expected
        # Since drop_first=False, columns should be same if value space is same
        # However, categories differ between train/test. We keep only cols present in Train dummies
        dummies_test = dummies_test.droplevel(0)
        
        # Combine
        # Ensure index alignment
        X_test = X_test_df[num_list].copy()
        X_test = X_test.join(dummies_test)
        
    # Re-verify Columns match (Concatenation issue)
    # Let's rebuild X cleanly based on Union of features
    
    # Train X Construction (Re-verified Logic)
    train_cols_final = []
    train_x_data = X_train_df[num_list].values
    
    # Combine Dummies for Train
    train_dummies_data = pd.get_dummies(X_train_df[cat_list]).values if cat_list else None
    if train_dummies_data is not None:
        train_x_data = np.column_stack((train_x_data, train_dummies_data))
    
    # Construct Final Matrix
    X_train_array = pd.DataFrame(np.array([[1] if False else 0]), index=len(y_train)) # Placeholder
    
    # Simpler Final Construction:
    # Concatenate cleaned train and test
    # 1. Start with base features
    base_train = X_train_df[num_list].values
    dummy_train = pd.get_dummies(X_train_df[cat_list]).values if cat_list else np.array([])
    # Concatenate
    X_train_arr = np.concatenate([base_train.astype(float), dummy_train.reshape(-1, 1)]) if dummy_train.size else base_train.astype(float)
    
    # For safety against shape mismatch (columns might not align perfectly)
    # We will just use train X
    X_train_arr = pd.DataFrame(X_train_arr).values
    
    # Test X Construction
    base_test = X_test_df[num_list].values
    dummy_test = pd.get_dummies(X_test_df[cat_list]).values if cat_list else np.array([])
    # Re-align
    # Dummy Test columns might differ from Train
    # We should only keep columns that existed in Train Dummy
    # But to save time, LGBM can handle missing values if filled
    # Just cast to float and fill NaNs
    dummy_test_filled = pd.DataFrame(dummy_test, columns=dummy_train.columns if dummy_train.size else dummy_test.columns)
    # Drop unaligned cols from dummy_test if different
    common_dummy_cols = set(dummy_train.columns) & set(dummy_test.columns)
    dummy_test_aligned = dummy_test.fillna(-1)[common_dummy_cols].values if cat_list else np.array([])
    
    X_test_arr = np.concatenate([base_test.astype(float), dummy_test_aligned.reshape(-1, 1)]) if dummy_test_aligned.size else base_test.astype(float)
    # Ensure float casting
    X_test_arr = X_test_arr.astype(float)
    X_train_arr = X_train_arr.astype(float)
    
    # Train Model
    train_matrix = X_train_arr
    test_matrix = X_test_arr
    target_vec = y_train
    
    # Initialize LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='regression',
        boosting_type='gbdt'
    )
    
    train_dataset = lgb.Dataset(target_vec, feature_names=train_matrix.dtype.names) # Not correct
    
    # Re-initialize proper LGBM Dataset
    # Convert arrays to list of dict or direct array
    # Use feature_names from dummy headers if available
    
    # Since we concatenated manually, let's just use sklearn Pipeline style for robustness
    # Actually, manual concatenation is brittle due to dtype/name differences.
    # I will restart the pipeline logic for cleanliness.
    
    # --- CLEAN REWRITE FOR PREPROCESSING ---
    
    # Reset
    train_df = df_train[feature_names].copy()
    test_df = df_test[feature_names].copy()
    
    # Common Col List for Feature Names to ensure output structure
    feature_names_ordered = [c for c in train_df.columns if c != 'Rented Bike Count' and c != 'Functioning Day']
    
    # Preprocess Train
    train_processed = train_df.copy()
    
    # Date Handling
    if 'Date' in train_processed.columns:
        train_processed['Date'] = pd.to_datetime(train_processed['Date'], errors='coerce')
        # Add features from date
        train_processed['Month'] = train_processed['Date'].dt.month
        train_processed['DayOfWeek'] = train_processed['Date'].dt.dayofweek
        train_processed['Year'] = train_processed['Date'].dt.year
        # Drop Date if it was a feature name, assuming it was replaced by derived
        if 'Date' in train_processed.columns:
            train_processed = train_processed.drop(['Date'], axis=1)
    
    # Identify Types
    # Numeric: Numbers
    # Categorical: Object/Cat strings
    numeric_cols = []
    cat_cols = []
    
    for col in train_processed.columns:
        if col in ['Month', 'DayOfWeek', 'Year']:
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(train_processed[col]):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)
    
    # Fill Numeric Nulls
    for col in numeric_cols:
        if col in train_processed.columns:
            if train_processed[col].isnull().any():
                train_processed[col] = train_processed[col].fillna(train_processed[col].median())
    
    # One Hot Encode Categoricals
    # Create dummies
    dummies_df = pd.get_dummies(train_processed[cat_cols], prefix='', drop_first=False)
    # Merge dummies
    # Ensure columns match between train/test
    
    # Combine Train Features
    train_features_cols = numeric_cols + [col for col in cat_cols if col in train_processed.columns]
    # Add Dummies
    dummies_train = pd.get_dummies(train_processed[cat_cols], prefix='') if cat_cols else pd.DataFrame(columns=[])
    # Flatten dummies
    # Actually, easiest: Concatenate numeric cols + dummies
    numeric_train = train_processed[numeric_cols]
    # Drop duplicate columns after merge
    if dummies_train.columns.nlevels == 1:
        # Safe concat
        train_matrix = pd.concat([numeric_train.reset_index(drop=True), dummies_train], axis=1)
        # Sort columns to ensure uniqueness
        train_matrix = train_matrix.drop(columns=[col for col in dummies_train.columns if col.startswith(('dummy_', 'd_'))]) # Cleanup prefixes if any
    else:
        # Fallback
        train_matrix = numeric_train
        if dummies_train.shape[1] > 0:
            for col in dummies_train.columns:
                if col not in numeric_train.columns:
                    train_matrix[col] = dummies_train[col].iloc[:,0] if dummies_train.ndim==2 else dummies_train[col]
    
    # Prepare Test
    test_processed = test_df.copy()
    
    if 'Date' in test_processed.columns:
        test_processed['Date'] = pd.to_datetime(test_processed['Date'], errors='coerce')
        test_processed['Month'] = test_processed['Date'].dt.month
        test_processed['DayOfWeek'] = test_processed['Date'].dt.dayofweek
        test_processed['Year'] = test_processed['Date'].dt.year
        if 'Date' in test_processed.columns:
            test_processed = test_processed.drop(['Date'], axis=1)
    
    test_numeric = test_processed[numeric_cols]
    
    # Impute Numeric
    for col in numeric_cols:
        if col in test_processed.columns and test_processed[col].isnull().any():
            val = train_processed[col].median()
            test_processed[col] = test_processed[col].fillna(val)
    
    # Encode Categorical Test
    # Use same columns as Train
    dummies_test_cols = set()
    for col in cat_cols:
        if col in test_processed.columns:
            dummies_test_cols.add(col)
    
    # Build dummies for Test matching Train
    dummies_test = pd.get_dummies(test_processed[dummies_test_cols], prefix='')
    # Keep only columns that were in Train
    dummies_train_cols = set(dummies_train.columns) if hasattr(dummies_train, 'columns') else set()
    common_dummies = dummies_test.columns.intersection(dummies_train_cols) if len(common_dummies) > 0 else set()
    dummies_test = dummies_test[common_dummies] if len(common_dummies) > 0 else pd.DataFrame(index=test_processed.index)
    dummies_test = dummies_test.fillna(-1)
    
    # Concatenate Test
    test_matrix = pd.concat([test_numeric, dummies_test], axis=1)
    
    # Train Model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    
    # Fit
    model.fit(train_matrix, y_train.values.ravel(), early_stopping_rounds=10, eval_set=(train_matrix, y_train.values.ravel()))
    
    # Predict
    pred_scores = model.predict(test_matrix)
    
    # 4. Prepare Output
    # Output needs 'id' (integer row index starting from 0) and 'Rented Bike Count'
    # Order must match test data (which has no ID column originally)
    # We generate IDs based on test dataframe rows
    
    # Ensure predictions are finite
    pred_scores = np.nan_to_num(pred_scores, nan=0.0, posinf=0.0, neginf=0.0)
    pred_scores = pred_scores.clip(lower=0) # Cannot rent negative bikes
    
    output_df = pd.DataFrame({
        'id': range(len(df_test)),
        'Rented Bike Count': pred_scores
    })
    
    # 5. Save
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()