import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Constants
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/competitive-data-science-predict-future-sales/data/test.parquet'
TARGET_COLUMN = 'item_cnt_day'
OUTPUT_FILE = 'predictions.csv'

def preprocess_features(df):
    """
    Preprocess dataframe:
    1. Drop 'item_cnt_day'
    2. Impute numeric columns
    3. Encode categorical columns
    4. Return processed matrix and feature names
    """
    df = df.copy()
    
    # Identify target column index
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
    
    # Identify feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    
    # Split into numeric and categorical
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        else:
            # Treat non-numeric as categorical (handling strings)
            categorical_cols.append(col)
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN].values
    
    # --- Processing Numeric Features ---
    if len(numeric_cols) > 0:
        imp = SimpleImputer(strategy='median')
        X.loc[:, numeric_cols] = imp.fit_transform(X[numeric_cols].values)
    
    # --- Processing Categorical Features ---
    # Use One-Hot Encoding for categorical columns
    if len(categorical_cols) > 0:
        encoders = {}
        encoded_names = []
        cat_X = X[categorical_cols]
        
        # To handle unknown categories safely in OneHotEncoder (if we wanted to generalize)
        # but here we stick to known features for train/Test alignment.
        for col in categorical_cols:
            # Store original column to ensure shape remains consistent
            # We will do a simple approach: replace strings with labels or OneHot
            # To save space and time on one-hot without sklearn complexity, 
            # we will use pandas get_dummies for categorical columns
            dummies = pd.get_dummies(cat_X[col], drop_first=False)
            X[dummies.columns] = dummies.values.astype(float)
            encoded_names.extend(list(dummies.columns))
        
        # Ensure float64 everywhere
        X = X.astype(np.float64)
    else:
        X = X.astype(np.float64)
    
    # Replace any remaining Inf/Nan with 0
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    
    return X, feature_cols, y

def main():
    # Load Data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Validate Target Column
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN not in test_df.columns # Target column usually missing in test
    
    # Process Train Data
    # We extract features and prepare them
    X_train_raw, train_feature_names, y_train = preprocess_features(train_df)
    
    # Process Test Data
    # We must ensure test df has the same columns as train for feature engineering
    # Identify columns present in train_feature_names
    valid_feature_cols = [col for col in test_df.columns if col in train_feature_names]
    
    # Map test dataframe to valid columns only
    test_df_subset = test_df[valid_feature_cols]
    
    # Run preprocessor logic again on test data
    # Since preprocessor logic is static, we can reuse it but need to filter cols first
    X_test, _, _ = preprocess_features(test_df_subset)
    
    # Verify shapes
    assert X_train_raw.shape[1] == X_test.shape[1], f"Shape mismatch: {X_train_raw.shape} vs {X_test.shape}"
    
    # Initialize and Train LightGBM Model
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train_raw, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Prepare Output DataFrame
    # Requirement: 'id' (integer row index starting from 0) and 'item_cnt_day'
    output_df = pd.DataFrame({
        'id': range(len(test_df)),
        'item_cnt_day': y_pred
    })
    
    # Save
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()