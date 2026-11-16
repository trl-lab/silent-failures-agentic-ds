import pandas as pd
import numpy as np
from sklearn.model_selection import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Define file paths
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/test.parquet"
TARGET_COL = "RATIO_ENROLL_EMP"
OUTPUT_CSV = "predictions.csv"

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Extract features and target
    target_col = train_df.columns.get_loc(TARGET_COL)
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    
    if len(feature_cols) == 0:
        raise ValueError("No features found in the dataset.")
    
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values
    y_test_ref = test_df[TARGET_COL].values  # Reference for alignment if needed (though we won't use for prediction here)

    # Clean target data: Remove rows with missing target values in train
    if np.any(np.isnan(y_train)):
        mask = ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]
    
    # Identify numeric and categorical columns for preprocessing
    numeric_cols = []
    categorical_cols = []
    
    for col in X_train.columns:
        col_data = X_train[col]
        dtype = col_data.dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            categorical_cols.append(col)
    
    # Setup Preprocessing Pipeline
    # 1. Impute missing values: Median for numeric, Mode for categorical
    num_imp = SimpleImputer(strategy='median')
    cat_imp = SimpleImputer(strategy='most_frequent')
    
    # 2. Encode categorical variables: One-Hot Encoding
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    num_transformer = Pipeline(steps=[
        ('imputer', num_imp)
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', cat_imp),
        ('encoder', ohe)
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Define Regression Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Combine into Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit model on training data
    print("Training model...")
    clf.fit(X_train, y_train)
    
    # Predict on test data
    print("Generating predictions...")
    X_test_prepped = preprocessor.transform(X_test)
    predictions = clf.predict(X_test_prepped)
    
    # Prepare output dataframe
    # Requirement: 'id' (integer row index starting from 0) and 'RATIO_ENROLL_EMP'
    output_df = pd.DataFrame({
        'id': range(len(predictions)),
        'RATIO_ENROLL_EMP': predictions
    })
    
    # Save to CSV
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()