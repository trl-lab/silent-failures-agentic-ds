import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Define paths and constants
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/test.parquet"
TARGET_NAME = "Total IFC investment as approved by Board(Million - USD)"
OUTPUT_FILE = "predictions.csv"

def load_and_preprocess(train_path, test_path, target_name):
    # Load Data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Identify Target Index
    target_index = train_df.columns.tolist().index(target_name)
    
    # Get Features
    feature_columns = [col for col in train_df.columns if col != target_name]
    
    # Ensure test dataframe has the same feature columns
    test_df = test_df[feature_columns]
    train_df = train_df[feature_columns + [target_name]]
    
    X_train = train_df.drop(columns=[target_name])
    X_test = test_df
    
    # Identify Numerical and Categorical columns
    num_cols = []
    cat_cols = []
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            num_cols.append(col)
        elif pd.api.types.is_string_dtype(X_train[col]) or pd.api.types.is_object_dtype(X_train[col]):
            cat_cols.append(col)
        else:
            # Fallback to numeric if inferred incorrectly
            # Try converting to numeric
            temp = pd.to_numeric(X_train[col], errors='coerce')
            if not temp.isna().all():
                num_cols.append(col)
    
    # Impute and Encode
    transformer_num = SimpleImputer(strategy='median')
    transformer_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformer_num, num_cols),
            ('cat', transformer_cat, cat_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit Preprocessor on Train
    X_train_processed = preprocessor.fit_transform(X_train)
    y_train = train_df[target_name].values
    
    # Transform Test
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, y_train, X_test_processed, feature_columns

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and Preprocess
    X_train, y_train, X_test, features = load_and_preprocess(
        TRAIN_PATH, TEST_PATH, TARGET_NAME
    )
    
    # Initialize Model
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    # Train Model
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)
    
    # Prepare Output DataFrame
    output_df = pd.DataFrame({
        'id': list(range(len(X_test))),
        TARGET_NAME: y_pred
    })
    
    # Save to CSV
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()