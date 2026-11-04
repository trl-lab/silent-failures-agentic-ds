import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Define file paths
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/test.parquet'
OUTPUT_FILE = 'predictions.csv'

def main():
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Target column
    TARGET_COL = 'SALE_PRC'
    EXCLUDE_COLS = ['PARCELNO', 'SALE_PRC']
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]
    
    # Filter features for test dataframe
    test_features = [col for col in test_df.columns if col in feature_cols and col != 'SALE_PRC']
    
    # Identify numeric and categorical columns
    # Note: We use the original train dataframe to determine types
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [col for col in feature_cols if col not in num_cols]
    
    # Preprocessing Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    # Prepare Training Data
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    
    # Prepare Test Data (ensure correct order of columns matches feature_cols)
    # Sort columns to match feature_cols order exactly
    test_df_sorted = test_df.reindex(sorted(test_df.columns), axis=1)
    X_test = test_df_sorted[test_features]
    
    # Fit Preprocessor
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    
    # Train Model
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        min_child_samples=20,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_processed, y_train)
    
    # Predict on Test Data
    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)
    
    # Create Output DataFrame with ID
    ids = list(range(len(predictions)))
    
    output_data = {
        'id': ids,
        'SALE_PRC': predictions
    }
    
    # Save to CSV
    pd.DataFrame(output_data).to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()