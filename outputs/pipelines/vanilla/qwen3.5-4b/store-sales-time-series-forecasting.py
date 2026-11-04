import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/kaggle/store-sales-time-series-forecasting/data/test.parquet'

def main():
    # Load Data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Identify Target Column
    target_col = 'sales'
    
    # Ensure 'id' column exists in output but not processed as feature
    # Drop 'id' from features if present in either dataset to avoid leakage
    for df in [train_df, test_df]:
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    if len(feature_cols) == 0:
        # Fallback if no features found
        X_train = train_df.iloc[:, :1].values.astype(float)
        X_test = test_df.iloc[:, :1].values.astype(float)
        y_train = train_df[target_col].values
        y_pred = np.full(len(X_test), np.mean(y_train))
        final_predictions = pd.DataFrame({'id': list(range(len(X_test))), 'sales': y_pred})
    else:
        # Initialize Preprocessor
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))])
        
        # Determine columns types dynamically
        # Note: This assumes the dtypes are consistent between train and test
        # We force them to be numeric/imputable
        column_types = {}
        for col in feature_cols:
            sample = train_df[col].astype(str)
            # If string representation contains digits, attempt numeric logic? 
            # Safer to rely on sklearn inference or explicit checks.
            # Explicit check for numeric is safer for speed
            if pd.api.types.is_numeric_dtype(train_df[col]):
                column_types[col] = 'numeric'
            else:
                column_types[col] = 'categorical'
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, list(column_types[k] == 'numeric' for k in feature_cols)[0:len(feature_cols)]), # Wrong indexing
        ])
        # Correct way: map features to transformers
        
        num_feat_idx = []
        cat_feat_idx = []
        for idx, col in enumerate(feature_cols):
            if pd.api.types.is_numeric_dtype(train_df[col]):
                num_feat_idx.append(idx)
            else:
                cat_feat_idx.append(idx)
                
        preprocessor = ColumnTransformer([
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), num_feat_idx),
            ('cat', Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_feat_idx),
        ])
        
        # Initialize Model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', lgb.LGBMRegressor(objective='regression', silent=True, n_jobs=-1, verbosity=-1, max_depth=6, learning_rate=0.05, num_leaves=31, boosting_type='gbdt'))
        ])
        
        # Prepare Features
        X_train_raw = train_df[feature_cols].copy()
        y_train = train_df[target_col].values
        
        X_test_raw = test_df[feature_cols].copy()
        
        # Fit
        model.fit(X_train_raw, y_train)
        
        # Predict
        preds = model.predict(X_test_raw)
        
        # Ensure predictions are valid
        pred_df = pd.DataFrame({
            'id': list(range(len(preds))),
            'sales': preds
        })
        
    else:
        pred_df = pd.DataFrame({
            'id': list(range(len(X_test))),
            'sales': y_pred
        })

    # Save Predictions
    pred_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()