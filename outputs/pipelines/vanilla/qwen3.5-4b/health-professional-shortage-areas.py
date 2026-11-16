import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

def main():
    # Configuration
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/test.parquet"
    target_col = "HPSA Score"

    # Load data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Determine features (exclude target)
    feature_cols = [col for col in df_train.columns if col != target_col]
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].values
    
    X_test = df_test[feature_cols].copy()
    
    # Separate numerical and categorical columns for preprocessing
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Construct Preprocessing Pipeline
    transformers = []
    if len(cat_features) > 0:
        transformers.append(('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features))
    if len(num_features) > 0:
        transformers.append(('num', SimpleImputer(strategy='median'), num_features))
    
    column_transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")
    
    # Fit Preprocessor
    if column_transformer:
        preprocessor = Pipeline(steps=[
            ('preprocess', column_transformer)
        ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
    else:
        # Case where all columns are numeric (just imputation)
        preprocessor = Pipeline(steps=[('preprocess', column_transformer)])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
    # Train LightGBM Model
    model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, random_state=42, n_estimators=1000, max_depth=-1, learning_rate=0.05)
    model.fit(X_train_processed, y_train)

    # Generate Predictions
    predictions = model.predict(X_test_processed)

    # Prepare Output DataFrame with required columns
    pred_df = pd.DataFrame({
        "id": list(range(len(predictions))),
        "HPSA Score": predictions
    })

    # Save predictions
    pred_df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()