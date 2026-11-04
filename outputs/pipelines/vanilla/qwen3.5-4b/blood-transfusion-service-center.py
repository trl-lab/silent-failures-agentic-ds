import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Define the paths for training and testing data as specified
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/test.parquet"

def main():
    # Load data from parquet files
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Identify target and feature columns
    target_column = "Class"
    y_train = train_df[target_column].copy()
    X_train = train_df.drop(columns=[target_column])
    X_test = test_df.drop(columns=[target_column])
    
    # Separate numeric and categorical columns for preprocessing
    # This ensures missing values and categorical columns are handled automatically
    feature_columns = list(X_train.columns)
    numeric_features = []
    categorical_features = []
    
    for col in feature_columns:
        dtype = X_train[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    # Create Preprocessor Pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'))
    ])
    
    # Preprocessor wrapper handling mixed types
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Initialize and compile model with LightGBM
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(
            objective='binary',
            eval_metric='logloss',
            random_state=42,
            num_class=2,
            force_col_order=True,
            verbosity=-1
        ))
    ])
    
    # Train the model
    model.fit(X_train.values, y_train.values)
    
    # Generate predictions on test set
    y_test_proba = model.predict_proba(X_test.values)[:, 1]
    
    # Convert probability to class labels (threshold at 0.5)
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Prepare output DataFrame with required columns: 'id' and 'Class'
    # 'id' is the integer row index starting from 0
    output_df = pd.DataFrame({
        'id': range(len(y_test_pred)),
        'Class': y_test_pred
    })
    
    # Save predictions to CSV
    output_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()