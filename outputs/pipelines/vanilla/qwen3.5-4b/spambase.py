import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def main():
    # Define file paths as specified
    base_path = "/path/to/user/workspace/ds_failures/datasets/openml/spambase/data/"
    train_path = base_path + "train.parquet"
    test_path = base_path + "test.parquet"
    
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Target variable name
    target_col = 'class'
    
    # Ensure target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data")
    
    # Get feature columns from training set
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Extract features and labels
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    
    # Initialize preprocessors
    imputer = SimpleImputer(strategy='median')
    
    # Define column transformers
    numeric_features = X_train.select_dtypes(include='number').columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', imputer, numeric_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )
    
    # Define model
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,
        objective='binary',
        learning_rate=0.1,
        num_leaves=31
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Generate predictions for test set
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    # Convert probability to class (threshold 0.5)
    y_pred_class = np.round(y_pred_prob).astype(int)
    
    # Prepare output DataFrame
    # id: integer row index starting from 0
    predictions = pd.DataFrame({
        'id': range(len(X_test)),
        'class': y_pred_class
    })
    
    # Save to CSV
    predictions.to_csv('predictions.csv', index=False)
    
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()