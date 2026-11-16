import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    # Configuration paths
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/yelp_business/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/yelp_business/data/test.parquet'
    output_path = 'predictions.csv'
    
    # Load Data
    try:
        train_df = pd.read_parquet(train_path)
    except Exception as e:
        print(f"Failed to load train: {e}")
        return

    try:
        test_df = pd.read_parquet(test_path)
    except Exception as e:
        print(f"Failed to load test: {e}")
        return

    # Target Column
    target_col = 'stars'
    if target_col not in train_df.columns:
        print("Target column 'stars' not found")
        return
        
    # Identify Feature Columns
    feature_cols = [col for col in train_df.columns if col != target_col]
    if len(feature_cols) == 0:
        print("No features found.")
        return
    
    # Prepare Data Types
    numeric_data = train_df[feature_cols].select_dtypes(include=['float', 'int'])
    categorical_data = train_df[feature_cols].select_dtypes(include=['object', 'category'])
    
    # Feature Names for Num/Cat
    num_feature_names = numeric_data.columns.tolist() if len(numeric_data.columns) > 0 else []
    cat_feature_names = categorical_data.columns.tolist() if len(categorical_data.columns) > 0 else []
    
    # Preprocessing Components
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', output_mode='numpy')

    # Define Transformers
    numeric_transformer = Pipeline([
        ('imputer', numeric_imputer)
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', categorical_imputer),
        ('encoder', categorical_encoder)
    ])

    # Combine into Column Transformer
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feature_names),
            ('cat', categorical_transformer, cat_feature_names)
        ],
        remainder='drop'
    )

    # Model Definition
    model = LGBMClassifier(
        objective='multiclass',
        class_weight='balanced',
        n_estimators=200,
        random_state=42,
        eval_metric='multi_logloss',
        verbose=-1
    )

    # Train Pipeline
    clf = Pipeline([
        ('preprocessing', col_transformer),
        ('model', model)
    ])

    # Cross Validation to check stability
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Weights adjustment for accuracy
    # For simplicity, let's train directly as requested with enough samples
    # Ensure we fit on training data
    # To avoid OOM issues, limit estimators if needed, but 200 is fine for this dataset type.

    clf.fit(train_df, train_df[target_col])

    # Predictions on Test Set
    # Apply the same preprocessing
    processed_test = clf.named_steps['preprocessing'].fit_transform(test_df)
    processed_train = clf.named_steps['preprocessing'].fit_transform(train_df)
    
    # Re-fit the transformer on full train set and apply on test set to align correctly
    # Pipeline approach handles this automatically but we extract to predict on test
    test_predictions = clf.named_steps['model'].predict(processed_test)
    
    # Prepare Output DataFrame
    output_df = pd.DataFrame({
        'id': range(len(test_df)),
        'stars': test_predictions
    })
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()