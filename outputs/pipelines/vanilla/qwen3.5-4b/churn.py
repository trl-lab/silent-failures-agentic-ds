import warnings
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Suppress warnings to ensure clean output
warnings.filterwarnings('ignore')

# Define data paths as provided in the prompt
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/churn/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/churn/data/test.parquet"

try:
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Separate target and features
    target_col = 'class'
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    # Initialize CatBoost Classifier
    # Handles categorical features (encoded as strings) and missing values automatically
    model = CatBoostClassifier(
        loss_function='Logloss',
        iterations=100,
        verbose_level=0,
        device='CPU',
        thread_count=-1,
        eval_metric='auc'
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    predictions = model.predict(X_test)

    # Create output dataframe with 'id' (row index starting from 0) and 'class' (predicted label)
    results_df = pd.DataFrame({
        'id': range(len(predictions)),
        'class': predictions.astype(int)
    })

    # Save predictions to csv
    results_df.to_csv('predictions.csv', index=False)

except Exception as e:
    # Handle potential loading errors gracefully
    print(f"An error occurred: {e}")
except ImportError:
    # Fallback to Scikit-Learn pipeline if CatBoost is somehow unavailable (unlikely per prompt)
    print("CatBoost not found. Attempting to import fallback...")
    # This block would implement skel learning pipeline if catboost failed
    raise ImportError("CatBoostLibraryMissing")
except Exception:
    raise