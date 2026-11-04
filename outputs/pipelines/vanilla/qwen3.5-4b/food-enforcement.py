import pandas as pd
import numpy as np
import os

# Configuration
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/test.parquet"
TARGET_COLUMN = "classification"
OUTPUT_FILE = "predictions.csv"

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("catboost is required but not found.")

def main():
    # Load data
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Check for target column
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in train dataframe.")
    
    # Extract target and features
    y_train = train_df[TARGET_COLUMN]
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    # Initialize model (CatBoost handles categorical and missing values automatically)
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="MultiClass",
        verbose=0,
        random_seed=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred_proba = model.predict_proba(X_test)
    # Get predicted classes
    # Since it's multi-class, predict returns indices
    pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Handle case where CatBoost.predict directly returns labels for calibration
    # CatBoostClassifier.predict() returns probabilities for proba, classes are argmax
    # However, for multiclass, sometimes fit expects 'Multi_Class' metric.
    # We can use model.predict(X_test) which returns class indices or labels depending on settings.
    # Safer to use predict for labels if eval_metric was MultiClass and we have unique classes.
    # Let's verify standard usage: model.predict(X_test) gives integer indices.
    # However, for CatBoost, predict returns probabilities if you call predict_proba?
    # No, predict_proba returns dense array of proba. predict returns int label indices.
    
    # Let's use model.predict for class labels
    # Ensure it returns string or numeric matching training labels if necessary
    # But accuracy calculation needs consistency. 
    # Using model.predict is standard for labels.
    # Re-checking CatBoost predict documentation: predicts class indices.
    
    # Correct approach for labels:
    # y_pred = model.predict(X_test)
    
    # To ensure we match the class labels exactly from training:
    # model.classes_ contains the unique targets in sorted order usually?
    # CatBoost stores unique classes in y_unique_.
    # We can map the model's output directly.
    
    # Let's reset to standard sklearn style mapping if unsure, but CatBoost is robust.
    # I'll use model.predict() which returns indices corresponding to y_train unique values.
    # If y_train were strings, y_train.values might be object.
    # I will convert indices back to labels using model.y_pred_? No.
    # I'll use model.predict() assuming consistent encoding.
    # However, for safety with labels matching exactly:
    y_pred_indices = model.predict(X_test)
    
    # Verify if classes need conversion.
    # In multi-class, predict returns indices of unique class labels found in train.
    # If original targets were strings, these indices refer to the sorted order.
    # We need to know the actual class names.
    # Unique classes from train
    train_y_unique = train_df[TARGET_COLUMN].unique()
    
    # Convert indices to actual class labels
    y_pred_labels = np.array([train_y_unique[i] for i in y_pred_indices])
    
    # Prepare output dataframe
    predictions_df = pd.DataFrame({
        "id": range(len(test_df)),
        "classification": y_pred_labels
    })
    
    # Save to CSV
    predictions_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()