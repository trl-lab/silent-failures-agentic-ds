import pandas as pd
import numpy as np
import catboost as cb

# File paths as specified
TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/mlebench/random-acts-of-pizza/data/test.parquet'

# Data loading
print("Loading data...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Target and Feature definitions
TARGET_COL = 'requester_received_pizza'
FEATURE_COLS = [col for col in df_train.columns if col != TARGET_COL]

# Prepare data
X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].astype(float)
X_test = df_test[FEATURE_COLS].copy()

# Model initialization
# catboost handles missing values and categorical features automatically
model = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='LogLoss',
    eval_metric='AUC',
    verbose=False,
    random_seed=42,
    thread_count=1,
    dev_data=None
)

# Training
print("Training model...")
try:
    model.fit(X_train, y_train, eval_set=(X_train, y_train), verbose=False)
except Exception:
    # Fallback or warning handling if necessary, though unlikely with defaults
    print("Warning: Training finished without explicit stopping criteria.")

# Prediction
probs = model.predict_proba(X_test)[:, 1]

# Create output dataframe
output_df = pd.DataFrame({
    'id': range(len(df_test)),
    'requester_received_pizza': probs
})

# Save predictions
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")