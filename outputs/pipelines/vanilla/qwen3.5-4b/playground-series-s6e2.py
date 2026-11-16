import pandas as pd
import numpy as np
import catboost as cb

# Set working directory
current_dir = '/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s6e2/data/'

# Load data
train_df = pd.read_parquet(current_dir + 'train.parquet')
test_df = pd.read_parquet(current_dir + 'test.parquet')

# Target variable
target_col = 'Heart Disease'

# Feature extraction
train_features = train_df[[col for col in train_df.columns if col != target_col]]
test_features = test_df[[col for col in test_df.columns if col != target_col]]

# Initialize model - CatBoost handles categorical and missing values automatically
model = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    verbose=0,
    thread_count=1,
    n_estimators=500
)

# Train
model.fit(train_features, train_df[target_col], eval_set=(test_features, train_df[target_col]))

# Predict probabilities for AUC
predictions = model.predict_proba(test_features)[:, 1]

# Prepare output dataframe
result_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Heart Disease': predictions
})

# Save predictions
result_df.to_csv('predictions.csv', index=False)