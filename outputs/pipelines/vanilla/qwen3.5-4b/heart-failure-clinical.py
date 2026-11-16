import pandas as pd
import numpy as np
import lightgbm as lgb

# Define file paths as specified
train_path = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/test.parquet"

# Load data
print("Loading data...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Define feature names based on the variable descriptions table
# Excluding the target 'death_event'
feature_columns = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]
target_column = 'death_event'

# Separate features and target for training
if target_column not in train_df.columns:
    raise ValueError(f"Target column '{target_column}' not found in training data.")

X_train = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()

# Prepare test features
if target_column in test_df.columns:
    test_df_for_pred = test_df.drop(columns=[target_column]).copy()
else:
    test_df_for_pred = test_df.copy()

X_test = test_df_for_pred[feature_columns].copy()

# Function to handle missing values and ensure compatibility
def preprocess_data(dataframe):
    # Replace NaN with 0 to ensure robust prediction without external sklearn fillna dependencies
    # LightGBM handles -nan- internally, but explicit 0 fill ensures numpy array stability
    return dataframe.fillna(0)

X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

# Initialize LightGBM models
# Using LGBMClassifier for ease and automatic category handling if needed
# Although direct fit/predict is slightly faster and avoids sklearn overhead
params = {
    'objective': 'binary',
    'metric': 'binary_logloss', # Using logloss for calibration with AUC
    'booster': 'gbdt',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'random_state': 42,
    'n_jobs': -1
}

# Define datasets
train_set = lgb.Dataset(X_train_processed, label=y_train)
val_set = lgb.Dataset(X_test_processed, reference=train_set) # Used for validation during training if applicable

# Train the model
print("Training model...")
model = lgb.train(
    params, 
    train_set, 
    num_boost_round=1000, 
    valid_sets=val_set, 
    valid_names=['val'],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# Generate predictions on test set
# predict_proba gives probabilities
probabilities = model.predict(X_test_processed)

# Convert probabilities to binary predictions based on threshold (0.5)
threshold_predictions = (probabilities >= 0.5).astype(int)

# Create the output dataframe with 'id' starting from 0
results_df = pd.DataFrame({
    'id': np.arange(len(threshold_predictions)),
    'DEATH_EVENT': threshold_predictions
})

# Save predictions to CSV in current working directory
results_df.to_csv('predictions.csv', index=False)

print("Done! Predictions saved to predictions.csv")