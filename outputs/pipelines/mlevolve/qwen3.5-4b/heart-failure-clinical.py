import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Load data
train_df = pd.read_parquet('./input/train.parquet')
test_df = pd.read_parquet('./input/test.parquet')

# Define features and target
feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
                'sex', 'smoking', 'time']
target_col = 'DEATH_EVENT'

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df[target_col]

# Split data for validation
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Define callbacks for LightGBM (API compatibility note)
callbacks = [
    lgb.early_stopping(50),
    lgb.log_evaluation(-1)
]

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

# Train model on full training data
model.fit(
    X_train, y_train,
    eval_set=[(X_train_val, y_train_val), (X_test_val, y_test_val)],
    callbacks=callbacks
)

# Make predictions on validation set
y_val_pred = model.predict(X_train_val)
y_val_pred_proba = model.predict_proba(X_train_val)[:, 1]

# Calculate validation score
val_score = roc_auc_score(y_train_val, y_val_pred_proba)
print(f'Final Validation Score: {val_score:.4f}')

# Make predictions on test set
y_test_pred_proba = model.predict_proba(test_df[feature_cols])[:, 1]

# Create submission in correct format
submission_df = pd.DataFrame({
    'id': range(len(test_df)),
    'DEATH_EVENT': y_test_pred_proba
})
submission_df.to_csv('./submission/submission.csv', index=False)

print('Submission saved to ./submission/submission.csv')