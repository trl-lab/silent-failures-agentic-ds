import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import os

# Load data
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

# Identify target and features
target = 'class'
features = [col for col in train.columns if col != target]

# Sort by date for temporal ordering
if 'date' in train.columns:
    train = train.sort_values('date').reset_index(drop=True)

X = train[features]
y = train[target]

# Temporal validation split (last 20% as validation) to respect time-series structure
split_idx = int(len(train) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

# Train model with LightGBM
model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    verbose=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# Validate using official metric (accuracy)
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Final Validation Score (Accuracy): {val_accuracy:.4f}")

# Retrain on full training data with best iterations
best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') and model.best_iteration_ > 0 else 300
model_full = lgb.LGBMClassifier(
    n_estimators=best_iter,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    verbose=-1
)
model_full.fit(X, y)

# Predict on test
test_preds = model_full.predict(test[features])

# Create submission
os.makedirs('./submission', exist_ok=True)
submission = pd.DataFrame({
    'id': range(len(test_preds)),
    'class': test_preds
})
submission.to_csv('./submission/submission.csv', index=False)
print(f"Submission saved with {len(submission)} predictions.")