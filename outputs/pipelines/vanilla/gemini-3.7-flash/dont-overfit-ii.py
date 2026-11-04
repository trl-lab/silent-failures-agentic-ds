import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/dont-overfit-ii/data/test.parquet')

# Prepare features and target
feature_cols = [col for col in train_df.columns if col not in ['id', 'target']]
X = train_df[feature_cols].values
y = train_df['target'].values.astype(int)
X_test = test_df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Cross-validation ensemble with regularized linear models (standard best practice for Don't Overfit II)
n_splits = 10
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

test_preds = np.zeros(len(X_test))

models = [
    LogisticRegression(penalty='l1', C=0.1, solver='liblinear', class_weight='balanced', random_state=42),
    LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.1, solver='saga', class_weight='balanced', max_iter=2000, random_state=42),
    LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', class_weight='balanced', random_state=42)
]

total_models = 0
for train_idx, val_idx in rskf.split(X_scaled, y):
    X_train_fold, y_train_fold = X_scaled[train_idx], y[train_idx]
    
    for model in models:
        model.fit(X_train_fold, y_train_fold)
        test_preds += model.predict_proba(X_test_scaled)[:, 1]
        total_models += 1

test_preds /= total_models

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_preds)),
    'target': test_preds
})
output_df.to_csv('predictions.csv', index=False)