import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy import sparse

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/test.parquet')

# Features and target
target_col = 'target'
feature_cols = [col for col in train_df.columns if col not in ['id', target_col]]

X_train_df = train_df[feature_cols].astype(str)
y_train = train_df[target_col].values
X_test_df = test_df[feature_cols].astype(str)

# One-hot encode all features
ohe = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
X_train_sparse = ohe.fit_transform(X_train_df)
X_test_sparse = ohe.transform(X_test_df)

# Train Logistic Regression with K-Fold cross validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(X_test_sparse.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_sparse, y_train)):
    X_tr, y_tr = X_train_sparse[train_idx], y_train[train_idx]
    
    model = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_tr, y_tr)
    
    test_preds += model.predict_proba(X_test_sparse)[:, 1] / n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'target': test_preds
})
output_df.to_csv('predictions.csv', index=False)