import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat-ii/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target and drop id
y_train = train_df['target'].values
features = [col for col in train_df.columns if col not in ['id', 'target']]

X_train = train_df[features].copy()
X_test = test_df[features].copy()

# Fill missing values and treat all columns as categorical strings
for col in features:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

# One-Hot Encoding
ohe = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)

# Stratified K-Fold Training with Logistic Regression
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds = np.zeros(X_test_ohe.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_ohe, y_train)):
    X_tr, y_tr = X_train_ohe[train_idx], y_train[train_idx]
    
    model = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_tr, y_tr)
    
    test_preds += model.predict_proba(X_test_ohe)[:, 1] / n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'target': test_preds
})
output_df.to_csv('predictions.csv', index=False)