import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/KDDCup09_appetency/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'APPETENCY'

# Extract target and map to binary (0, 1)
y_raw = train_df[target_col]
try:
    y = (pd.to_numeric(y_raw) > 0).astype(int)
except Exception:
    y = y_raw.astype(str).apply(lambda x: 1 if ('1' in x and '-' not in x) else 0).astype(int)

X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Identify categorical and object columns
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or isinstance(X[c].dtype, pd.CategoricalDtype)]

# Convert categorical columns to uniform string categories
for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

# K-Fold Cross Validation & Ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        importance_type='gain'
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'APPETENCY': test_preds
})
output_df.to_csv('predictions.csv', index=False)