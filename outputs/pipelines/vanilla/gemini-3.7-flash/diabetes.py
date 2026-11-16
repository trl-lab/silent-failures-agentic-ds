import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/diabetes/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Preprocess data
target_col = 'class'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = (train_df[target_col].astype(str) == 'tested_positive').astype(int)
X_test = test_df[feature_cols].copy()

# Replace physiologically impossible 0s with NaN for imputation/tree handling
cols_with_zero_missing = ['plas', 'pres', 'skin', 'insu', 'mass']
for col in cols_with_zero_missing:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)
        X_test[col] = X_test[col].replace(0, np.nan)

# 3. Model Ensembling with Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

models = [
    ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, verbose=-1)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, eval_metric='logloss', random_state=42)),
    ('cat', CatBoostClassifier(iterations=150, learning_rate=0.03, depth=3, verbose=0, random_state=42)),
    ('gb', Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42))])),
    ('rf', Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42))])),
    ('lr', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LogisticRegression(C=1.0, random_state=42))]))
]

for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    
    for name, model in models:
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_test)[:, 1]
        test_preds += preds / (len(models) * n_splits)

# 4. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'class': test_preds
})

submission_df.to_csv('predictions.csv', index=False)