import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# File paths
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/credit-g/data/test.parquet'

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Encode target ('bad' -> 0, 'good' -> 1 if binary string)
if y.dtype == 'category' or y.dtype == 'object':
    classes = sorted(y.dropna().unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = y.map(class_to_idx).astype(int)
else:
    y_encoded = y.astype(int)

# Identify column types
categorical_cols = [c for c in X.columns if X[c].dtype == 'category' or X[c].dtype == 'object']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Cast categorical columns to string for consistent handling
for c in categorical_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ]
)

X_prep = preprocessor.fit_transform(X)
X_test_prep = preprocessor.transform(X_test)

# Cross-validated ensembling
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

models = [
    ('lgb', LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    ('xgb', XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')),
    ('cat', CatBoostClassifier(iterations=200, learning_rate=0.03, depth=4, random_seed=42, verbose=0)),
    ('hgb', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_depth=4, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))
]

for train_idx, val_idx in cv.split(X_prep, y_encoded):
    X_tr, y_tr = X_prep[train_idx], y_encoded.iloc[train_idx]
    
    fold_preds = np.zeros(len(X_test))
    for name, model in models:
        model.fit(X_tr, y_tr)
        fold_preds += model.predict_proba(X_test_prep)[:, 1] / len(models)
        
    test_preds += fold_preds / cv.n_splits

# Output predictions
submission = pd.DataFrame({
    'id': range(len(test_preds)),
    'class': test_preds
})

submission.to_csv('predictions.csv', index=False)