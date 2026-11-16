import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
target_col = 'Class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Encode target to 0 and 1
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y.astype(str))

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# Preprocess categorical features for tree-based models
for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    le = LabelEncoder()
    # Fit on combined values to handle all categories
    le.fit(list(X[col].unique()) + list(X_test[col].unique()))
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# K-Fold Cross-Validation Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train_f, y_train_f = X.iloc[train_idx], y_encoded[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y_encoded[val_idx]
    
    # LightGBM Classifier
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[]
    )
    
    # CatBoost Classifier
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_train_f, y_train_f,
        eval_set=(X_val_f, y_val_f),
        verbose=False
    )
    
    # Predict on test set
    lgb_test_pred = lgb_model.predict_proba(X_test)[:, 1]
    cb_test_pred = cb_model.predict_proba(X_test)[:, 1]
    
    test_preds += (0.5 * lgb_test_pred + 0.5 * cb_test_pred) / n_splits

# Create submission dataframe
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Class': test_preds
})

submission.to_csv('predictions.csv', index=False)