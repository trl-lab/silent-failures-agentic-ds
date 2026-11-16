import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'DEATH_EVENT'
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Handle any missing values
for col in X.columns:
    if X[col].dtype in ['float64', 'int64', 'int32', 'float32', 'uint8']:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    else:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
        X[col] = X[col].fillna(mode_val).astype(str)
        X_test[col] = X_test[col].fillna(mode_val).astype(str)

# Stratified K-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    
    # Model 2: XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    
    # Model 3: CatBoost
    cat_model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_seed=42 + fold
    )
    cat_model.fit(X_tr, y_tr)
    
    # Model 4: Logistic Regression
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    lr_model.fit(X_tr_scaled, y_tr)
    
    # Blend fold predictions
    fold_test_pred = (
        0.30 * lgb_model.predict_proba(X_test)[:, 1] +
        0.30 * xgb_model.predict_proba(X_test)[:, 1] +
        0.30 * cat_model.predict_proba(X_test)[:, 1] +
        0.10 * lr_model.predict_proba(X_test_scaled)[:, 1]
    )
    
    test_preds += fold_test_pred / n_splits

# Generate output submission
submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'DEATH_EVENT': test_preds
})

submission.to_csv('predictions.csv', index=False)