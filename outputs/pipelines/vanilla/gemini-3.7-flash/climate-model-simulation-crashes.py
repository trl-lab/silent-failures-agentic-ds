import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Prepare target
target_col = 'outcome'
y = train_df[target_col].astype(int).values

feature_cols = [c for c in train_df.columns if c != target_col]
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values if any
for col in feature_cols:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

X_mat = X.values
X_test_mat = X_test.values

# Stratified K-Fold Cross-Validation and Ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y)):
    X_tr, y_tr = X_mat[train_idx], y[train_idx]
    X_va, y_va = X_mat[val_idx], y[val_idx]
    
    # Scale for linear models / trees
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_test_s = scaler.transform(X_test_mat)
    
    # LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        random_state=42 + fold,
        verbose=-1
    )
    lgbm.fit(X_tr, y_tr)
    p_lgb = lgbm.predict_proba(X_test_mat)[:, 1]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        eval_metric='logloss',
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    p_xgb = xgb_model.predict_proba(X_test_mat)[:, 1]
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=4,
        verbose=0,
        random_state=42 + fold
    )
    cat_model.fit(X_tr, y_tr)
    p_cat = cat_model.predict_proba(X_test_mat)[:, 1]
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42 + fold
    )
    rf_model.fit(X_tr, y_tr)
    p_rf = rf_model.predict_proba(X_test_mat)[:, 1]
    
    # Logistic Regression
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    lr_model.fit(X_tr_s, y_tr)
    p_lr = lr_model.predict_proba(X_test_s)[:, 1]
    
    fold_pred = 0.25 * p_lgb + 0.25 * p_xgb + 0.25 * p_cat + 0.15 * p_rf + 0.10 * p_lr
    test_preds += fold_pred / skf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'outcome': test_preds
})

output_df.to_csv('predictions.csv', index=False)