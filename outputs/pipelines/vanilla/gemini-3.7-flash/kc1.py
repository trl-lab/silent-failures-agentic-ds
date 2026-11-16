import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/kc1/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'defects'
X = train_df.drop(columns=[target_col])
y = train_df[target_col].astype(int)
X_test = test_df.copy()

# Setup stratified k-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]
    
    # Model 1: LightGBM
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_train_fold, y_train_fold)
    
    # Model 2: CatBoost
    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=False
    )
    cat.fit(X_train_fold, y_train_fold)
    
    # Model 3: XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        eval_metric='logloss',
        random_state=42 + fold
    )
    xgb.fit(X_train_fold, y_train_fold)
    
    # Fold predictions on test set
    fold_preds = (
        lgb.predict_proba(X_test)[:, 1] +
        cat.predict_proba(X_test)[:, 1] +
        xgb.predict_proba(X_test)[:, 1]
    ) / 3.0
    
    test_preds += fold_preds / skf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'defects': test_preds
})
output_df.to_csv('predictions.csv', index=False)