import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop rows with missing target
train_df = train_df.dropna(subset=['sii']).reset_index(drop=True)
train_df['sii'] = train_df['sii'].astype(int)

# Identify features present in both train and test (excluding id and target)
features = [col for col in test_df.columns if col != 'id' and col in train_df.columns]

X = train_df[features].copy()
y = train_df['sii'].values
X_test = test_df[features].copy()

# Categorical column encoding
categorical_cols = [col for col in features if X[col].dtype == 'object' or str(X[col].dtype) == 'category']

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

# Numerical columns with object types conversion
for col in features:
    if col not in categorical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

cat_features_idx = [i for i, col in enumerate(features) if col in categorical_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    val_pred_lgb = lgb_model.predict(X_val)
    test_pred_lgb = lgb_model.predict(X_test)
    
    # CatBoost Regressor
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    X_test_cat = X_test.copy()
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].astype(str).fillna('missing')
        X_val_cat[col] = X_val_cat[col].astype(str).fillna('missing')
        X_test_cat[col] = X_test_cat[col].astype(str).fillna('missing')
        
    cb_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        cat_features=categorical_cols,
        verbose=False
    )
    cb_model.fit(
        X_train_cat, y_train,
        eval_set=(X_val_cat, y_val),
        early_stopping_rounds=50,
        verbose=False
    )
    
    val_pred_cb = cb_model.predict(X_val_cat)
    test_pred_cb = cb_model.predict(X_test_cat)
    
    # Blend predictions for fold
    oof_preds[val_idx] = 0.5 * val_pred_lgb + 0.5 * val_pred_cb
    test_preds += (0.5 * test_pred_lgb + 0.5 * test_pred_cb) / n_splits

# Threshold optimizer for Quadratic Weighted Kappa
class QWKTunedThresholds:
    def __init__(self):
        self.thresholds = [0.5, 1.5, 2.5]

    def _loss(self, thresholds, y_true, y_pred):
        preds = np.digitize(y_pred, thresholds)
        return -cohen_kappa_score(y_true, preds, weights='quadratic')

    def fit(self, y_true, y_pred):
        res = minimize(
            self._loss,
            x0=[0.5, 1.5, 2.5],
            args=(y_true, y_pred),
            method='Nelder-Mead'
        )
        self.thresholds = sorted(res.x)

    def predict(self, y_pred):
        return np.digitize(y_pred, self.thresholds)

optimizer = QWKTunedThresholds()
optimizer.fit(y, oof_preds)

final_preds = optimizer.predict(test_preds)
final_preds = np.clip(final_preds, 0, 3)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'sii': final_preds
})
output_df.to_csv('predictions.csv', index=False)