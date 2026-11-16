import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'strength'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Feature engineering
def add_features(df):
    df = df.copy()
    # Total binder/cementitious materials
    df['total_binder'] = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
    # Water-to-binder ratio
    df['water_binder_ratio'] = df['water'] / (df['total_binder'] + 1e-6)
    # Water-to-cement ratio
    df['water_cement_ratio'] = df['water'] / (df['cement'] + 1e-6)
    # Total aggregate
    df['total_aggregate'] = df['coarse_aggregate'] + df['fine_aggregate']
    # Aggregate to binder ratio
    df['agg_binder_ratio'] = df['total_aggregate'] / (df['total_binder'] + 1e-6)
    # Coarse to fine aggregate ratio
    df['coarse_fine_ratio'] = df['coarse_aggregate'] / (df['fine_aggregate'] + 1e-6)
    # Log transformed age
    df['log_age'] = np.log1p(df['age'])
    return df

X_feat = add_features(X)
X_test_feat = add_features(X_test)

# Fill missing values if any
X_feat = X_feat.fillna(X_feat.median())
X_test_feat = X_test_feat.fillna(X_feat.median())

# Model definitions
def get_models(seed=42):
    return {
        'lgb': LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1
        ),
        'xgb': XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0
        ),
        'cat': CatBoostRegressor(
            iterations=700,
            learning_rate=0.04,
            depth=6,
            random_seed=seed,
            verbose=False
        ),
        'et': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=seed,
            n_jobs=-1
        ),
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            random_state=seed,
            n_jobs=-1
        )
    }

# 10-Fold Cross-Validation Ensemble
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test_feat))
oof_preds = np.zeros(len(X_feat))

model_weights = {
    'lgb': 0.25,
    'xgb': 0.25,
    'cat': 0.25,
    'et': 0.15,
    'rf': 0.10
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_feat, y)):
    X_tr, y_tr = X_feat.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_feat.iloc[val_idx], y.iloc[val_idx]
    
    fold_val_pred = np.zeros(len(val_idx))
    fold_test_pred = np.zeros(len(X_test_feat))
    
    models = get_models(seed=42 + fold)
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_va)
        t_pred = model.predict(X_test_feat)
        
        weight = model_weights[name]
        fold_val_pred += weight * val_pred
        fold_test_pred += weight * t_pred
    
    oof_preds[val_idx] = fold_val_pred
    test_preds += fold_test_pred / n_splits

rmse_cv = np.sqrt(mean_squared_error(y, oof_preds))

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'strength': test_preds
})

output_df.to_csv('predictions.csv', index=False)