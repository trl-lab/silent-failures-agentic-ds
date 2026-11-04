import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Load dataset
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/test.parquet')

target_col = 'ISE'
y = train_df[target_col].values

feature_cols = [c for c in test_df.columns if c != target_col]

def preprocess(df, feature_cols):
    df_feat = df[feature_cols].copy()
    
    # Identify date columns
    for col in list(df_feat.columns):
        if 'date' in col.lower() or df_feat[col].dtype == 'datetime64[ns]' or df_feat[col].dtype == 'object':
            try:
                d = pd.to_datetime(df_feat[col])
                df_feat[col + '_year'] = d.dt.year
                df_feat[col + '_month'] = d.dt.month
                df_feat[col + '_day'] = d.dt.day
                df_feat[col + '_dayofweek'] = d.dt.dayofweek
                df_feat[col + '_dayofyear'] = d.dt.dayofyear
                df_feat[col + '_quarter'] = d.dt.quarter
                df_feat = df_feat.drop(columns=[col])
            except Exception:
                pass
                
    # Convert remaining columns to numeric
    for col in df_feat.columns:
        if df_feat[col].dtype == 'object':
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
            
    num_cols = list(df_feat.columns)
    idx_cols = [c for c in num_cols if not any(k in c.lower() for k in ['year', 'month', 'day', 'quarter'])]
    
    # Calculate row aggregates across financial market indices
    if len(idx_cols) > 1:
        df_feat['row_mean'] = df_feat[idx_cols].mean(axis=1)
        df_feat['row_std'] = df_feat[idx_cols].std(axis=1)
        df_feat['row_min'] = df_feat[idx_cols].min(axis=1)
        df_feat['row_max'] = df_feat[idx_cols].max(axis=1)
        df_feat['row_median'] = df_feat[idx_cols].median(axis=1)
        
    return df_feat

X_train_full = preprocess(train_df, feature_cols)
X_test_full = preprocess(test_df, feature_cols)

common_cols = [c for c in X_train_full.columns if c in X_test_full.columns]
X_train = X_train_full[common_cols].copy()
X_test = X_test_full[common_cols].copy()

# Fill missing values
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-Validation Ensembling
kf = KFold(n_splits=10, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    X_tr_sc, X_va_sc = X_train_scaled[train_idx], X_train_scaled[val_idx]
    
    # Ridge
    m1 = Ridge(alpha=10.0)
    m1.fit(X_tr_sc, y_tr)
    p1_test = m1.predict(X_test_scaled)
    
    # ElasticNet
    m2 = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    m2.fit(X_tr_sc, y_tr)
    p2_test = m2.predict(X_test_scaled)
    
    # LightGBM
    m3 = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=3,
        num_leaves=7, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )
    m3.fit(X_tr, y_tr)
    p3_test = m3.predict(X_test)
    
    # CatBoost
    m4 = CatBoostRegressor(
        iterations=200, learning_rate=0.03, depth=4,
        l2_leaf_reg=3.0, random_state=42, verbose=0
    )
    m4.fit(X_tr, y_tr)
    p4_test = m4.predict(X_test)
    
    # ExtraTrees
    m5 = ExtraTreesRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    m5.fit(X_tr, y_tr)
    p5_test = m5.predict(X_test)
    
    fold_test_pred = (0.30 * p1_test + 0.20 * p2_test + 0.20 * p3_test + 0.20 * p4_test + 0.10 * p5_test)
    test_preds += fold_test_pred / kf.n_splits

# Save predictions
output = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'ISE': test_preds
})
output.to_csv('predictions.csv', index=False)