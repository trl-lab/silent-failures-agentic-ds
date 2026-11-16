import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/istanbul-stock-exchange/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'ISE'
y_train = train_df[target_col].values

# Identify feature columns present in both train and test (excluding target)
feature_cols = [col for col in test_df.columns if col in train_df.columns and col != target_col]

def preprocess(df, cols):
    df_feat = pd.DataFrame(index=df.index)
    for col in cols:
        if col.lower() == 'date':
            date_series = pd.to_datetime(df[col], errors='coerce')
            df_feat['year'] = date_series.dt.year.fillna(2010).astype(int)
            df_feat['month'] = date_series.dt.month.fillna(6).astype(int)
            df_feat['day'] = date_series.dt.day.fillna(15).astype(int)
            df_feat['dayofweek'] = date_series.dt.dayofweek.fillna(2).astype(int)
            df_feat['dayofyear'] = date_series.dt.dayofyear.fillna(180).astype(int)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df_feat[col] = df[col].astype(float)
        else:
            try:
                df_feat[col] = pd.to_numeric(df[col])
            except Exception:
                df_feat[col] = df[col].astype('category').cat.codes
    return df_feat

X_train_df = preprocess(train_df, feature_cols)
X_test_df = preprocess(test_df, feature_cols)

# Ensure matching columns
common_cols = [c for c in X_train_df.columns if c in X_test_df.columns]
X_train = X_train_df[common_cols].copy()
X_test = X_test_df[common_cols].copy()

# Fill missing values
for c in common_cols:
    median_val = X_train[c].median()
    if pd.isna(median_val):
        median_val = 0.0
    X_train[c] = X_train[c].fillna(median_val)
    X_test[c] = X_test[c].fillna(median_val)

X_train_np = X_train.values
X_test_np = X_test.values

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'ridge': Ridge(alpha=1.0),
    'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
    'rf': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    'et': ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    'lgb': LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, num_leaves=15, random_state=42, verbose=-1),
    'xgb': XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0),
    'cat': CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, random_seed=42, verbose=0)
}

test_preds = np.zeros(len(test_df))

for name, model in models.items():
    model_test_preds = np.zeros(len(test_df))
    for train_idx, val_idx in kf.split(X_train_np):
        if name in ['ridge', 'elastic']:
            X_tr, y_tr = X_train_scaled[train_idx], y_train[train_idx]
            X_te = X_test_scaled
        else:
            X_tr, y_tr = X_train_np[train_idx], y_train[train_idx]
            X_te = X_test_np

        model.fit(X_tr, y_tr)
        model_test_preds += model.predict(X_te) / kf.n_splits
    test_preds += model_test_preds / len(models)

# Create submission dataframe
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'ISE': test_preds
})

submission.to_csv('predictions.csv', index=False)