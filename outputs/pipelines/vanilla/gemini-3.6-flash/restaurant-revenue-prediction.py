import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/restaurant-revenue-prediction/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Target
y = train['revenue'].values
y_log = np.log1p(y)

# Combine datasets for preprocessing
train_len = len(train)
df = pd.concat([train.drop(columns=['revenue'], errors='ignore'), test], ignore_index=True)

# Parse Date
if 'Open Date' in df.columns:
    df['Open Date'] = pd.to_datetime(df['Open Date'])
    ref_date = pd.to_datetime('2015-01-01')
    df['days_open'] = (ref_date - df['Open Date']).dt.days
    df['year'] = df['Open Date'].dt.year
    df['month'] = df['Open Date'].dt.month
    df['day'] = df['Open Date'].dt.day
    df['dayofweek'] = df['Open Date'].dt.dayofweek
    df.drop(columns=['Open Date'], inplace=True)

# Drop ID columns
for col in ['Id', 'id', 'ID']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Categorical Encodings
if 'City Group' in df.columns:
    df['City Group'] = df['City Group'].map({'Big Cities': 1, 'Other': 0}).fillna(0)

if 'Type' in df.columns:
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)

if 'City' in df.columns:
    city_counts = df['City'].value_counts()
    df['City_freq'] = df['City'].map(city_counts)
    df.drop(columns=['City'], inplace=True)

# P-variable Aggregations
p_cols = [c for c in df.columns if c.startswith('P') and c[1:].isdigit()]
if p_cols:
    df['P_sum'] = df[p_cols].sum(axis=1)
    df['P_mean'] = df[p_cols].mean(axis=1)
    df['P_std'] = df[p_cols].std(axis=1)
    df['P_max'] = df[p_cols].max(axis=1)
    df['P_min'] = df[p_cols].min(axis=1)
    df['P_zeros'] = (df[p_cols] == 0).sum(axis=1)

# Ensure numeric types
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

df = df.fillna(0)

# Train/Test Split
X_train = df.iloc[:train_len].copy().values
X_test = df.iloc[train_len:].copy().values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross-Validation Ensembling
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
n_splits_total = 10 * 5

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
rf_preds = np.zeros(len(X_test))
et_preds = np.zeros(len(X_test))
ridge_preds = np.zeros(len(X_test))

for train_idx, val_idx in rkf.split(X_train, y_log):
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train[train_idx], y_log[train_idx])
    lgb_preds += np.expm1(lgb_model.predict(X_test)) / n_splits_total

    # CatBoost
    cb_model = cb.CatBoostRegressor(
        iterations=150,
        learning_rate=0.03,
        depth=3,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_train[train_idx], y_log[train_idx])
    cb_preds += np.expm1(cb_model.predict(X_test)) / n_splits_total

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train[train_idx], y_log[train_idx])
    xgb_preds += np.expm1(xgb_model.predict(X_test)) / n_splits_total

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=4, max_features=0.5, random_state=42)
    rf_model.fit(X_train[train_idx], y_log[train_idx])
    rf_preds += np.expm1(rf_model.predict(X_test)) / n_splits_total

    # Extra Trees
    et_model = ExtraTreesRegressor(n_estimators=100, max_depth=4, max_features=0.5, random_state=42)
    et_model.fit(X_train[train_idx], y_log[train_idx])
    et_preds += np.expm1(et_model.predict(X_test)) / n_splits_total

    # Ridge
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled[train_idx], y_log[train_idx])
    ridge_preds += np.expm1(ridge_model.predict(X_test_scaled)) / n_splits_total

# Weighted Average
final_preds = (
    0.20 * lgb_preds +
    0.20 * cb_preds +
    0.20 * xgb_preds +
    0.15 * rf_preds +
    0.15 * et_preds +
    0.10 * ridge_preds
)

# Post-processing
min_rev = y.min() * 0.5
final_preds = np.clip(final_preds, min_rev, None)

# Save predictions
output = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'revenue': final_preds
})
output.to_csv('predictions.csv', index=False)