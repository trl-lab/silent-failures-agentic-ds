import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/kin8nm/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "y"
features = [c for c in train_df.columns if c != target_col]

X = train_df[features].copy()
y = train_df[target_col].copy()
X_test = test_df[features].copy()

for col in features:
    if X[col].isnull().any() or X_test[col].isnull().any():
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb.fit(X_tr, y_tr)
    
    cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cat.fit(X_tr, y_tr)
    
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation='relu',
        max_iter=400,
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_tr_scaled, y_tr)
    
    p_lgb = lgb.predict(X_test)
    p_cat = cat.predict(X_test)
    p_xgb = xgb.predict(X_test)
    p_mlp = mlp.predict(X_test_scaled)
    
    fold_pred = (p_lgb * 0.3) + (p_cat * 0.3) + (p_xgb * 0.25) + (p_mlp * 0.15)
    test_preds += fold_pred / kf.n_splits

predictions_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'y': test_preds
})

predictions_df.to_csv("predictions.csv", index=False)