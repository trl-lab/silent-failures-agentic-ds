import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class_number_of_rings"
if target_col not in train_df.columns:
    candidates = [c for c in train_df.columns if "ring" in c.lower() or "class" in c.lower()]
    if candidates:
        target_col = candidates[0]

y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

def create_features(df):
    df = df.copy()
    cols_map = {c.lower().replace('_', ' ').strip(): c for c in df.columns}
    
    length_col = cols_map.get('length')
    diam_col = cols_map.get('diameter')
    height_col = cols_map.get('height')
    whole_w = cols_map.get('whole weight') or cols_map.get('whole_weight')
    shucked_w = cols_map.get('shucked weight') or cols_map.get('shucked_weight')
    viscera_w = cols_map.get('viscera weight') or cols_map.get('viscera_weight')
    shell_w = cols_map.get('shell weight') or cols_map.get('shell_weight')
    
    eps = 1e-6
    if length_col and diam_col and height_col:
        df['volume'] = df[length_col] * df[diam_col] * df[height_col]
        df['length_to_diam'] = df[length_col] / (df[diam_col] + eps)
        df['length_to_height'] = df[length_col] / (df[height_col] + eps)
        df['diam_to_height'] = df[diam_col] / (df[height_col] + eps)
        
    if whole_w and shucked_w and viscera_w and shell_w:
        df['weight_diff'] = df[whole_w] - (df[shucked_w] + df[viscera_w] + df[shell_w])
        df['shucked_ratio'] = df[shucked_w] / (df[whole_w] + eps)
        df['viscera_ratio'] = df[viscera_w] / (df[whole_w] + eps)
        df['shell_ratio'] = df[shell_w] / (df[whole_w] + eps)
        
    if whole_w and 'volume' in df.columns:
        df['density'] = df[whole_w] / (df['volume'] + eps)
    if shell_w and 'volume' in df.columns:
        df['shell_density'] = df[shell_w] / (df['volume'] + eps)

    df = pd.get_dummies(df, drop_first=True)
    return df

combined_feat = create_features(combined)

n_train = len(X_train_raw)
X_train = combined_feat.iloc[:n_train].reset_index(drop=True)
X_test = combined_feat.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
hgb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train, y):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        tree_method='hist'
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    cat_preds += model_cat.predict(X_test) / kf.n_splits

    model_hgb = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )
    model_hgb.fit(X_tr, y_tr)
    hgb_preds += model_hgb.predict(X_test) / kf.n_splits

final_preds = 0.35 * lgb_preds + 0.35 * cat_preds + 0.20 * xgb_preds + 0.10 * hgb_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Class_number_of_rings': final_preds
})

output_df.to_csv('predictions.csv', index=False)