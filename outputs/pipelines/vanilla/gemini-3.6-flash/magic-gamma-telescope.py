import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/magic-gamma-telescope/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

def map_target(series):
    if series.dtype == object or series.dtype.name == 'category':
        unique_vals = series.unique()
        target_map = {}
        for v in unique_vals:
            v_str = str(v).lower()
            if 'g' in v_str or '1' in v_str or 'sig' in v_str:
                target_map[v] = 1
            else:
                target_map[v] = 0
        return series.map(target_map).astype(int)
    else:
        vals = set(series.dropna().unique())
        if vals.issubset({0, 1}):
            return series.astype(int)
        else:
            min_v = series.min()
            return (series > min_v).astype(int)

y = map_target(train_df['class'])

def create_features(df):
    X = df.copy()
    if 'class' in X.columns:
        X = X.drop(columns=['class'])

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    fLength = np.maximum(X['fLength'].fillna(0).values, 1e-6)
    fWidth = np.maximum(X['fWidth'].fillna(0).values, 1e-6)
    fSize = X['fSize'].fillna(0).values
    fConc = X['fConc'].fillna(0).values
    fConc1 = X['fConc1'].fillna(0).values
    fAsym = X['fAsym'].fillna(0).values
    fM3Long = X['fM3Long'].fillna(0).values
    fM3Trans = X['fM3Trans'].fillna(0).values
    fAlpha = X['fAlpha'].fillna(0).values
    fDist = X['fDist'].fillna(0).values

    X['aspect_ratio'] = fWidth / fLength
    X['area_approx'] = fLength * fWidth
    X['size_per_length'] = fSize / fLength
    X['size_per_width'] = fSize / fWidth
    X['conc_ratio'] = np.where(fConc != 0, fConc1 / np.maximum(np.abs(fConc), 1e-6), 0)
    X['dist_per_length'] = fDist / fLength
    X['dist_per_width'] = fDist / fWidth

    alpha_rad = np.radians(fAlpha)
    X['sin_alpha'] = np.sin(alpha_rad)
    X['cos_alpha'] = np.cos(alpha_rad)
    X['abs_alpha'] = np.abs(fAlpha)

    X['m3_ratio'] = np.where(np.abs(fM3Trans) > 1e-6, fM3Long / np.maximum(np.abs(fM3Trans), 1e-6), 0)
    X['m3_product'] = fM3Long * fM3Trans
    X['asym_per_length'] = fAsym / fLength

    X['log_size'] = np.log1p(np.maximum(fSize, 0))
    X['log_length'] = np.log1p(np.maximum(fLength, 0))
    X['log_width'] = np.log1p(np.maximum(fWidth, 0))
    X['log_dist'] = np.log1p(np.maximum(fDist, 0))

    X = X.replace([np.inf, -np.inf], np.nan)
    return X

X_train = create_features(train_df)
X_test = create_features(test_df)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

    model_lgb = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test)[:, 1] / n_splits

    model_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        tree_method='hist',
        early_stopping_rounds=100,
        n_jobs=-1
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    model_cb = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=100,
        thread_count=-1
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        verbose=False
    )
    cb_preds += model_cb.predict_proba(X_test)[:, 1] / n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cb_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})

submission.to_csv('predictions.csv', index=False)