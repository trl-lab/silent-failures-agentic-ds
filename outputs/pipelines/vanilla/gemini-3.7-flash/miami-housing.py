import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

drop_cols = ['SALE_PRC', 'PARCELNO']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X = train_df[feature_cols].copy()
y = train_df['SALE_PRC'].values
X_test = test_df[feature_cols].copy()

# Feature engineering
for df in [X, X_test]:
    df['lvg_to_land_ratio'] = df['TOT_LVG_AREA'] / (df['LND_SQFOOT'] + 1.0)
    df['spec_feat_per_sqft'] = df['SPEC_FEAT_VAL'] / (df['TOT_LVG_AREA'] + 1.0)
    df['dist_sum'] = df['OCEAN_DIST'] + df['WATER_DIST'] + df['CNTR_DIST'] + df['SUBCNTR_DI']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    lgb = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    
    cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)

    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb.fit(X_tr, y_tr)

    fold_test_pred = (
        lgb.predict(X_test) * 0.4 +
        cb.predict(X_test) * 0.35 +
        xgb.predict(X_test) * 0.25
    )
    test_preds += fold_test_pred / kf.n_splits

submission = pd.DataFrame({
    'id': range(len(test_preds)),
    'SALE_PRC': test_preds
})
submission.to_csv('predictions.csv', index=False)