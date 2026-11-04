import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/allstate-claims-severity/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train['loss'].values
shift = 200.0
y_log = np.log(y + shift)

drop_cols = ['id', 'loss']
features = [c for c in train.columns if c not in drop_cols]
cat_cols = [c for c in features if c.startswith('cat')]

all_df = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)

for c in cat_cols:
    all_df[c] = all_df[c].astype('category').cat.codes

train_x = all_df.iloc[:len(train)].copy()
test_x = all_df.iloc[len(train):].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(test_x))
cat_test_preds = np.zeros(len(test_x))

lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 12,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 20,
    'verbosity': -1,
    'n_jobs': -1,
    'random_state': 42
}

cat_params = {
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'learning_rate': 0.05,
    'depth': 7,
    'iterations': 1200,
    'verbose': 0,
    'random_seed': 42,
    'thread_count': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x, y_log)):
    X_tr, y_tr = train_x.iloc[train_idx], y_log[train_idx]
    X_va, y_va = train_x.iloc[val_idx], y_log[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
    
    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    lgb_test_preds += model_lgb.predict(test_x, num_iteration=model_lgb.best_iteration) / kf.n_splits

    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    cat_test_preds += model_cat.predict(test_x) / kf.n_splits

final_preds_log = 0.5 * lgb_test_preds + 0.5 * cat_test_preds
final_preds = np.exp(final_preds_log) - shift
final_preds = np.clip(final_preds, a_min=0.0, a_max=None)

output = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'loss': final_preds
})

output.to_csv('predictions.csv', index=False)