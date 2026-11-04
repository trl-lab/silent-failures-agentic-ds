import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/bank-marketing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class'
y_raw = train_df[target_col]
classes = sorted(y_raw.unique())
pos_class = classes[1]
y = (y_raw == pos_class).astype(int).values

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.drop(columns=[target_col], errors='ignore').copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()

if 'pdays' in df_all.columns:
    df_all['pdays_contacted'] = (df_all['pdays'] != -1).astype(int)

for c in cat_cols:
    df_all[c] = df_all[c].astype('category')

X_tr = df_all.iloc[:n_train].copy()
X_te = df_all.iloc[n_train:].copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_te))
cat_preds = np.zeros(len(X_te))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_train_fold, y_train_fold = X_tr.iloc[trn_idx], y[trn_idx]
    X_val_fold, y_val_fold = X_tr.iloc[val_idx], y[val_idx]
    
    trn_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=trn_data)
    
    clf_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += clf_lgb.predict(X_te, num_iteration=clf_lgb.best_iteration) / 5.0

    X_tr_cb = X_train_fold.copy()
    X_va_cb = X_val_fold.copy()
    X_te_cb = X_te.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_te_cb[c] = X_te_cb[c].astype(str)
        
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        cat_features=cat_cols
    )
    cb_model.fit(X_tr_cb, y_train_fold, eval_set=(X_va_cb, y_val_fold), early_stopping_rounds=50)
    cat_preds += cb_model.predict_proba(X_te_cb)[:, 1] / 5.0

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Class': final_preds
})

submission.to_csv('predictions.csv', index=False)