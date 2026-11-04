import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/petfinder-adoption-prediction/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

y_train = train_df['AdoptionSpeed'].values
train_df = train_df.drop(columns=['AdoptionSpeed'])

n_train = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

df['Name'] = df['Name'].fillna('').astype(str)
df['Description'] = df['Description'].fillna('').astype(str)

df['HasName'] = ((~df['Name'].isin(['', 'Unnamed', 'No Name', 'None'])) & (df['Name'].str.len() > 1)).astype(int)
df['NameLen'] = df['Name'].str.len()

df['DescLen'] = df['Description'].str.len()
df['DescWordCount'] = df['Description'].apply(lambda x: len(x.split()))

df['IsPurebred'] = ((df['Breed2'] == 0) | (df['Breed2'] == df['Breed1'])).astype(int)
df['NumColors'] = (df['Color1'] > 0).astype(int) + (df['Color2'] > 0).astype(int) + (df['Color3'] > 0).astype(int)

df['FeePerPet'] = df['Fee'] / df['Quantity'].clip(lower=1)
df['IsFree'] = (df['Fee'] == 0).astype(int)

rescuer_counts = df['RescuerID'].value_counts().to_dict()
df['RescuerPetCount'] = df['RescuerID'].map(rescuer_counts).fillna(1)

for col in ['Breed1', 'Breed2', 'State']:
    freq = df[col].value_counts().to_dict()
    df[f'{col}_freq'] = df[col].map(freq)

breed_aggs = df.groupby('Breed1')[['Age', 'Fee', 'PhotoAmt']].agg(['mean', 'std']).fillna(0)
breed_aggs.columns = [f'Breed1_{col}_{stat}' for col, stat in breed_aggs.columns]
df = df.merge(breed_aggs, on='Breed1', how='left')

state_aggs = df.groupby('State')[['Age', 'Fee', 'PhotoAmt']].agg(['mean', 'std']).fillna(0)
state_aggs.columns = [f'State_{col}_{stat}' for col, stat in state_aggs.columns]
df = df.merge(state_aggs, on='State', how='left')

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
desc_tfidf = tfidf.fit_transform(df['Description'])
svd = TruncatedSVD(n_components=16, random_state=42)
desc_svd = svd.fit_transform(desc_tfidf)
for i in range(16):
    df[f'desc_svd_{i}'] = desc_svd[:, i]

drop_cols = ['Name', 'Description', 'PetID', 'RescuerID']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(X_test))

oof_preds_xgb = np.zeros(n_train)
test_preds_xgb = np.zeros(len(X_test))

oof_preds_cat = np.zeros(n_train)
test_preds_cat = np.zeros(len(X_test))

cat_features = [col for col in X_train.columns if X_train[col].dtype.name in ['category', 'object']]

X_train_num = X_train.copy()
X_test_num = X_test.copy()
for col in cat_features:
    X_train_num[col] = X_train_num[col].cat.codes
    X_test_num[col] = X_test_num[col].cat.codes

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    X_tr_n, X_va_n = X_train_num.iloc[train_idx], X_train_num.iloc[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        random_state=42 + fold, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / skf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        random_state=42 + fold, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, tree_method='hist', early_stopping_rounds=50
    )
    model_xgb.fit(X_tr_n, y_tr, eval_set=[(X_va_n, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict(X_va_n)
    test_preds_xgb += model_xgb.predict(X_test_num) / skf.n_splits

    model_cat = CatBoostRegressor(
        iterations=600, learning_rate=0.03, depth=6,
        random_seed=42 + fold, verbose=0, early_stopping_rounds=50
    )
    model_cat.fit(X_tr_n, y_tr, eval_set=(X_va_n, y_va))
    oof_preds_cat[val_idx] = model_cat.predict(X_va_n)
    test_preds_cat += model_cat.predict(X_test_num) / skf.n_splits

oof_blend = 0.4 * oof_preds_lgb + 0.3 * oof_preds_xgb + 0.3 * oof_preds_cat
test_blend = 0.4 * test_preds_lgb + 0.3 * test_preds_xgb + 0.3 * test_preds_cat

def predict_with_thresholds(preds, coef):
    return np.digitize(preds, coef)

def kappa_loss(coef, X, y):
    sorted_coef = np.sort(coef)
    preds = predict_with_thresholds(X, sorted_coef)
    return -cohen_kappa_score(y, preds, weights='quadratic')

initial_coef = [0.5, 1.5, 2.5, 3.5]
res = minimize(kappa_loss, initial_coef, args=(oof_blend, y_train), method='Nelder-Mead')
opt_coef = np.sort(res.x)

final_test_preds = predict_with_thresholds(test_blend, opt_coef)

submission = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'AdoptionSpeed': final_test_preds
})

submission.to_csv('predictions.csv', index=False)