import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'class'

X_train = train.drop(columns=[target_col])
y_train_raw = train[target_col]

X_test = test.drop(columns=[target_col], errors='ignore')

le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train_raw)

feature_cols = list(X_train.columns)

df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in feature_cols:
    df_all[col] = df_all[col].fillna('missing').astype(str)
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

X_train_enc = df_all.iloc[:len(X_train)].copy()
X_test_enc = df_all.iloc[len(X_train):].copy()

lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.08,
    depth=8,
    random_seed=42,
    verbose=0,
    thread_count=-1
)

xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

lgb.fit(X_train_enc, y_train)
cb.fit(X_train_enc, y_train)
xgb.fit(X_train_enc, y_train)

p_lgb = lgb.predict_proba(X_test_enc)
p_cb = cb.predict_proba(X_test_enc)
p_xgb = xgb.predict_proba(X_test_enc)

p_ens = (p_lgb + p_cb + p_xgb) / 3.0
y_pred_indices = np.argmax(p_ens, axis=1)

y_pred = le_y.inverse_transform(y_pred_indices)

predictions_df = pd.DataFrame({
    'id': np.arange(len(y_pred)),
    'class': y_pred
})

predictions_df.to_csv('predictions.csv', index=False)