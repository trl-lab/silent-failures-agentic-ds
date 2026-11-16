import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/device-covid19serology/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'antibody_truth'

# Encode target: 'Positive' -> 1, 'Negative' -> 0
y = train_df[target_col].astype(str).str.strip().str.lower()
y = (y == 'positive').astype(int).values

X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature Engineering
def preprocess_features(df):
    df = df.copy()
    if 'date_performed' in df.columns:
        dt = pd.to_datetime(df['date_performed'], errors='coerce')
        df['date_year'] = dt.dt.year.fillna(-1).astype(int)
        df['date_month'] = dt.dt.month.fillna(-1).astype(int)
        df['date_day'] = dt.dt.day.fillna(-1).astype(int)
        df['date_dayofweek'] = dt.dt.dayofweek.fillna(-1).astype(int)
    
    # Try converting sample_no to numeric if possible
    if 'sample_no' in df.columns:
        df['sample_no_num'] = pd.to_numeric(df['sample_no'], errors='coerce').fillna(-1)
        
    return df

X = preprocess_features(X)
X_test = preprocess_features(X_test)

# Fill missing values and convert object columns to string
cat_cols = []
for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
        X[col] = X[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)
        cat_cols.append(col)
    else:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Ordinal encode categorical features for LightGBM
X_encoded = X.copy()
X_test_encoded = X_test.copy()

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if cat_cols:
    X_encoded[cat_cols] = oe.fit_transform(X[cat_cols])
    X_test_encoded[cat_cols] = oe.transform(X_test[cat_cols])

# K-Fold Cross Validation & Model Training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # CatBoost
    X_tr_cb, y_tr_cb = X.iloc[train_idx], y[train_idx]
    X_va_cb, y_va_cb = X.iloc[val_idx], y[val_idx]
    
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr_cb, y_tr_cb, eval_set=(X_va_cb, y_va_cb), early_stopping_rounds=50, verbose=False)
    cb_preds += cb_model.predict_proba(X_test)[:, 1] / n_splits
    
    # LightGBM
    X_tr_lgb, y_tr_lgb = X_encoded.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va_lgb = X_encoded.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr_lgb,
        eval_set=[(X_va_lgb, y_va_lgb)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict_proba(X_test_encoded)[:, 1] / n_splits

# Ensemble predictions
final_preds = 0.5 * cb_preds + 0.5 * lgb_preds

# Save submission
submission = pd.DataFrame({
    'id': range(len(final_preds)),
    'antibody_truth': final_preds
})

submission.to_csv('predictions.csv', index=False)