import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "HPSA Score"
y = train_df[target_col].values
X_train_df = train_df.drop(columns=[target_col])
X_test_df = test_df.copy()

n_train = len(train_df)
combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)

# Parse date columns
for col in list(combined.columns):
    if combined[col].dtype == 'object' or combined[col].dtype == 'string':
        sample = combined[col].dropna()
        if len(sample) > 0:
            try:
                sample_dt = pd.to_datetime(sample.head(50), errors='coerce')
                if sample_dt.notna().sum() > 25:
                    parsed_dt = pd.to_datetime(combined[col], errors='coerce')
                    combined[f"{col}_year"] = parsed_dt.dt.year
                    combined[f"{col}_month"] = parsed_dt.dt.month
                    combined[f"{col}_day"] = parsed_dt.dt.day
                    combined[f"{col}_dayofweek"] = parsed_dt.dt.dayofweek
            except Exception:
                pass

cat_cols = []
num_cols = []

for col in combined.columns:
    if combined[col].dtype == 'object' or combined[col].dtype == 'string' or combined[col].dtype == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in cat_cols:
    combined[col] = combined[col].fillna('__MISSING__').astype(str)
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
    combined[col] = combined[col].astype('category').cat.codes

X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / 5.0

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / 5.0

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / 5.0

final_preds = 0.4 * lgb_preds + 0.3 * cat_preds + 0.3 * xgb_preds

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'HPSA Score': final_preds
})

submission.to_csv('predictions.csv', index=False)