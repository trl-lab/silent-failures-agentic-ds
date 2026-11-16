import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "PAYMENTS_BY_ISSUER"
y = train_df[target_col].values.astype(np.float64)
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
df = pd.concat([train_df, test_df], ignore_index=True)

if "id" in df.columns:
    df = df.drop(columns=["id"])

# 1. Datetime feature extraction
for col in list(df.columns):
    if (
        pd.api.types.is_datetime64_any_dtype(df[col])
        or "date" in col.lower()
        or "year" in col.lower()
    ):
        try:
            date_series = pd.to_datetime(df[col], errors="coerce")
            if date_series.notnull().sum() > 0:
                df[f"{col}_year"] = date_series.dt.year
                df[f"{col}_month"] = date_series.dt.month
                df[f"{col}_day"] = date_series.dt.day
                df[f"{col}_dayofweek"] = date_series.dt.dayofweek
                df[f"{col}_quarter"] = date_series.dt.quarter
                if df[col].dtype == object or "date" in col.lower():
                    df = df.drop(columns=[col])
        except Exception:
            pass

# Identify base column types
num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

# 2. Row-wise aggregate statistics across numerical features
if len(num_cols) > 0:
    num_sub = df[num_cols]
    df["row_mean"] = num_sub.mean(axis=1)
    df["row_std"] = num_sub.std(axis=1)
    df["row_min"] = num_sub.min(axis=1)
    df["row_max"] = num_sub.max(axis=1)
    df["row_sum"] = num_sub.sum(axis=1)
    df["row_median"] = num_sub.median(axis=1)
    df["row_null_count"] = num_sub.isnull().sum(axis=1)

# 3. Frequency Encoding
for col in cat_cols:
    freq = df[col].value_counts().to_dict()
    df[f"{col}_freq"] = df[col].map(freq).fillna(0)

# 4. Skewed feature log transform
for col in num_cols:
    if (df[col].dropna() >= 0).all():
        skewness = df[col].skew()
        if skewness > 1.5:
            df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))

# 5. Expanded Group Aggregations across all numeric columns
for cat_col in cat_cols:
    if 1 < df[cat_col].nunique() < 2000:
        agg = df.groupby(cat_col, observed=False)[num_cols].agg(["mean", "std"])
        agg.columns = [f"{cat_col}_{c}_{stat}" for c, stat in agg.columns]
        df = df.merge(agg, on=cat_col, how="left")

# Replace any inf values resulting from calculations
df = df.replace([np.inf, -np.inf], np.nan)

# Update list of categorical features
cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

# Encode categoricals for numerical algorithms
X = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

X_encoded = X.copy()
X_test_encoded = X_test.copy()

for col in cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded[col] = oe.fit_transform(X_encoded[[col]].astype(str)).astype(int).ravel()
    X_test_encoded[col] = (
        oe.transform(X_test_encoded[[col]].astype(str)).astype(int).ravel()
    )

X_lgb = X_encoded.copy()
X_test_lgb = X_test_encoded.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(n_train)
lgb_preds = np.zeros(len(X_test))

xgb_oof = np.zeros(n_train)
xgb_preds = np.zeros(len(X_test))

cb_oof = np.zeros(n_train)
cb_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    X_trn_lgb, y_trn = X_lgb.iloc[trn_idx], y[trn_idx]
    X_val_lgb, y_val = X_lgb.iloc[val_idx], y[val_idx]

    X_trn_enc = X_encoded.iloc[trn_idx]
    X_val_enc = X_encoded.iloc[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_trn_lgb,
        y_trn,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_oof[val_idx] = model_lgb.predict(X_val_lgb)
    lgb_preds += model_lgb.predict(X_test_lgb) / 5.0

    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model_xgb.fit(X_trn_enc, y_trn, eval_set=[(X_val_enc, y_val)], verbose=False)
    xgb_oof[val_idx] = model_xgb.predict(X_val_enc)
    xgb_preds += model_xgb.predict(X_test_encoded) / 5.0

    # CatBoost
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    model_cb.fit(X_trn_enc, y_trn, eval_set=(X_val_enc, y_val))
    cb_oof[val_idx] = model_cb.predict(X_val_enc)
    cb_preds += model_cb.predict(X_test_encoded) / 5.0

# Post-processing target lower bound clipping
y_min = max(0.0, float(np.min(y)))
lgb_oof = np.clip(lgb_oof, a_min=y_min, a_max=None)
xgb_oof = np.clip(xgb_oof, a_min=y_min, a_max=None)
cb_oof = np.clip(cb_oof, a_min=y_min, a_max=None)

lgb_preds = np.clip(lgb_preds, a_min=y_min, a_max=None)
xgb_preds = np.clip(xgb_preds, a_min=y_min, a_max=None)
cb_preds = np.clip(cb_preds, a_min=y_min, a_max=None)


# Optimize ensemble weights
def loss_func(weights):
    w1, w2, w3 = weights
    pred = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
    pred = np.clip(pred, a_min=y_min, a_max=None)
    return np.sqrt(mean_squared_error(y, pred))


res = minimize(
    loss_func, [0.34, 0.33, 0.33], bounds=[(0, 1), (0, 1), (0, 1)], method="SLSQP"
)
weights = res.x / np.sum(res.x)
w1, w2, w3 = weights

oof_ensemble = np.clip(
    w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof, a_min=y_min, a_max=None
)
oof_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
print(f"Validation RMSE: {oof_rmse:.5f}")

# Generate test predictions
test_preds = np.clip(
    w1 * lgb_preds + w2 * xgb_preds + w3 * cb_preds, a_min=y_min, a_max=None
)

# Save predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "PAYMENTS_BY_ISSUER": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
