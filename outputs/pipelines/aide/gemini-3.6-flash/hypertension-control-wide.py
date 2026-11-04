import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Estimated Percentage of Patients with Controlled Blood Pressure"

# Prepare Target Variable
y_raw = train_df[target_col]
if pd.api.types.is_numeric_dtype(y_raw):
    unique_vals = np.unique(y_raw.dropna())
    if len(unique_vals) == 2:
        y = (y_raw == unique_vals.max()).astype(int).values
    else:
        y = (y_raw >= y_raw.median()).astype(int).values
else:
    unique_vals = y_raw.unique()
    if len(unique_vals) == 2:
        y = (y_raw == unique_vals[1]).astype(int).values
    else:
        y = (y_raw.astype(str) == str(unique_vals[0])).astype(int).values

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

# Drop ID columns if existing
id_cols = [c for c in X_train_raw.columns if c.lower() in ["id", "row_id", "index"]]
if id_cols:
    X_train_raw.drop(columns=id_cols, inplace=True)
    X_test_raw.drop(
        columns=[c for c in id_cols if c in X_test_raw.columns], inplace=True
    )

# 2. Extract Datetime Features
for col in list(X_train_raw.columns):
    if (
        pd.api.types.is_datetime64_any_dtype(X_train_raw[col])
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            d_train = pd.to_datetime(X_train_raw[col])
            d_test = pd.to_datetime(X_test_raw[col])
            X_train_raw[f"{col}_year"] = d_train.dt.year
            X_train_raw[f"{col}_month"] = d_train.dt.month
            X_train_raw[f"{col}_day"] = d_train.dt.day
            X_train_raw[f"{col}_dayofweek"] = d_train.dt.dayofweek
            X_test_raw[f"{col}_year"] = d_test.dt.year
            X_test_raw[f"{col}_month"] = d_test.dt.month
            X_test_raw[f"{col}_day"] = d_test.dt.day
            X_test_raw[f"{col}_dayofweek"] = d_test.dt.dayofweek
            X_train_raw.drop(columns=[col], inplace=True)
            X_test_raw.drop(columns=[col], inplace=True)
        except Exception:
            pass

# 3. Separate Categorical and Numeric Features
cat_cols = [
    c
    for c in X_train_raw.columns
    if X_train_raw[c].dtype == "object" or X_train_raw[c].dtype.name == "category"
]
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

# Frequency Encoding for Categoricals
combined_cat = pd.concat([X_train_raw[cat_cols], X_test_raw[cat_cols]], axis=0)
for c in cat_cols:
    X_train_raw[c] = X_train_raw[c].astype(str).fillna("Missing")
    X_test_raw[c] = X_test_raw[c].astype(str).fillna("Missing")
    freq = combined_cat[c].astype(str).value_counts(normalize=True).to_dict()
    X_train_raw[f"{c}_freq"] = X_train_raw[c].map(freq).astype(float)
    X_test_raw[f"{c}_freq"] = X_test_raw[c].map(freq).astype(float)

# Missing Indicators and Median Imputation for Numerics
for c in num_cols:
    if X_train_raw[c].isnull().sum() > 0:
        X_train_raw[f"{c}_isnull"] = X_train_raw[c].isnull().astype(int)
        X_test_raw[f"{c}_isnull"] = X_test_raw[c].isnull().astype(int)

all_num_cols = [
    c
    for c in X_train_raw.columns
    if X_train_raw[c].dtype in [np.float64, np.float32, np.int64, np.int32]
]
for c in all_num_cols:
    med = X_train_raw[c].median()
    X_train_raw[c] = X_train_raw[c].fillna(med)
    X_test_raw[c] = X_test_raw[c].fillna(med)

# Row Summary Statistics
X_train_raw["row_mean"] = X_train_raw[all_num_cols].mean(axis=1)
X_train_raw["row_std"] = X_train_raw[all_num_cols].std(axis=1)
X_train_raw["row_sum"] = X_train_raw[all_num_cols].sum(axis=1)

X_test_raw["row_mean"] = X_test_raw[all_num_cols].mean(axis=1)
X_test_raw["row_std"] = X_test_raw[all_num_cols].std(axis=1)
X_test_raw["row_sum"] = X_test_raw[all_num_cols].sum(axis=1)

# Categorical Formatting for Gradient Boosted Trees
X_train_lgb = X_train_raw.copy()
X_test_lgb = X_test_raw.copy()
for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

X_train_num = X_train_raw.copy()
X_test_num = X_test_raw.copy()
for c in cat_cols:
    X_train_num[c] = X_train_num[c].astype("category").cat.codes
    X_test_num[c] = X_test_num[c].astype("category").cat.codes

# 4. Stratified 5-Fold Cross Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cb = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_lgb, y)):
    # LightGBM Fold
    X_tr_lgb, y_tr = X_train_lgb.iloc[trn_idx], y[trn_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y[val_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test_lgb)[:, 1] / skf.n_splits

    # XGBoost Fold
    X_tr_num, X_va_num = X_train_num.iloc[trn_idx], X_train_num.iloc[val_idx]
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_num)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test_num)[:, 1] / skf.n_splits

    # CatBoost Fold
    cb_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        eval_metric="AUC",
        early_stopping_rounds=50,
        verbose=0,
    )
    cb_model.fit(X_tr_num, y_tr, eval_set=(X_va_num, y_va))
    oof_cb[val_idx] = cb_model.predict_proba(X_va_num)[:, 1]
    test_cb += cb_model.predict_proba(X_test_num)[:, 1] / skf.n_splits

# 5. Model Blending and Evaluation
oof_blend = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cb
test_blend = 0.4 * test_lgb + 0.3 * test_xgb + 0.3 * test_cb

cv_score = roc_auc_score(y, oof_blend)
print(f"Validation ROC AUC Score: {cv_score:.5f}")

# 6. Save Predictions
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_blend})
sub.to_csv("./working/submission.csv", index=False)
