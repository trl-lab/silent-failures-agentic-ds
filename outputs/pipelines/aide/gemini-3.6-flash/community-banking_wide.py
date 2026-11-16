import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "LOANTOASSET"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

if "id" in X_train_raw.columns:
    X_train_raw = X_train_raw.drop(columns=["id"])
if "id" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["id"])

n_train = len(train_df)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

# 2. Row-level Summary Statistics
if len(num_cols) > 0:
    df_all["num_nulls"] = df_all[num_cols].isnull().sum(axis=1)
    df_all["num_mean"] = df_all[num_cols].mean(axis=1)
    df_all["num_std"] = df_all[num_cols].std(axis=1)
    df_all["num_min"] = df_all[num_cols].min(axis=1)
    df_all["num_max"] = df_all[num_cols].max(axis=1)
    df_all["num_sum"] = df_all[num_cols].sum(axis=1)

# Categorical Frequency Encoding
for col in cat_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Categorical Peer-Group Numerical Aggregations
for cat in cat_cols:
    if df_all[cat].nunique() > 1 and df_all[cat].nunique() < 500:
        for num in num_cols[:15]:
            grp_mean = df_all.groupby(cat, observed=False)[num].transform("mean")
            df_all[f"{num}_by_{cat}_mean"] = grp_mean
            df_all[f"{num}_by_{cat}_diff"] = df_all[num] - grp_mean

# Split back to Train and Test sets
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# 3. Smoothed Out-of-Fold Target Encoding (m-estimate smoothing)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y.mean())
m_smooth = 10.0

for col in cat_cols:
    train_cat = X_train[col].astype(str)
    test_cat = X_test[col].astype(str)

    te_oof = np.zeros(n_train, dtype=np.float32)
    for train_idx, val_idx in kf.split(X_train):
        tr_cat_s = train_cat.iloc[train_idx]
        tr_y_s = pd.Series(y[train_idx], index=tr_cat_s.index)
        stats = tr_y_s.groupby(tr_cat_s).agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (
            stats["count"] + m_smooth
        )
        te_map = smooth.to_dict()
        te_oof[val_idx] = train_cat.iloc[val_idx].map(te_map).fillna(global_mean).values

    X_train[f"{col}_te"] = te_oof

    full_stats = pd.Series(y).groupby(train_cat).agg(["count", "mean"])
    full_smooth = (
        full_stats["count"] * full_stats["mean"] + m_smooth * global_mean
    ) / (full_stats["count"] + m_smooth)
    X_test[f"{col}_te"] = test_cat.map(full_smooth.to_dict()).fillna(global_mean).values

# Integer Encoding for categorical columns
for col in cat_cols:
    X_train[col] = X_train[col].astype("category").cat.codes
    X_test[col] = X_test[col].astype("category").cat.codes

# Base model predictions storage
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_hgb = np.zeros(n_train)

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_hgb = np.zeros(len(X_test))

# 4. Train Base Gradient Boosting Models across 5 Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / 5.0

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        tree_method="hist",
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_test) / 5.0

    # Model 3: CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_va)
    test_cat += cat_model.predict(X_test) / 5.0

    # Model 4: HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        early_stopping=True,
        n_iter_no_change=50,
    )
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_va)
    test_hgb += hgb_model.predict(X_test) / 5.0

# 5. Stacking via Ridge Meta Learner
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hgb])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_hgb])

meta_model = Ridge(alpha=10.0, random_state=42)
meta_model.fit(oof_matrix, y)

oof_ensemble = meta_model.predict(oof_matrix)
test_predictions = meta_model.predict(test_matrix)

# Compute Evaluation Metric
overall_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Out-of-Fold Stacking Ensemble RMSE: {overall_rmse:.6f}")

# Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), "LOANTOASSET": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
