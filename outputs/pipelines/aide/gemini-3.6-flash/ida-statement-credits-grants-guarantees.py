import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Original Principal Amount (US$)"
y = train[target_col].values.astype(np.float64)
min_target = max(0.0, float(np.nanmin(y)))

train_features = train.drop(columns=[target_col])
test_features = test.copy()

n_train = len(train_features)
n_test = len(test_features)

# Combine for feature engineering
df_all = pd.concat([train_features, test_features], ignore_index=True)

# 2. Date and Datetime Processing
datetime_cols = []
object_cols = []
numeric_cols = []

for col in df_all.columns:
    if df_all[col].dtype == "object" or df_all[col].dtype == "string":
        sample = df_all[col].dropna().astype(str)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notnull().mean() > 0.6:
                df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
                datetime_cols.append(col)
            else:
                object_cols.append(col)
        else:
            object_cols.append(col)
    elif np.issubdtype(df_all[col].dtype, np.datetime64):
        datetime_cols.append(col)
    else:
        numeric_cols.append(col)

for col in datetime_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear

for i in range(len(datetime_cols)):
    for j in range(i + 1, len(datetime_cols)):
        col1, col2 = datetime_cols[i], datetime_cols[j]
        df_all[f"datediff_{col1}_{col2}"] = (df_all[col1] - df_all[col2]).dt.days

df_all = df_all.drop(columns=datetime_cols)

# 3. Text SVD Features
if len(object_cols) > 0:
    text_data = df_all[object_cols].fillna("").astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    text_tfidf = tfidf.fit_transform(text_data)
    svd_dim = min(16, max(1, text_tfidf.shape[1] - 1))
    if svd_dim > 0:
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        svd_features = svd.fit_transform(text_tfidf)
        for i in range(svd_dim):
            df_all[f"svd_{i}"] = svd_features[:, i]

# 4. Categorical Encodings & Frequency Features
cat_cols = []
for col in object_cols:
    freq = df_all[col].value_counts().to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    cat_cols.append(col)

# Group Aggregations & Relative Differences/Ratios
top_cats = [col for col in cat_cols if df_all[col].nunique() < 500][:3]
num_cols_for_grp = [
    c
    for c in df_all.columns
    if df_all[c].dtype in ["int64", "float64"] and not c.startswith("svd")
][:5]

for cat in top_cats:
    for num in num_cols_for_grp:
        grp_mean = df_all.groupby(cat, observed=False)[num].transform("mean")
        grp_std = df_all.groupby(cat, observed=False)[num].transform("std")
        df_all[f"grp_mean_{cat}_{num}"] = grp_mean
        df_all[f"grp_std_{cat}_{num}"] = grp_std
        df_all[f"grp_diff_{cat}_{num}"] = df_all[num] - grp_mean
        df_all[f"grp_ratio_{cat}_{num}"] = df_all[num] / (grp_mean.abs() + 1.0)

# Convert categoricals to category dtype for LGBM
for col in cat_cols:
    df_all[col] = df_all[col].astype("category")

# Separate train and test
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# Numerical version of features for XGBoost and CatBoost
X_train_num = X_train.copy()
X_test_num = X_test.copy()
for col in cat_cols:
    X_train_num[col] = X_train_num[col].cat.codes
    X_test_num[col] = X_test_num[col].cat.codes

X_train_num = X_train_num.fillna(0)
X_test_num = X_test_num.fillna(0)

# 5. Out-of-fold Cross Validation and Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)

target_enc_cols = [c for c in cat_cols if df_all[c].nunique() < 1000]
m_smooth = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr_lgb, y_tr = X_train.iloc[train_idx].copy(), y[train_idx]
    X_val_lgb, y_val = X_train.iloc[val_idx].copy(), y[val_idx]
    X_te_lgb = X_test.copy()

    X_tr_num = X_train_num.iloc[train_idx].copy()
    X_val_num = X_train_num.iloc[val_idx].copy()
    X_te_num = X_test_num.copy()

    global_mean = float(np.mean(y_tr))

    # Smoothed Out-of-Fold Target Encoding (m-estimate)
    for col in target_enc_cols:
        col_str_tr = X_tr_lgb[col].astype(str)
        col_str_val = X_val_lgb[col].astype(str)
        col_str_te = X_te_lgb[col].astype(str)

        y_tr_series = pd.Series(y_tr, index=X_tr_lgb.index)
        grp = y_tr_series.groupby(col_str_tr, observed=False)
        counts = grp.count()
        means = grp.mean()

        smoothed_te = (counts * means + m_smooth * global_mean) / (counts + m_smooth)
        te_dict = smoothed_te.to_dict()

        te_name = f"te_{col}"
        te_tr = col_str_tr.map(te_dict).fillna(global_mean).astype(float)
        te_val = col_str_val.map(te_dict).fillna(global_mean).astype(float)
        te_te = col_str_te.map(te_dict).fillna(global_mean).astype(float)

        X_tr_lgb[te_name] = te_tr
        X_val_lgb[te_name] = te_val
        X_te_lgb[te_name] = te_te

        X_tr_num[te_name] = te_tr
        X_val_num[te_name] = te_val
        X_te_num[te_name] = te_te

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_lgb)
    test_preds_lgb += lgb_model.predict(X_te_lgb) / kf.n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=6,
        random_state=42 + fold,
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_val_num, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_num)
    test_preds_xgb += xgb_model.predict(X_te_num) / kf.n_splits

    # --- CatBoost ---
    cat_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.025,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_model.fit(X_tr_num, y_tr, eval_set=(X_val_num, y_val), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_val_num)
    test_preds_cat += cat_model.predict(X_te_num) / kf.n_splits


# 6. Ensemble Optimization (Blending)
def rmse_obj(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    pred = np.clip(pred, min_target, None)
    return root_mean_squared_error(y, pred)


res = minimize(
    rmse_obj, [1 / 3, 1 / 3, 1 / 3], method="SLSQP", bounds=[(0, 1), (0, 1), (0, 1)]
)
best_weights = res.x / np.sum(res.x)

oof_ensemble = (
    best_weights[0] * oof_lgb + best_weights[1] * oof_xgb + best_weights[2] * oof_cat
)
oof_ensemble = np.clip(oof_ensemble, min_target, None)
final_cv_rmse = root_mean_squared_error(y, oof_ensemble)

# 7. Generate Test Predictions and Save Submission
test_ensemble = (
    best_weights[0] * test_preds_lgb
    + best_weights[1] * test_preds_xgb
    + best_weights[2] * test_preds_cat
)
test_ensemble = np.clip(test_ensemble, min_target, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_ensemble)),
        "Original Principal Amount (US$)": test_ensemble,
    }
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation Metric: {final_cv_rmse:.4f}")
