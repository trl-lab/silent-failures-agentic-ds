import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from catboost import CatBoostRegressor

    has_catboost = True
except ImportError:
    has_catboost = False

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Total interior livable area"

# Filter invalid/missing target rows in train set
train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_raw = train[target_col].values
y_log = np.log1p(np.maximum(0, y_raw))

n_train = len(train)
n_test = len(test)

# Concatenate for uniform feature engineering
X_all = pd.concat(
    [
        train.drop(columns=[target_col], errors="ignore"),
        test.drop(columns=[target_col], errors="ignore"),
    ],
    axis=0,
).reset_index(drop=True)

if "id" in X_all.columns:
    X_all = X_all.drop(columns=["id"])

X_all = X_all.replace([np.inf, -np.inf], np.nan)

# 2. Separate Column Types
cat_cols = []
num_cols = []
text_cols = []

for col in X_all.columns:
    if X_all[col].dtype == "object" or X_all[col].dtype.name == "category":
        non_null = X_all[col].dropna().astype(str)
        mean_len = non_null.str.len().mean() if len(non_null) > 0 else 0
        if mean_len > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# 3. Text Feature Extraction (TF-IDF + TruncatedSVD)
for col in text_cols:
    text_data = X_all[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf_mat = tfidf.fit_transform(text_data)
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd_mat = svd.fit_transform(tfidf_mat)
    for i in range(5):
        X_all[f"{col}_svd_{i}"] = svd_mat[:, i]
    X_all = X_all.drop(columns=[col])

# 4. Multi-level Geographic & Domain Aggregation Statistics
loc_cols = [
    c
    for c in cat_cols
    if any(k in c.lower() for k in ["zip", "city", "county", "state", "neighborhood"])
]
if not loc_cols and cat_cols:
    loc_cols = cat_cols[:3]

num_cols_for_grp = num_cols[:8]

for loc_c in loc_cols:
    for num_c in num_cols_for_grp:
        grp = (
            X_all.groupby(loc_c, observed=False)[num_c]
            .agg(["mean", "median", "std"])
            .reset_index()
        )
        grp.columns = [
            loc_c,
            f"{loc_c}_{num_c}_mean",
            f"{loc_c}_{num_c}_median",
            f"{loc_c}_{num_c}_std",
        ]
        X_all = X_all.merge(grp, on=loc_c, how="left")
        # Relative differences from localized area medians
        X_all[f"{num_c}_diff_{loc_c}_med"] = (
            X_all[num_c] - X_all[f"{loc_c}_{num_c}_median"]
        )

# 5. Frequency Encoding & Categorical Integer Encoding
for col in cat_cols:
    freq = X_all[col].value_counts(normalize=True).to_dict()
    X_all[f"{col}_freq"] = X_all[col].map(freq).fillna(0)
    X_all[col] = pd.Categorical(X_all[col].astype(str)).codes

# 6. Missingness Indicators & Imputation for Numeric Features
for col in list(X_all.columns):
    missing_count = X_all[col].isnull().sum()
    if missing_count > 0:
        if X_all[col].dtype != "category":
            X_all[f"{col}_isna"] = X_all[col].isnull().astype(int)
            med = X_all[col].median()
            X_all[col] = X_all[col].fillna(med if not pd.isna(med) else 0)

# Split dataset back to Train and Test
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 7. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_estimators": 1000,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1000,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

cb_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_log)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_log[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_log[val_idx]

    # Train LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / 5.0

    # Train XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / 5.0

    # Train CatBoost if available
    if has_catboost:
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cb[val_idx] = model_cb.predict(X_va)
        test_preds_cb += model_cb.predict(X_test) / 5.0

# Combine OOF and Test predictions
if has_catboost:
    oof_blend = (oof_lgb + oof_xgb + oof_cb) / 3.0
    test_blend = (test_preds_lgb + test_preds_xgb + test_preds_cb) / 3.0
else:
    oof_blend = 0.5 * oof_lgb + 0.5 * oof_xgb
    test_blend = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

# Compute Evaluation Metric
oof_rmse = root_mean_squared_error(y_log, oof_blend)
print(f"Validation RMSE: {oof_rmse:.4f}")

# Inverse Transform Predictions to Original Scale
test_final = np.expm1(np.maximum(0, test_blend))

# Save Test Predictions to CSV
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_final)), "Total interior livable area": test_final}
)
sub.to_csv("./working/submission.csv", index=False)
