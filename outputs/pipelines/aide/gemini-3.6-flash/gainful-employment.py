import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Debt-to-Earnings Annual Rate"


# Helper function to convert formatted strings to float
def parse_numeric(series):
    if series.dtype == "object" or series.dtype == "string":
        cleaned = (
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        cleaned = cleaned.replace(
            [
                "PrivacySuppressed",
                "Privacy Suppressed",
                "PS",
                "NULL",
                "NaN",
                "nan",
                "N/A",
                "None",
                "",
            ],
            np.nan,
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


# Clean target column and drop missing target rows
train[target_col] = parse_numeric(train[target_col])
train = train.dropna(subset=[target_col]).reset_index(drop=True)

y = train[target_col].values
X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.copy()

# Combine for uniform feature engineering
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

numeric_cols = []
cat_cols = []

for col in df_all.columns:
    parsed = parse_numeric(df_all[col])
    non_null_raw = df_all[col].notna().sum()
    if non_null_raw > 0:
        parsed_ratio = parsed.notna().sum() / non_null_raw
    else:
        parsed_ratio = 0.0

    if parsed.notna().sum() > 0.05 * len(df_all) and (
        df_all[col].dtype != "object" or parsed_ratio > 0.5
    ):
        df_all[col] = parsed
        numeric_cols.append(col)
    else:
        df_all[col] = df_all[col].astype(str).fillna("Missing")
        cat_cols.append(col)

# Initial fast LightGBM fit to select top important numerical features
n_train = len(train)
X_init = df_all.iloc[:n_train].copy()
for c in cat_cols:
    X_init[c] = X_init[c].astype("category")

quick_model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1
)
quick_model.fit(X_init, y)

importances = pd.Series(quick_model.feature_importances_, index=df_all.columns)
top_num_cols = [
    col for col in importances.sort_values(ascending=False).index if col in numeric_cols
][:15]

# Missingness indicators and row summary statistics
df_all["row_null_count"] = df_all.isnull().sum(axis=1)

if len(numeric_cols) > 0:
    df_all["row_numeric_mean"] = df_all[numeric_cols].mean(axis=1)
    df_all["row_numeric_std"] = df_all[numeric_cols].std(axis=1)

# Pairwise ratios and differences among top numerical features
ratio_diff_features = {}
for i in range(len(top_num_cols)):
    for j in range(i + 1, len(top_num_cols)):
        c1, c2 = top_num_cols[i], top_num_cols[j]
        ratio_diff_features[f"{c1}_div_{c2}"] = df_all[c1] / (df_all[c2].abs() + 1e-5)
        ratio_diff_features[f"{c1}_diff_{c2}"] = df_all[c1] - df_all[c2]

# Category-grouped aggregations for top numerical features
group_agg_features = {}
for cat in cat_cols:
    n_unique = df_all[cat].nunique()
    if 2 <= n_unique <= 500:
        for num in top_num_cols[:6]:
            grp = df_all.groupby(cat)[num]
            mean_val = grp.transform("mean")
            std_val = grp.transform("std")
            med_val = grp.transform("median")
            group_agg_features[f"{num}_mean_by_{cat}"] = mean_val
            group_agg_features[f"{num}_std_by_{cat}"] = std_val
            group_agg_features[f"{num}_med_by_{cat}"] = med_val
            group_agg_features[f"{num}_diff_mean_{cat}"] = df_all[num] - mean_val

# Frequency encoding for categorical columns
freq_features = {}
for c in cat_cols:
    freq = df_all[c].value_counts(normalize=True).to_dict()
    freq_features[f"{c}_freq"] = df_all[c].map(freq)

# Combine engineered features
extra_df = pd.DataFrame(
    {**ratio_diff_features, **group_agg_features, **freq_features},
    index=df_all.index,
)
df_all = pd.concat([df_all, extra_df], axis=1)

# Convert categorical columns to category dtype
for c in cat_cols:
    df_all[c] = df_all[c].astype("category")

# Separate train and test sets
X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(n_train)
oof_preds_lgb_deep = np.zeros(n_train)
oof_preds_cat = np.zeros(n_train)
oof_preds_xgb = np.zeros(n_train)

test_preds_lgb = np.zeros(len(test))
test_preds_lgb_deep = np.zeros(len(test))
test_preds_cat = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # 1. Standard LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / kf.n_splits

    # 2. Deeper LightGBM
    model_lgb_deep = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=142 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb_deep.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds_lgb_deep[val_idx] = model_lgb_deep.predict(X_va)
    test_preds_lgb_deep += model_lgb_deep.predict(X_test) / kf.n_splits

    # 3. CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_te_cb[c] = X_te_cb[c].astype(str)

    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        subsample=0.8,
        random_seed=42 + fold,
        verbose=0,
        cat_features=cat_cols,
    )
    model_cat.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)
    oof_preds_cat[val_idx] = model_cat.predict(X_va_cb)
    test_preds_cat += model_cat.predict(X_te_cb) / kf.n_splits

    # 4. XGBoost
    model_xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
    )
    model_xgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_preds_xgb[val_idx] = model_xgb.predict(X_va)
    test_preds_xgb += model_xgb.predict(X_test) / kf.n_splits


# Constrained optimization to blend ensemble predictions
def objective(weights):
    w1, w2, w3, w4 = weights
    blend = (
        w1 * oof_preds_lgb
        + w2 * oof_preds_lgb_deep
        + w3 * oof_preds_cat
        + w4 * oof_preds_xgb
    )
    return root_mean_squared_error(y, blend)


res = minimize(
    objective,
    [0.25, 0.25, 0.25, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w1, w2, w3, w4 = res.x

oof_blend = (
    w1 * oof_preds_lgb
    + w2 * oof_preds_lgb_deep
    + w3 * oof_preds_cat
    + w4 * oof_preds_xgb
)
val_rmse = root_mean_squared_error(y, oof_blend)

print(f"Validation RMSE: {val_rmse:.5f}")

# Output submission
test_blend = (
    w1 * test_preds_lgb
    + w2 * test_preds_lgb_deep
    + w3 * test_preds_cat
    + w4 * test_preds_xgb
)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test)), "Debt-to-Earnings Annual Rate": test_blend}
)
submission.to_csv("./working/submission.csv", index=False)
