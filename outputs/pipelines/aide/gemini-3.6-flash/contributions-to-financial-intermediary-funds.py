import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount in USD"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
n_test = len(test_df)
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Preprocessing & Feature Engineering
date_cols = []
cat_cols = []
num_cols = []

for col in df.columns:
    if col.lower().startswith("id") and col != "id":
        num_cols.append(col)
        continue
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        date_cols.append(col)
    else:
        sample = df[col].dropna().astype(str)
        is_date = False
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample.head(50), errors="coerce")
                if parsed.notna().sum() > 30:
                    is_date = True
            except Exception:
                pass
        if is_date:
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

# Date Features
min_date = None
for col in date_cols:
    dt_series = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = dt_series.dt.year
    df[f"{col}_month"] = dt_series.dt.month
    df[f"{col}_day"] = dt_series.dt.day
    df[f"{col}_quarter"] = dt_series.dt.quarter
    df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
    df[f"{col}_dayofyear"] = dt_series.dt.dayofyear

    col_min = dt_series.min()
    if min_date is None or (col_min is not pd.NaT and col_min < min_date):
        min_date = col_min
    if min_date is not pd.NaT:
        df[f"{col}_elapsed"] = (dt_series - min_date).dt.days

    df.drop(columns=[col], inplace=True)

# Update categorical and numeric columns after date processing
cat_cols = [c for c in cat_cols if c in df.columns]
num_cols = [c for c in df.columns if c not in cat_cols]

# Create pairwise categorical interactions for top categoricals
top_cats = cat_cols[:4]
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        inter_name = f"{c1}__inter__{c2}"
        df[inter_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        cat_cols.append(inter_name)

# Fill missing categoricals & Frequency Encoding
for col in cat_cols:
    df[col] = df[col].fillna("Missing").astype(str)
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)

# Categorical Label Encoding for Tree models
cat_encoded_df = df.copy()
for col in cat_cols:
    cat_encoded_df[col] = cat_encoded_df[col].astype("category").cat.codes

# Prepare numerical features
feature_cols = [c for c in cat_encoded_df.columns if c != "id"]

X_all = cat_encoded_df[feature_cols].copy()
for c in X_all.columns:
    X_all[c] = X_all[c].fillna(
        X_all[c].median() if X_all[c].median() == X_all[c].median() else 0
    )

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# 10-Fold Cross-Validation Setup
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Target Encoding Feature Matrix Generation using 10-fold CV
te_features_train = pd.DataFrame(index=range(n_train))
te_features_test = pd.DataFrame(index=range(n_test))

smooth_weight = 10.0
global_mean = np.mean(y)

for col in cat_cols:
    te_train_col = np.zeros(n_train)
    te_test_col = np.zeros(n_test)

    train_cat_vals = df.iloc[:n_train][col].values
    test_cat_vals = df.iloc[n_train:][col].values

    for train_idx, val_idx in kf.split(X_train, y):
        tr_cats, tr_y = train_cat_vals[train_idx], y[train_idx]
        va_cats = train_cat_vals[val_idx]

        cat_stats = (
            pd.DataFrame({"cat": tr_cats, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth = (
            cat_stats["count"] * cat_stats["mean"] + smooth_weight * global_mean
        ) / (cat_stats["count"] + smooth_weight)

        te_train_col[val_idx] = (
            pd.Series(va_cats).map(smooth).fillna(global_mean).values
        )

    full_stats = (
        pd.DataFrame({"cat": train_cat_vals, "y": y})
        .groupby("cat")["y"]
        .agg(["count", "mean"])
    )
    full_smooth = (
        full_stats["count"] * full_stats["mean"] + smooth_weight * global_mean
    ) / (full_stats["count"] + smooth_weight)
    te_test_col = pd.Series(test_cat_vals).map(full_smooth).fillna(global_mean).values

    te_features_train[f"{col}_te"] = te_train_col
    te_features_test[f"{col}_te"] = te_test_col

X_train_ext = pd.concat([X_train, te_features_train], axis=1)
X_test_ext = pd.concat([X_test, te_features_test], axis=1)

# Standardized data for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ext)
X_test_scaled = scaler.transform(X_test_ext)

# 3. Model Training & Cross Validation Across Scale Transforms
oof_preds = {
    "lgb_raw": np.zeros(n_train),
    "xgb_raw": np.zeros(n_train),
    "cat_raw": np.zeros(n_train),
    "lgb_log": np.zeros(n_train),
    "xgb_log": np.zeros(n_train),
    "cat_log": np.zeros(n_train),
    "lgb_sqrt": np.zeros(n_train),
    "xgb_sqrt": np.zeros(n_train),
    "cat_sqrt": np.zeros(n_train),
    "ridge": np.zeros(n_train),
}

test_preds = {
    "lgb_raw": np.zeros(n_test),
    "xgb_raw": np.zeros(n_test),
    "cat_raw": np.zeros(n_test),
    "lgb_log": np.zeros(n_test),
    "xgb_log": np.zeros(n_test),
    "cat_log": np.zeros(n_test),
    "lgb_sqrt": np.zeros(n_test),
    "xgb_sqrt": np.zeros(n_test),
    "cat_sqrt": np.zeros(n_test),
    "ridge": np.zeros(n_test),
}

y_log = np.log1p(np.maximum(0, y))
y_sqrt = np.sqrt(np.maximum(0, y))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_ext, y)):
    X_tr, y_tr = X_train_ext.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_train_ext.iloc[va_idx], y[va_idx]

    X_tr_sc, y_tr_sc = X_train_scaled[tr_idx], y[tr_idx]
    X_va_sc, y_va_sc = X_train_scaled[va_idx], y[va_idx]

    y_tr_log, y_va_log = y_log[tr_idx], y_log[va_idx]
    y_tr_sqrt, y_va_sqrt = y_sqrt[tr_idx], y_sqrt[va_idx]

    # --- RAW TARGET MODELS ---
    # Model 1: LightGBM on Raw Target
    model_lgb_raw = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb_raw.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds["lgb_raw"][va_idx] = model_lgb_raw.predict(X_va)
    test_preds["lgb_raw"] += model_lgb_raw.predict(X_test_ext) / n_splits

    # Model 2: XGBoost on Raw Target
    model_xgb_raw = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42 + fold,
        tree_method="hist",
    )
    model_xgb_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb_raw"][va_idx] = model_xgb_raw.predict(X_va)
    test_preds["xgb_raw"] += model_xgb_raw.predict(X_test_ext) / n_splits

    # Model 3: CatBoost on Raw Target
    model_cat_raw = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42 + fold, verbose=0
    )
    model_cat_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50)
    oof_preds["cat_raw"][va_idx] = model_cat_raw.predict(X_va)
    test_preds["cat_raw"] += model_cat_raw.predict(X_test_ext) / n_splits

    # --- LOG TARGET MODELS ---
    # Model 4: LightGBM on Log Target
    model_lgb_log = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_log_va = model_lgb_log.predict(X_va)
    oof_preds["lgb_log"][va_idx] = np.expm1(np.maximum(0, pred_log_va))
    pred_log_te = model_lgb_log.predict(X_test_ext)
    test_preds["lgb_log"] += np.expm1(np.maximum(0, pred_log_te)) / n_splits

    # Model 5: XGBoost on Log Target
    model_xgb_log = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42 + fold,
        tree_method="hist",
    )
    model_xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    pred_log_va_xgb = model_xgb_log.predict(X_va)
    oof_preds["xgb_log"][va_idx] = np.expm1(np.maximum(0, pred_log_va_xgb))
    pred_log_te_xgb = model_xgb_log.predict(X_test_ext)
    test_preds["xgb_log"] += np.expm1(np.maximum(0, pred_log_te_xgb)) / n_splits

    # Model 6: CatBoost on Log Target
    model_cat_log = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42 + fold, verbose=0
    )
    model_cat_log.fit(
        X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], early_stopping_rounds=50
    )
    pred_log_va_cat = model_cat_log.predict(X_va)
    oof_preds["cat_log"][va_idx] = np.expm1(np.maximum(0, pred_log_va_cat))
    pred_log_te_cat = model_cat_log.predict(X_test_ext)
    test_preds["cat_log"] += np.expm1(np.maximum(0, pred_log_te_cat)) / n_splits

    # --- SQRT TARGET MODELS ---
    # Model 7: LightGBM on Sqrt Target
    model_lgb_sqrt = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb_sqrt.fit(
        X_tr,
        y_tr_sqrt,
        eval_set=[(X_va, y_va_sqrt)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_sqrt_va = model_lgb_sqrt.predict(X_va)
    oof_preds["lgb_sqrt"][va_idx] = np.maximum(0, pred_sqrt_va) ** 2
    pred_sqrt_te = model_lgb_sqrt.predict(X_test_ext)
    test_preds["lgb_sqrt"] += (np.maximum(0, pred_sqrt_te) ** 2) / n_splits

    # Model 8: XGBoost on Sqrt Target
    model_xgb_sqrt = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42 + fold,
        tree_method="hist",
    )
    model_xgb_sqrt.fit(X_tr, y_tr_sqrt, eval_set=[(X_va, y_va_sqrt)], verbose=False)
    pred_sqrt_va_xgb = model_xgb_sqrt.predict(X_va)
    oof_preds["xgb_sqrt"][va_idx] = np.maximum(0, pred_sqrt_va_xgb) ** 2
    pred_sqrt_te_xgb = model_xgb_sqrt.predict(X_test_ext)
    test_preds["xgb_sqrt"] += (np.maximum(0, pred_sqrt_te_xgb) ** 2) / n_splits

    # Model 9: CatBoost on Sqrt Target
    model_cat_sqrt = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6, random_seed=42 + fold, verbose=0
    )
    model_cat_sqrt.fit(
        X_tr, y_tr_sqrt, eval_set=[(X_va, y_va_sqrt)], early_stopping_rounds=50
    )
    pred_sqrt_va_cat = model_cat_sqrt.predict(X_va)
    oof_preds["cat_sqrt"][va_idx] = np.maximum(0, pred_sqrt_va_cat) ** 2
    pred_sqrt_te_cat = model_cat_sqrt.predict(X_test_ext)
    test_preds["cat_sqrt"] += (np.maximum(0, pred_sqrt_te_cat) ** 2) / n_splits

    # Model 10: Ridge Regressor
    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_tr_sc, y_tr_sc)
    oof_preds["ridge"][va_idx] = model_ridge.predict(X_va_sc)
    test_preds["ridge"] += model_ridge.predict(X_test_scaled) / n_splits

# 4. Optimization of Ensemble Weights
model_names = list(oof_preds.keys())
OOF_matrix = np.column_stack([oof_preds[m] for m in model_names])
TEST_matrix = np.column_stack([test_preds[m] for m in model_names])


def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_pred = np.maximum(0, OOF_matrix @ weights)
    return root_mean_squared_error(y, blend_pred)


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1) for _ in range(len(model_names))]
constraints = {"type": "eq", "fun": lambda w: 1 - np.sum(w)}

res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
best_weights = res.x / np.sum(res.x)

oof_final = np.maximum(0, OOF_matrix @ best_weights)
final_cv_rmse = root_mean_squared_error(y, oof_final)

# 5. Generate Submission File
test_final = np.maximum(0, TEST_matrix @ best_weights)

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(n_test), "Amount in USD": test_final})
sub.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSE: {final_cv_rmse:.4f}")
