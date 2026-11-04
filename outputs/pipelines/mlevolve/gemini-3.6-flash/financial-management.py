import json
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Setup Directories & Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "State Share"
id_col = (
    "id" if "id" in train_df.columns else ("id" if "id" in test_df.columns else None)
)

if id_col is None:
    train_df["id"] = train_df.index
    test_df["id"] = test_df.index
    id_col = "id"

train_df[id_col] = train_df[id_col].astype(int)
test_df[id_col] = test_df[id_col].astype(int)

y_train = train_df[target_col].values.astype(np.float32)
feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

# 2. Identify Column Types & Extract Datetime Features
num_cols = []
cat_cols = []
date_cols = []

for col in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        sample_str = train_df[col].dropna().astype(str).head(10)
        try:
            pd.to_datetime(sample_str)
            date_cols.append(col)
        except Exception:
            cat_cols.append(col)

for col in date_cols:
    for df in [train_df, test_df]:
        dt_s = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt_s.dt.year.fillna(-1)
        df[f"{col}_quarter"] = dt_s.dt.quarter.fillna(-1)
        df[f"{col}_month"] = dt_s.dt.month.fillna(-1)
        df[f"{col}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1)
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt_s.dt.month / 12.0).fillna(0)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt_s.dt.month / 12.0).fillna(0)
    train_df.drop(columns=[col], inplace=True)
    test_df.drop(columns=[col], inplace=True)

feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

# 3. Clean Missing Values (Train-only statistics)
for col in cat_cols:
    train_df[col] = train_df[col].astype(str).fillna("Missing")
    test_df[col] = test_df[col].astype(str).fillna("Missing")

for col in num_cols:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col] = test_df[col].fillna(med)

# 4. Feature Engineering: Financial Interactions & Ratios
if len(num_cols) >= 2:
    top_num = num_cols[: min(6, len(num_cols))]
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            train_df[f"ratio_{c1}_{c2}"] = train_df[c1] / (train_df[c2].abs() + 1e-5)
            test_df[f"ratio_{c1}_{c2}"] = test_df[c1] / (test_df[c2].abs() + 1e-5)
            train_df[f"diff_{c1}_{c2}"] = train_df[c1] - train_df[c2]
            test_df[f"diff_{c1}_{c2}"] = test_df[c1] - test_df[c2]

# 5. Log Transformations for Skewed Features
updated_num = [
    c
    for c in train_df.columns
    if c not in [target_col, id_col] and pd.api.types.is_numeric_dtype(train_df[c])
]
for col in updated_num:
    skew_val = train_df[col].skew()
    if abs(skew_val) > 1.5:
        min_val = train_df[col].min()
        train_df[f"log1p_{col}"] = np.log1p(np.maximum(0, train_df[col] - min_val))
        test_df[f"log1p_{col}"] = np.log1p(np.maximum(0, test_df[col] - min_val))

# 6. Frequency Encoding
for col in cat_cols:
    freq = train_df[col].value_counts(normalize=True).to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)

# 7. Out-Of-Fold Group Target Encoding (Zero-Leakage 5-Fold Scheme)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

global_mean = float(y_train.mean())
smoothing = 10.0

for col in cat_cols:
    target_enc_col = f"{col}_target_enc"
    train_df[target_enc_col] = global_mean
    test_df[target_enc_col] = global_mean

    for fold in range(5):
        val_mask = train_df["fold"] == fold
        tr_mask = ~val_mask
        tr_fold = train_df[tr_mask]

        stats = tr_fold.groupby(col)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        val_cats = train_df.loc[val_mask, col]
        train_df.loc[val_mask, target_enc_col] = val_cats.map(smooth).fillna(
            global_mean
        )

    full_stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
    full_smooth = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    test_df[target_enc_col] = test_df[col].map(full_smooth).fillna(global_mean)

# 7b. Categorical Group Aggregations for Continuous Features
group_num_cols = num_cols[: min(6, len(num_cols))]
for c_col in cat_cols:
    for n_col in group_num_cols:
        for stat_name in ["mean", "std", "min", "max"]:
            feat_name = f"grp_{c_col}_{n_col}_{stat_name}"
            train_df[feat_name] = 0.0
            test_df[feat_name] = 0.0

            for fold in range(5):
                val_mask = train_df["fold"] == fold
                tr_mask = ~val_mask
                tr_fold = train_df[tr_mask]

                stat_df = tr_fold.groupby(c_col)[n_col].agg(stat_name)
                val_cats = train_df.loc[val_mask, c_col]
                train_df.loc[val_mask, feat_name] = val_cats.map(stat_df).fillna(0.0)

            full_stat_df = train_df.groupby(c_col)[n_col].agg(stat_name)
            test_df[feat_name] = test_df[c_col].map(full_stat_df).fillna(0.0)

# Convert categorical features to numeric codes for array compatibility
for col in cat_cols:
    cats = sorted(list(set(train_df[col].unique()).union(set(test_df[col].unique()))))
    cat_type = pd.CategoricalDtype(categories=cats, ordered=False)
    train_df[col] = train_df[col].astype(cat_type).cat.codes
    test_df[col] = test_df[col].astype(cat_type).cat.codes

processed_features = [
    c for c in train_df.columns if c not in [target_col, id_col, "fold"]
]


# 8. Define Model Architectures
def get_lgb_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
    )


def get_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=0,
        thread_count=-1,
    )


def get_ridge_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=random_state)),
        ]
    )


# 9. Cross-Validation & Model Training
X_all = np.nan_to_num(
    train_df[processed_features].to_numpy(dtype=np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_all = train_df[target_col].to_numpy(dtype=np.float32)
X_test = np.nan_to_num(
    test_df[processed_features].to_numpy(dtype=np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
test_ids = test_df[id_col].values
folds = train_df["fold"].values
n_splits = len(np.unique(folds))

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_pred_lgb = np.zeros(len(test_df))
test_pred_xgb = np.zeros(len(test_df))
test_pred_cat = np.zeros(len(test_df))
test_pred_et = np.zeros(len(test_df))

for fold in range(n_splits):
    val_mask = folds == fold
    train_mask = ~val_mask

    X_train, y_train_fold = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    val_idx = np.where(val_mask)[0]

    # --- LightGBM ---
    lgb_model = get_lgb_model(random_state=42 + fold)
    lgb_model.fit(
        X_train,
        y_train_fold,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_pred_lgb += lgb_model.predict(X_test) / n_splits

    # --- XGBoost ---
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.set_params(early_stopping_rounds=50)
    xgb_model.fit(
        X_train,
        y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_pred_xgb += xgb_model.predict(X_test) / n_splits

    # --- CatBoost ---
    cat_model = get_catboost_model(random_state=42 + fold)
    cat_model.fit(
        X_train,
        y_train_fold,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_pred_cat += cat_model.predict(X_test) / n_splits

    # --- Ridge ---
    et_model = get_ridge_model(random_state=42 + fold)
    et_model.fit(X_train, y_train_fold)
    oof_et[val_idx] = et_model.predict(X_val)
    test_pred_et += et_model.predict(X_test) / n_splits

    rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    rmse_et = np.sqrt(mean_squared_error(y_val, oof_et[val_idx]))
    print(
        f"Fold {fold+1}/{n_splits} - LGB: {rmse_lgb:.4f} | XGB: {rmse_xgb:.4f} | CAT: {rmse_cat:.4f} | ET: {rmse_et:.4f}"
    )


# 10. Optimize Ensemble Weights & Generate Submission
def ensemble_loss(weights, oof_preds, y_true):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = sum(w * pred for w, pred in zip(weights, oof_preds))
    return np.sqrt(mean_squared_error(y_true, blend))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}

res = minimize(
    ensemble_loss,
    init_weights,
    args=([oof_lgb, oof_xgb, oof_cat, oof_et], y_all),
    bounds=bounds,
    constraints=constraints,
    method="SLSQP",
)

best_weights = res.x / np.sum(res.x)
oof_final = (
    best_weights[0] * oof_lgb
    + best_weights[1] * oof_xgb
    + best_weights[2] * oof_cat
    + best_weights[3] * oof_et
)
final_rmse = np.sqrt(mean_squared_error(y_all, oof_final))

test_final = (
    best_weights[0] * test_pred_lgb
    + best_weights[1] * test_pred_xgb
    + best_weights[2] * test_pred_cat
    + best_weights[3] * test_pred_et
)

submission_path = os.path.join(submission_dir, "submission.csv")
sub_df = pd.DataFrame({"id": test_ids, "State Share": test_final})
sub_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_rmse:.6f}")