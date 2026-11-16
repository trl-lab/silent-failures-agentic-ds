import os
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total Enrollment"
features = [c for c in train_df.columns if c != target_col]

# 2. Feature Engineering & Preprocessing
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Detect and parse datetime features
datetime_cols = []
for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        datetime_cols.append(col)
    elif combined[col].dtype == "object":
        try:
            parsed = pd.to_datetime(combined[col], errors="raise")
            combined[col] = parsed
            datetime_cols.append(col)
        except (ValueError, TypeError):
            pass

for col in datetime_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_is_month_start"] = combined[col].dt.is_month_start.astype(int)
    combined[f"{col}_is_month_end"] = combined[col].dt.is_month_end.astype(int)
    # Cyclical representations
    combined[f"{col}_month_sin"] = np.sin(2 * np.pi * combined[col].dt.month / 12.0)
    combined[f"{col}_month_cos"] = np.cos(2 * np.pi * combined[col].dt.month / 12.0)
    combined.drop(columns=[col], inplace=True)

# Identify base categorical and numerical columns
base_cat_cols = []
num_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        base_cat_cols.append(col)
        combined[col] = combined[col].fillna("Missing").astype(str)
    else:
        num_cols.append(col)
        combined[col] = combined[col].fillna(combined[col].median())

# Construct pairwise categorical interaction features
cat_interaction_cols = []
for col1, col2 in itertools.combinations(base_cat_cols[:6], 2):
    inter_name = f"{col1}_x_{col2}"
    combined[inter_name] = (
        combined[col1].astype(str) + "___" + combined[col2].astype(str)
    )
    cat_interaction_cols.append(inter_name)

all_cat_cols = base_cat_cols + cat_interaction_cols

# Frequency encoding for all categorical and interaction features
for col in all_cat_cols:
    freq = combined[col].value_counts(normalize=True)
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

# Compute multi-variable group aggregations
agg_candidates = [c for c in all_cat_cols if 1 < combined[c].nunique() < 20000]
for cat in agg_candidates[:8]:
    for num in num_cols[:4]:
        agg_df = (
            combined.groupby(cat)[num].agg(["mean", "std", "min", "max"]).reset_index()
        )
        agg_df.columns = [
            cat,
            f"{cat}_{num}_mean",
            f"{cat}_{num}_std",
            f"{cat}_{num}_min",
            f"{cat}_{num}_max",
        ]
        combined = combined.merge(agg_df, on=cat, how="left")
        for stat in ["mean", "std", "min", "max"]:
            col_name = f"{cat}_{num}_{stat}"
            combined[col_name] = combined[col_name].fillna(combined[col_name].median())

# Separate back into train and test sets
n_train = len(train_df)
X = combined.iloc[:n_train].copy()
y = train_df[target_col].values
X_test = combined.iloc[n_train:].copy()

# Prepare datasets for CatBoost and LightGBM
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in all_cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

cat_feature_indices = [X.columns.get_loc(c) for c in all_cat_cols]

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))

lgb_oof = np.zeros(len(X))
lgb_test = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # --- CatBoost ---
    train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
    val_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, cat_features=cat_feature_indices)

    cb_model = CatBoostRegressor(
        iterations=1800,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=4.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
    )
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60, verbose=False)
    cb_oof[val_idx] = cb_model.predict(val_pool)
    cb_test += cb_model.predict(test_pool) / kf.n_splits

    # --- LightGBM ---
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    lgb_train_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=lgb_train_data)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 45,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "random_state": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=1800,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_val_pred = lgb_model.predict(X_va_lgb, num_iteration=lgb_model.best_iteration)
    lgb_oof[val_idx] = lgb_val_pred
    lgb_test += (
        lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)
        / kf.n_splits
    )

    cb_fold_rmse = root_mean_squared_error(y_va, cb_oof[val_idx])
    lgb_fold_rmse = root_mean_squared_error(y_va, lgb_oof[val_idx])
    print(
        f"Fold {fold + 1} - CatBoost RMSE: {cb_fold_rmse:.4f}, LightGBM RMSE: {lgb_fold_rmse:.4f}"
    )


# 4. Ensemble Optimization
def objective(weights):
    w_cb, w_lgb = weights
    pred = np.clip(w_cb * cb_oof + w_lgb * lgb_oof, 0, None)
    return root_mean_squared_error(y, pred)


res = minimize(
    objective,
    [0.5, 0.5],
    bounds=[(0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)
w_cb, w_lgb = res.x
print(f"Optimal Ensemble Weights: CatBoost={w_cb:.4f}, LightGBM={w_lgb:.4f}")

oof_preds = np.clip(w_cb * cb_oof + w_lgb * lgb_oof, 0, None)
test_preds = np.clip(w_cb * cb_test + w_lgb * lgb_test, 0, None)

cb_overall_rmse = root_mean_squared_error(y, np.clip(cb_oof, 0, None))
lgb_overall_rmse = root_mean_squared_error(y, np.clip(lgb_oof, 0, None))
cv_rmse = root_mean_squared_error(y, oof_preds)

print(f"CatBoost OOF RMSE: {cb_overall_rmse:.4f}")
print(f"LightGBM OOF RMSE: {lgb_overall_rmse:.4f}")
print(f"Overall Ensemble OOF RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Total Enrollment": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
