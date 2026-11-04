import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Recovery"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()


# 2. Feature Engineering
def preprocess_features(df_train, df_test):
    combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Identify and parse date columns
    date_cols = []
    for col in combined.columns:
        if combined[col].dtype == "object" or pd.api.types.is_string_dtype(
            combined[col]
        ):
            sample = combined[col].dropna().astype(str).head(100)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_cols.append(col)
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(combined[col]):
            date_cols.append(col)

    parsed_dates = {}
    for col in date_cols:
        parsed_dates[col] = pd.to_datetime(combined[col], errors="coerce")
        combined[f"{col}_year"] = parsed_dates[col].dt.year
        combined[f"{col}_month"] = parsed_dates[col].dt.month
        combined[f"{col}_day"] = parsed_dates[col].dt.day
        combined[f"{col}_dayofweek"] = parsed_dates[col].dt.dayofweek
        combined[f"{col}_dayofyear"] = parsed_dates[col].dt.dayofyear

    # Pairwise date differences in days
    date_names = list(parsed_dates.keys())
    for i in range(len(date_names)):
        for j in range(i + 1, len(date_names)):
            col_a, col_b = date_names[i], date_names[j]
            diff_days = (
                parsed_dates[col_a] - parsed_dates[col_b]
            ).dt.total_seconds() / 86400.0
            combined[f"diff_days_{col_a}_{col_b}"] = diff_days

    # Drop raw date columns
    combined.drop(columns=date_cols, inplace=True, errors="ignore")

    # Categorical columns
    cat_cols = [
        col
        for col in combined.columns
        if combined[col].dtype == "object"
        or pd.api.types.is_string_dtype(combined[col])
    ]

    # Frequency encoding
    for col in cat_cols:
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype("category").cat.codes

    # Split back
    n_train = len(df_train)
    X_train_proc = combined.iloc[:n_train].reset_index(drop=True)
    X_test_proc = combined.iloc[n_train:].reset_index(drop=True)

    return X_train_proc, X_test_proc, cat_cols


X_train, X_test, cat_cols = preprocess_features(X_train_raw, X_test_raw)

# 3. Model Training: Ensemble of XGBoost, LightGBM, and CatBoost
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr = X_train.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    # Out-of-fold smoothed target encoding
    global_mean = y_tr.mean()
    for col in cat_cols:
        col_encoded = f"{col}_te"
        te_map = (
            pd.DataFrame({"cat": X_tr[col], "target": y_tr})
            .groupby("cat")["target"]
            .agg(["mean", "count"])
        )
        smoothing = 10
        smoothed_te = (te_map["count"] * te_map["mean"] + smoothing * global_mean) / (
            te_map["count"] + smoothing
        )

        X_tr[col_encoded] = X_tr[col].map(smoothed_te).fillna(global_mean)
        X_va[col_encoded] = X_va[col].map(smoothed_te).fillna(global_mean)
        X_te[col_encoded] = X_te[col].map(smoothed_te).fillna(global_mean)

    # 3a. XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_te) / n_splits

    # 3b. LightGBM
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }
    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1200,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_te, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # 3c. CatBoost
    model_cat = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
        loss_function="RMSE",
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_te) / n_splits

# 4. Optimize Ensemble Blend Weights
oof_matrix = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_matrix = np.column_stack([test_xgb, test_lgb, test_cat])


def objective(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    pred_clipped = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred_clipped)


initial_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0, 1), (0, 1), (0, 1)]
opt_result = minimize(
    objective,
    initial_weights,
    bounds=bounds,
    method="SLSQP",
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)

optimal_weights = opt_result.x / np.sum(opt_result.x)
print(
    f"Optimal Ensemble Weights: XGB={optimal_weights[0]:.4f}, LGB={optimal_weights[1]:.4f}, CAT={optimal_weights[2]:.4f}"
)

# Final Ensemble Predictions
final_oof = np.clip(np.dot(oof_matrix, optimal_weights), 0, None)
final_test = np.clip(np.dot(test_matrix, optimal_weights), 0, None)

cv_rmse = root_mean_squared_error(y, final_oof)
print(f"Validation Metric (5-Fold CV RMSE): {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(final_test)), "Recovery": final_test})
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
