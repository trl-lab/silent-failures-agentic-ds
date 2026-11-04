import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "State Share"
y = train_df[target_col].copy().values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

n_train = len(X)
n_test = len(X_test)
combined = pd.concat([X, X_test], axis=0).reset_index(drop=True)

# Identify column types
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()

# 2. Categorical Frequency & Ordinal Encoding
for col in cat_cols:
    combined[col] = combined[col].astype(str)
    freq_map = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).astype(float)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined[col] = encoder.fit_transform(combined[[col]]).astype(int)

# 3. Numerical Pairwise Arithmetic & Group Aggregations
eps = 1e-6
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        col_a, col_b = num_cols[i], num_cols[j]
        combined[f"{col_a}_minus_{col_b}"] = combined[col_a] - combined[col_b]
        combined[f"{col_a}_plus_{col_b}"] = combined[col_a] + combined[col_b]
        combined[f"{col_a}_ratio_{col_b}"] = combined[col_a] / (
            combined[col_b].abs() + eps
        )

if len(cat_cols) > 0 and len(num_cols) > 0:
    for cat in cat_cols[:4]:
        for num in num_cols[:4]:
            grp_mean = combined.groupby(cat)[num].transform("mean")
            grp_std = combined.groupby(cat)[num].transform("std").fillna(0)
            combined[f"{num}_mean_by_{cat}"] = grp_mean
            combined[f"{num}_std_by_{cat}"] = grp_std
            combined[f"{num}_diff_mean_by_{cat}"] = combined[num] - grp_mean

# Split back into train and test
X_processed = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_processed = combined.iloc[n_train:].copy().reset_index(drop=True)

# 4. 5-Fold Cross Validation Setup with 4-Model Ensemble
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(n_train)
cb_oof = np.zeros(n_train)
xgb_oof = np.zeros(n_train)
ridge_oof = np.zeros(n_train)

lgb_test_preds = np.zeros(n_test)
cb_test_preds = np.zeros(n_test)
xgb_test_preds = np.zeros(n_test)
ridge_test_preds = np.zeros(n_test)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed, y)):
    X_train_fold = X_processed.iloc[train_idx].copy()
    y_train_fold = y[train_idx]
    X_val_fold = X_processed.iloc[val_idx].copy()
    y_val_fold = y[val_idx]
    X_test_fold = X_test_processed.copy()

    # Out-of-Fold Target Encoding for categorical features
    smoothing = 10.0
    global_mean = float(np.mean(y_train_fold))
    for cat in cat_cols:
        cat_stats = (
            pd.DataFrame({"cat": X_train_fold[cat], "target": y_train_fold})
            .groupby("cat")["target"]
            .agg(["count", "mean", "std"])
        )
        cat_stats["std"] = cat_stats["std"].fillna(0)
        smoothed_mean = (
            cat_stats["count"] * cat_stats["mean"] + smoothing * global_mean
        ) / (cat_stats["count"] + smoothing)

        mean_map = smoothed_mean.to_dict()
        std_map = cat_stats["std"].to_dict()

        te_mean_col = f"{cat}_te_mean"
        te_std_col = f"{cat}_te_std"

        X_train_fold[te_mean_col] = X_train_fold[cat].map(mean_map).fillna(global_mean)
        X_train_fold[te_std_col] = X_train_fold[cat].map(std_map).fillna(0)

        X_val_fold[te_mean_col] = X_val_fold[cat].map(mean_map).fillna(global_mean)
        X_val_fold[te_std_col] = X_val_fold[cat].map(std_map).fillna(0)

        X_test_fold[te_mean_col] = X_test_fold[cat].map(mean_map).fillna(global_mean)
        X_test_fold[te_std_col] = X_test_fold[cat].map(std_map).fillna(0)

    # 1. LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_oof[val_idx] = lgb_model.predict(
        X_val_fold, num_iteration=lgb_model.best_iteration
    )
    lgb_test_preds += (
        lgb_model.predict(X_test_fold, num_iteration=lgb_model.best_iteration)
        / n_splits
    )

    # 2. CatBoost
    cb_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        use_best_model=True,
    )
    cb_oof[val_idx] = cb_model.predict(X_val_fold)
    cb_test_preds += cb_model.predict(X_test_fold) / n_splits

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    xgb_oof[val_idx] = xgb_model.predict(X_val_fold)
    xgb_test_preds += xgb_model.predict(X_test_fold) / n_splits

    # 4. Regularized Linear Model (Ridge)
    ridge_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0, random_state=42 + fold),
    )
    ridge_pipeline.fit(X_train_fold, y_train_fold)
    ridge_oof[val_idx] = ridge_pipeline.predict(X_val_fold)
    ridge_test_preds += ridge_pipeline.predict(X_test_fold) / n_splits

# 5. Optimize Blending Weights Across 4 Diverse Models
oof_matrix = np.column_stack([lgb_oof, cb_oof, xgb_oof, ridge_oof])
test_matrix = np.column_stack(
    [lgb_test_preds, cb_test_preds, xgb_test_preds, ridge_test_preds]
)


def rmse_objective(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y, pred))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1) for _ in range(4)]
opt_result = minimize(
    rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: 1 - sum(w)},
)

optimal_weights = opt_result.x / np.sum(opt_result.x)
ensemble_oof = np.dot(oof_matrix, optimal_weights)
ensemble_test_preds = np.dot(test_matrix, optimal_weights)

cv_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"Optimal Ensemble Weights: {optimal_weights.round(4)}")
print(f"5-Fold CV RMSE (Ensemble): {cv_rmse:.6f}")

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(ensemble_test_preds)), "State Share": ensemble_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
