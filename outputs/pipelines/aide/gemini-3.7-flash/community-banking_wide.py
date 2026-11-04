import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "LOANTOASSET"
drop_cols = [target_col]
if "id" in train_df.columns:
    drop_cols.append("id")

feature_cols = [c for c in train_df.columns if c not in drop_cols]

# 2. Data Cleaning & Preprocessing
for c in feature_cols:
    if (
        train_df[c].dtype == "object"
        or train_df[c].dtype.name == "category"
        or train_df[c].dtype == "bool"
    ):
        # Check if the column is a string-formatted numeric/percentage
        sample_vals = (
            train_df[c]
            .dropna()
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        converted = pd.to_numeric(sample_vals, errors="coerce")
        if len(sample_vals) > 0 and converted.notna().sum() / len(sample_vals) > 0.8:
            train_df[c] = pd.to_numeric(
                train_df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
            test_df[c] = pd.to_numeric(
                test_df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip(),
                errors="coerce",
            )

# Identify remaining categorical and numerical columns
cat_cols = []
num_cols = []
for c in feature_cols:
    if (
        train_df[c].dtype == "object"
        or train_df[c].dtype.name == "category"
        or train_df[c].dtype == "bool"
    ):
        cat_cols.append(c)
        train_df[c] = train_df[c].fillna("missing").astype(str)
        test_df[c] = test_df[c].fillna("missing").astype(str)
    else:
        num_cols.append(c)

# Prepare datasets
X_cat = train_df[feature_cols].copy()
X_test_cat = test_df[feature_cols].copy()

# Ordinal encode for tree models
X_encoded = train_df[feature_cols].copy()
X_test_encoded = test_df[feature_cols].copy()

if len(cat_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0))
    X_encoded[cat_cols] = encoder.transform(train_df[cat_cols])
    X_test_encoded[cat_cols] = encoder.transform(test_df[cat_cols])

y = train_df[target_col].copy()

# 3. 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(len(train_df))
test_cb = np.zeros(len(test_df))

oof_lgb = np.zeros(len(train_df))
test_lgb = np.zeros(len(test_df))

oof_xgb = np.zeros(len(train_df))
test_xgb = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(
        X_cat.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=(X_cat.iloc[val_idx], y.iloc[val_idx]),
        use_best_model=True,
    )
    oof_cb[val_idx] = cb_model.predict(X_cat.iloc[val_idx])
    test_cb += cb_model.predict(X_test_cat) / n_splits

    # LightGBM
    lgb_train = lgb.Dataset(
        X_encoded.iloc[train_idx],
        label=y.iloc[train_idx],
        categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    )
    lgb_val = lgb.Dataset(
        X_encoded.iloc[val_idx],
        label=y.iloc[val_idx],
        reference=lgb_train,
        categorical_feature=cat_cols if len(cat_cols) > 0 else "auto",
    )

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(
        X_encoded.iloc[val_idx], num_iteration=lgb_model.best_iteration
    )
    test_lgb += (
        lgb_model.predict(X_test_encoded, num_iteration=lgb_model.best_iteration)
        / n_splits
    )

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_encoded.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(X_encoded.iloc[val_idx], y.iloc[val_idx])],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_encoded.iloc[val_idx])
    test_xgb += xgb_model.predict(X_test_encoded) / n_splits

# 4. Optimize Ensemble Blend Weights
oof_matrix = np.column_stack([oof_cb, oof_lgb, oof_xgb])
test_matrix = np.column_stack([test_cb, test_lgb, test_xgb])


def objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    pred = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y, pred)


initial_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
bounds = [(0, 1) for _ in range(3)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
res = minimize(
    objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

best_weights = res.x / np.sum(res.x)
oof_ensemble = np.dot(oof_matrix, best_weights)
test_ensemble = np.dot(test_matrix, best_weights)

cv_rmse = root_mean_squared_error(y, oof_ensemble)
print(
    f"Individual OOF RMSEs - CatBoost: {root_mean_squared_error(y, oof_cb):.6f}, "
    f"LightGBM: {root_mean_squared_error(y, oof_lgb):.6f}, "
    f"XGBoost: {root_mean_squared_error(y, oof_xgb):.6f}"
)
print(f"Optimal Ensemble Weights (CB, LGB, XGB): {best_weights.round(4)}")
print(f"Ensemble 5-Fold CV RMSE: {cv_rmse:.6f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "LOANTOASSET": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
