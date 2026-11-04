import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "G3"
y = train_df[target_col].values.astype(np.float64)
train_features = train_df.drop(columns=[target_col], errors="ignore")
test_features = test_df.copy()

feature_cols = [col for col in train_features.columns if col in test_features.columns]


# 2. Domain Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Education interactions
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["total_edu"] = df["Medu"] + df["Fedu"]
        df["max_edu"] = np.maximum(df["Medu"], df["Fedu"])
        df["min_edu"] = np.minimum(df["Medu"], df["Fedu"])
        df["edu_diff"] = df["Medu"] - df["Fedu"]

    # Alcohol consumption interactions
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["total_alc"] = df["Dalc"] + df["Walc"]
        df["alc_ratio"] = df["Walc"] / (df["Dalc"] + 1e-5)

    # Time allocation and balance
    if "studytime" in df.columns and "freetime" in df.columns:
        df["study_to_free"] = df["studytime"] / (df["freetime"] + 1.0)

    if "studytime" in df.columns and "goout" in df.columns:
        df["study_to_goout"] = df["studytime"] / (df["goout"] + 1.0)

    if "absences" in df.columns and "studytime" in df.columns:
        df["absences_per_study"] = df["absences"] / (df["studytime"] + 1.0)

    # Academic risk score combining absences, alcohol, and past failures
    if (
        "failures" in df.columns
        and "absences" in df.columns
        and "Dalc" in df.columns
        and "Walc" in df.columns
    ):
        df["academic_risk"] = (
            df["failures"] * 2.5
            + (df["absences"] > 10).astype(int) * 1.5
            + (df["Dalc"] + df["Walc"]) * 0.5
        )

    # Period grades progression features
    if "G1" in df.columns and "G2" in df.columns:
        df["G_avg"] = (df["G1"] + df["G2"]) / 2.0
        df["G_weighted"] = 0.70 * df["G2"] + 0.30 * df["G1"]
        df["G_diff"] = df["G2"] - df["G1"]
        df["G_ratio"] = (df["G2"] + 1.0) / (df["G1"] + 1.0)
        df["G_drop_sharp"] = (df["G1"] - df["G2"] >= 3).astype(int)
        df["is_G2_zero"] = (df["G2"] == 0).astype(int)
        df["is_G1_zero"] = (df["G1"] == 0).astype(int)
        df["both_passed"] = ((df["G1"] >= 10) & (df["G2"] >= 10)).astype(int)
        df["both_failed"] = ((df["G1"] < 10) & (df["G2"] < 10)).astype(int)
        if "failures" in df.columns:
            df["failures_x_G2"] = df["failures"] * df["G2"]

    return df


X_train_raw = engineer_features(train_features[feature_cols])
X_test_raw = engineer_features(test_features[feature_cols])

cat_cols = [
    col
    for col in X_train_raw.columns
    if X_train_raw[col].dtype == "object" or X_train_raw[col].dtype.name == "category"
]
num_cols = [col for col in X_train_raw.columns if col not in cat_cols]

# Categorical types for LightGBM and CatBoost
X_train_tree = X_train_raw.copy()
X_test_tree = X_test_raw.copy()
for col in cat_cols:
    X_train_tree[col] = X_train_tree[col].astype("category")
    X_test_tree[col] = X_test_tree[col].astype("category")

# One-hot encoded data for XGBoost and Ridge
X_train_encoded = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join="left", axis=1, fill_value=0
)

# 3. Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train_raw))
test_lgb = np.zeros(len(X_test_raw))

oof_cb = np.zeros(len(X_train_raw))
test_cb = np.zeros(len(X_test_raw))

oof_xgb = np.zeros(len(X_train_raw))
test_xgb = np.zeros(len(X_test_raw))

oof_ridge = np.zeros(len(X_train_raw))
test_ridge = np.zeros(len(X_test_raw))

# Model Parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# 4. Training Loop across Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raw, y)):
    y_tr, y_va = y[train_idx], y[val_idx]

    # --- 1. LightGBM ---
    X_tr_lgb, X_va_lgb = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
    trn_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=trn_data)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=callbacks,
    )
    oof_lgb[val_idx] = model_lgb.predict(
        X_va_lgb, num_iteration=model_lgb.best_iteration
    )
    test_lgb += (
        model_lgb.predict(X_test_tree, num_iteration=model_lgb.best_iteration)
        / n_splits
    )

    # --- 2. CatBoost ---
    cb_cat_cols = list(cat_cols)
    X_tr_cb = X_train_raw.iloc[train_idx].copy()
    X_va_cb = X_train_raw.iloc[val_idx].copy()
    X_te_cb = X_test_raw.copy()
    for c in cb_cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
        X_te_cb[c] = X_te_cb[c].astype(str)

    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
    )
    model_cb.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cb_cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = model_cb.predict(X_va_cb)
    test_cb += model_cb.predict(X_te_cb) / n_splits

    # --- 3. XGBoost ---
    X_tr_enc, X_va_enc = (
        X_train_encoded.iloc[train_idx],
        X_train_encoded.iloc[val_idx],
    )
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model_xgb.fit(
        X_tr_enc,
        y_tr,
        eval_set=[(X_va_enc, y_va)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_va_enc)
    test_xgb += model_xgb.predict(X_test_encoded) / n_splits

    # --- 4. Ridge Regression ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_enc)
    X_va_sc = scaler.transform(X_va_enc)
    X_te_sc = scaler.transform(X_test_encoded)

    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = model_ridge.predict(X_va_sc)
    test_ridge += model_ridge.predict(X_te_sc) / n_splits

# 5. Optimize Ensemble Weights
oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb, oof_ridge])
test_matrix = np.column_stack([test_lgb, test_cb, test_xgb, test_ridge])


def rmse_objective(weights):
    weights = np.array(weights)
    pred = np.dot(oof_matrix, weights)
    return np.sqrt(mean_squared_error(y, pred))


init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    rmse_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x if opt_res.success else np.array([0.35, 0.35, 0.20, 0.10])

oof_ensemble = np.dot(oof_matrix, best_weights)
oof_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))

for i, name in enumerate(["LightGBM", "CatBoost", "XGBoost", "Ridge"]):
    model_rmse = np.sqrt(mean_squared_error(y, oof_matrix[:, i]))
    print(f"{name} CV RMSE: {model_rmse:.4f}")

print(f"Optimal Ensemble Weights (LGB, CB, XGB, Ridge): {best_weights.round(4)}")
print(f"Validation 5-Fold Ensemble RMSE: {oof_rmse:.4f}")

# 6. Generate and Save Submission
test_predictions = np.dot(test_matrix, best_weights)
test_predictions = np.clip(test_predictions, 0.0, 20.0)

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_predictions)), "G3": test_predictions})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
