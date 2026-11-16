import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Heart Disease"
if train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    y = (train_df[target_col] == "Presence").astype(int).values
else:
    y = train_df[target_col].values

test_ids = test_df["id"]
n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(
    drop=True
)

# 2. Feature Engineering
raw_cat_cols = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]
raw_cat_cols = [c for c in raw_cat_cols if c in df.columns]

# Frequency encoding for interaction of top predictive categorical variables
df["cp_thal_vessels"] = (
    df["Chest pain type"].astype(str)
    + "_"
    + df["Thallium"].astype(str)
    + "_"
    + df["Number of vessels fluro"].astype(str)
)
df["cp_thal_vessels_freq"] = df["cp_thal_vessels"].map(
    df["cp_thal_vessels"].value_counts()
)

df["sex_cp"] = df["Sex"].astype(str) + "_" + df["Chest pain type"].astype(str)
df["sex_cp_freq"] = df["sex_cp"].map(df["sex_cp"].value_counts())

feature_cols = [c for c in df.columns if c not in ["id", "cp_thal_vessels", "sex_cp"]]

for col in raw_cat_cols:
    df[col] = df[col].astype("category")

X = df.iloc[:n_train][feature_cols].copy()
X_test = df.iloc[n_train:][feature_cols].copy()

# 3. 5-Fold Stratified Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_cat_cols = raw_cat_cols

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # --- LightGBM ---
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=raw_cat_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=raw_cat_cols
    )

    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- XGBoost ---
    model_xgb = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=1.5,
        enable_categorical=True,
        eval_metric="auc",
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=60,
    )
    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits

    # --- CatBoost ---
    cat_train = X_train.copy()
    cat_val = X_val.copy()
    cat_test = X_test.copy()
    for col in raw_cat_cols:
        cat_train[col] = cat_train[col].astype(str)
        cat_val[col] = cat_val[col].astype(str)
        cat_test[col] = cat_test[col].astype(str)

    model_cat = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=4.0,
        eval_metric="AUC",
        random_seed=42 + fold,
        cat_features=raw_cat_cols,
        verbose=0,
        early_stopping_rounds=60,
    )
    model_cat.fit(
        cat_train,
        y_train,
        eval_set=(cat_val, y_val),
        use_best_model=True,
    )
    oof_cat[val_idx] = model_cat.predict_proba(cat_val)[:, 1]
    test_cat += model_cat.predict_proba(cat_test)[:, 1] / n_splits

print(f"LightGBM OOF ROC-AUC: {roc_auc_score(y, oof_lgb):.6f}")
print(f"XGBoost  OOF ROC-AUC: {roc_auc_score(y, oof_xgb):.6f}")
print(f"CatBoost OOF ROC-AUC: {roc_auc_score(y, oof_cat):.6f}")

# 4. Find Optimal Blending Weights
OOF = np.column_stack([oof_lgb, oof_xgb, oof_cat])
TEST = np.column_stack([test_lgb, test_xgb, test_cat])


def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = np.dot(OOF, w)
    return -roc_auc_score(y, blend)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0)] * 3
res = minimize(
    loss_func,
    init_weights,
    method="Nelder-Mead",
    bounds=bounds,
    options={"maxiter": 500},
)
opt_w = np.clip(res.x, 0, 1)
opt_w = opt_w / np.sum(opt_w)

oof_ensemble = np.dot(OOF, opt_w)
final_cv_auc = roc_auc_score(y, oof_ensemble)
print(f"\nOptimal Ensemble Weights (LGB, XGB, CAT): {opt_w.round(4).tolist()}")
print(f"Final Ensemble CV ROC-AUC: {final_cv_auc:.6f}")

# 5. Output Test Predictions
test_ensemble = np.dot(TEST, opt_w)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "Heart Disease": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
