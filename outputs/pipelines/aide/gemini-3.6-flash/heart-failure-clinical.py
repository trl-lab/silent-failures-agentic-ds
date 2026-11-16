import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = [c for c in train_df.columns if c.lower() == "death_event"][0]


# Feature Engineering
def add_features(df):
    df = df.copy()
    df["creatinine_sodium_ratio"] = df["serum_creatinine"] / (df["serum_sodium"] + 1e-5)
    df["ef_age_ratio"] = df["ejection_fraction"] / (df["age"] + 1e-5)
    df["time_age_ratio"] = df["time"] / (df["age"] + 1e-5)
    df["risk_factors_count"] = (
        df["anaemia"] + df["diabetes"] + df["high_blood_pressure"] + df["smoking"]
    )
    df["creatinine_age_prod"] = df["serum_creatinine"] * df["age"]
    return df


train_fe = add_features(train_df)
test_fe = add_features(test_df)

drop_cols = [target_col] + [c for c in train_fe.columns if c.lower() in ["id", "index"]]
X = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
y = train_fe[target_col]

test_drop = [c for c in test_fe.columns if c.lower() in ["id", "index"]]
X_test = test_fe.drop(columns=[c for c in test_drop if c in test_fe.columns])

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_names = ["lgb", "xgb", "cb", "lr", "et", "rf"]
num_models = len(model_names)

oof_preds = {m: np.zeros(len(train_df)) for m in model_names}
test_preds = {m: np.zeros(len(test_df)) for m in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds["lgb"][val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_preds["lgb"] += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_preds["xgb"][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_preds["xgb"] += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 3: CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=4,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    oof_preds["cb"][val_idx] = cb_model.predict_proba(X_val)[:, 1]
    test_preds["cb"] += cb_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Scaled inputs for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model 4: Logistic Regression
    lr_model = LogisticRegression(C=0.5, random_state=42 + fold, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    oof_preds["lr"][val_idx] = lr_model.predict_proba(X_val_scaled)[:, 1]
    test_preds["lr"] += lr_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits

    # Model 5: ExtraTrees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=300, max_depth=5, random_state=42 + fold
    )
    et_model.fit(X_train, y_train)
    oof_preds["et"][val_idx] = et_model.predict_proba(X_val)[:, 1]
    test_preds["et"] += et_model.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 6: Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=5, random_state=42 + fold
    )
    rf_model.fit(X_train, y_train)
    oof_preds["rf"][val_idx] = rf_model.predict_proba(X_val)[:, 1]
    test_preds["rf"] += rf_model.predict_proba(X_test)[:, 1] / skf.n_splits

# Convert out-of-fold and test predictions to percentile ranks
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])

oof_ranks = np.zeros_like(oof_matrix)
test_ranks = np.zeros_like(test_matrix)

for col in range(num_models):
    oof_ranks[:, col] = rankdata(oof_matrix[:, col]) / len(oof_matrix)
    test_ranks[:, col] = rankdata(test_matrix[:, col]) / len(test_matrix)


# SLSQP optimization on rank-transformed predictions
def obj_func(weights):
    weights = np.array(weights)
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0
    weights = weights / sum_w
    blend = oof_ranks @ weights
    return -roc_auc_score(y, blend)


init_weights = [1.0 / num_models] * num_models
bounds = [(0, 1)] * num_models
res = minimize(obj_func, init_weights, bounds=bounds, method="SLSQP")
best_weights = res.x / np.sum(res.x)

# Overall OOF prediction and evaluation metric
oof_blend = oof_ranks @ best_weights
cv_auc = roc_auc_score(y, oof_blend)
print(f"5-Fold Cross-Validation ROC AUC: {cv_auc:.5f}")

# Generate test prediction blend and save
test_blend = test_ranks @ best_weights
submission = pd.DataFrame({"id": np.arange(len(test_df)), "DEATH_EVENT": test_blend})
submission.to_csv("./working/submission.csv", index=False)
