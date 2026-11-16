import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col and c != "id"]

# Encode target values
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))

X = train_df[feature_cols]
X_test = test_df[feature_cols]

# 10-Fold Stratified Cross-Validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Model architectures
model_names = ["lgb", "xgb", "cat", "et", "rf", "lr", "svc"]
n_models = len(model_names)

oof_preds_matrix = np.zeros((len(train_df), n_models))
test_preds_matrix = np.zeros((len(test_df), n_models))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Standardize features for linear and kernel models without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.025,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Model 3: CatBoost
    cat_model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.025,
        depth=4,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Model 4: ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=400, max_depth=12, random_state=42 + fold, n_jobs=-1
    )
    et_model.fit(X_train, y_train)

    # Model 5: RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=400, max_depth=12, random_state=42 + fold, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Model 6: Logistic Regression
    lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42 + fold)
    lr_model.fit(X_train_scaled, y_train)

    # Model 7: Support Vector Classifier
    svc_model = SVC(C=1.0, kernel="rbf", probability=True, random_state=42 + fold)
    svc_model.fit(X_train_scaled, y_train)

    # Store validation predictions
    oof_preds_matrix[val_idx, 0] = lgb_model.predict_proba(X_val)[:, 1]
    oof_preds_matrix[val_idx, 1] = xgb_model.predict_proba(X_val)[:, 1]
    oof_preds_matrix[val_idx, 2] = cat_model.predict_proba(X_val)[:, 1]
    oof_preds_matrix[val_idx, 3] = et_model.predict_proba(X_val)[:, 1]
    oof_preds_matrix[val_idx, 4] = rf_model.predict_proba(X_val)[:, 1]
    oof_preds_matrix[val_idx, 5] = lr_model.predict_proba(X_val_scaled)[:, 1]
    oof_preds_matrix[val_idx, 6] = svc_model.predict_proba(X_val_scaled)[:, 1]

    # Accumulate test predictions across folds
    test_preds_matrix[:, 0] += lgb_model.predict_proba(X_test)[:, 1] / n_splits
    test_preds_matrix[:, 1] += xgb_model.predict_proba(X_test)[:, 1] / n_splits
    test_preds_matrix[:, 2] += cat_model.predict_proba(X_test)[:, 1] / n_splits
    test_preds_matrix[:, 3] += et_model.predict_proba(X_test)[:, 1] / n_splits
    test_preds_matrix[:, 4] += rf_model.predict_proba(X_test)[:, 1] / n_splits
    test_preds_matrix[:, 5] += lr_model.predict_proba(X_test_scaled)[:, 1] / n_splits
    test_preds_matrix[:, 6] += svc_model.predict_proba(X_test_scaled)[:, 1] / n_splits

# Rank-transform prediction probabilities for scale standardization
oof_ranks = np.zeros_like(oof_preds_matrix)
test_ranks = np.zeros_like(test_preds_matrix)

for m in range(n_models):
    oof_ranks[:, m] = rankdata(oof_preds_matrix[:, m]) / len(train_df)
    test_ranks[:, m] = rankdata(test_preds_matrix[:, m]) / len(test_df)


# Derivative-free Powell optimization to find optimal blend weights maximizing ROC AUC
def loss_func(raw_weights):
    weights = np.exp(raw_weights) / np.sum(np.exp(raw_weights))
    blend_oof = np.dot(oof_ranks, weights)
    return -roc_auc_score(y, blend_oof)


init_raw = np.zeros(n_models)
opt_res = minimize(loss_func, init_raw, method="Powell")
best_weights = np.exp(opt_res.x) / np.sum(np.exp(opt_res.x))

oof_ensemble_preds = np.dot(oof_ranks, best_weights)
final_cv_auc = roc_auc_score(y, oof_ensemble_preds)

print(
    f"Optimal Weights ({', '.join(model_names)}): {[round(w, 4) for w in best_weights]}"
)
print(f"10-Fold Cross-Validation ROC AUC: {final_cv_auc:.5f}")

# Weighted rank average predictions for test set
final_test_preds = np.dot(test_ranks, best_weights)

# Save submission
os.makedirs("./working", exist_ok=True)
sub_id = test_df["id"] if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": sub_id, "Class": final_test_preds})
submission.to_csv("./working/submission.csv", index=False)
