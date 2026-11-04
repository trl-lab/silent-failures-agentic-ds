import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Average or Greater than 30"
if target_col not in train_df.columns:
    # Look for possible target column variations if exact match fails
    for col in train_df.columns:
        if "30" in col or "Average" in col or "target" in col.lower():
            target_col = col
            break

y = train_df[target_col].astype(int).values
train_df = train_df.drop(columns=[target_col])

# Combine for consistent feature engineering
n_train = len(train_df)
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Drop redundant id columns if present
drop_cols = [
    col for col in combined.columns if col.lower() in ["id", "index", "unnamed: 0"]
]
if drop_cols:
    combined = combined.drop(columns=drop_cols)

# Identify numerical and categorical columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
    elif combined[col].nunique() < 15 and not np.issubdtype(
        combined[col].dtype, np.floating
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# 2. Feature Engineering
# Missing value indicators for high-sparsity columns
for col in num_cols:
    if combined[col].isnull().sum() > 0:
        combined[f"{col}_isnan"] = combined[col].isnull().astype(int)

# Frequency encoding for categoricals
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

# Statistical aggregations across top numerical features
if len(num_cols) >= 2:
    for i in range(min(5, len(num_cols))):
        for j in range(i + 1, min(5, len(num_cols))):
            c1, c2 = num_cols[i], num_cols[j]
            combined[f"{c1}_add_{c2}"] = combined[c1] + combined[c2]
            combined[f"{c1}_sub_{c2}"] = combined[c1] - combined[c2]
            combined[f"{c1}_mul_{c2}"] = combined[c1] * combined[c2]
            combined[f"{c1}_div_{c2}"] = (
                combined[c1] / (combined[c2].abs() + 1e-5)
            ).replace([np.inf, -np.inf], np.nan)

# Label encoding for categoricals
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna("__MISSING__")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Split back to train and test
X = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

# 3. Stratified 5-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx].copy(), y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx].copy(), y[val_idx]

    # Out-of-fold target encoding for high-cardinality categorical features
    for c in cat_cols:
        if c in X.columns:
            te_map = (
                y_train_fold.groupby(X_train_fold[c]).mean()
                if hasattr(y_train_fold, "groupby")
                else pd.Series(y_train_fold).groupby(X_train_fold[c]).mean()
            )
            global_mean = np.mean(y_train_fold)
            smooth = 10
            counts = X_train_fold[c].map(X_train_fold[c].value_counts()).fillna(0)
            smoothed_te = (te_map * counts + global_mean * smooth) / (counts + smooth)

            X_train_fold[f"{c}_te"] = (
                X_train_fold[c].map(smoothed_te).fillna(global_mean)
            )
            X_val_fold[f"{c}_te"] = X_val_fold[c].map(smoothed_te).fillna(global_mean)

    # Matching columns for test set on fold
    X_test_fold = X_test.copy()
    for c in cat_cols:
        if c in X.columns:
            te_map = pd.Series(y_train_fold).groupby(X_train_fold[c]).mean()
            global_mean = np.mean(y_train_fold)
            smooth = 10
            counts = X_train_fold[c].map(X_train_fold[c].value_counts()).fillna(0)
            smoothed_te = (te_map * counts + global_mean * smooth) / (counts + smooth)
            X_test_fold[f"{c}_te"] = X_test_fold[c].map(smoothed_te).fillna(global_mean)

    # Model 1: LightGBM
    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "random_state": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }

    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(
        X_val_fold, num_iteration=model_lgb.best_iteration
    )
    test_preds_lgb += (
        model_lgb.predict(X_test_fold, num_iteration=model_lgb.best_iteration)
        / n_splits
    )

    # Model 2: XGBoost
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "n_estimators": 1500,
        "early_stopping_rounds": 50,
    }

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_fold)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test_fold)[:, 1] / n_splits

    # Model 3: CatBoost
    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cat.fit(
        X_train_fold,
        y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        use_best_model=True,
    )
    oof_cat[val_idx] = model_cat.predict_proba(X_val_fold)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test_fold)[:, 1] / n_splits

# 4. Out-of-fold Evaluation & Blending
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat)

print(f"LGBM OOF ROC-AUC: {auc_lgb:.6f}")
print(f"XGBoost OOF ROC-AUC: {auc_xgb:.6f}")
print(f"CatBoost OOF ROC-AUC: {auc_cat:.6f}")

# Weighted blend optimization via simple grid
best_weights = (1 / 3, 1 / 3, 1 / 3)
best_auc = 0.0
for w1 in np.linspace(0, 1, 11):
    for w2 in np.linspace(0, 1 - w1, 11):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        blend_oof = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = roc_auc_score(y, blend_oof)
        if score > best_auc:
            best_auc = score
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
oof_ensemble = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
final_test_preds = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cat
final_auc = roc_auc_score(y, oof_ensemble)

print(f"Optimal Blend Weights (LGB, XGB, CAT): {w1:.2f}, {w2:.2f}, {w3:.2f}")
print(f"Final Ensemble Out-Of-Fold ROC-AUC: {final_auc:.6f}")

# 5. Generate Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds_lgb)),
        "Average or Greater than 30": final_test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
