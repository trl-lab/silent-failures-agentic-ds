import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import warnings

warnings.filterwarnings("ignore")

# 1. Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train["TARGET"].values
X_train = train.drop(columns=["TARGET", "ID"], errors="ignore").copy()
X_test = test.drop(columns=["ID"], errors="ignore").copy()

# 2. Clean data
# Fix sentinel missing values in var3
X_train["var3"] = X_train["var3"].replace(-999999, 2)
X_test["var3"] = X_test["var3"].replace(-999999, 2)

# Drop constant features
constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
X_train = X_train.drop(columns=constant_cols)
X_test = X_test.drop(columns=constant_cols)

# Drop duplicate columns based on train data
dup_cols = X_train.columns[X_train.T.duplicated()].tolist()
X_train = X_train.drop(columns=dup_cols)
X_test = X_test.drop(columns=dup_cols)

# 3. Feature Engineering
for df in [X_train, X_test]:
    # Zero counts prior to adding engineered floating point columns
    df["num_zeros"] = (df == 0).sum(axis=1)
    df["num_non_zeros"] = (df != 0).sum(axis=1)

    # Wealth var38 features
    if "var38" in df.columns:
        df["var38_log"] = np.log1p(df["var38"])
        df["var38_is_median"] = (np.abs(df["var38"] - 117310.97916494) < 1).astype(int)

    # Summary statistics across features
    df["row_mean"] = df.mean(axis=1)
    df["row_std"] = df.std(axis=1)

# Frequency encoding for key discrete features
freq_cols = [c for c in ["var15", "num_var4", "var36"] if c in X_train.columns]
for c in freq_cols:
    freq_map = pd.concat([X_train[c], X_test[c]]).value_counts().to_dict()
    X_train[f"{c}_freq"] = X_train[c].map(freq_map)
    X_test[f"{c}_freq"] = X_test[c].map(freq_map)

# 4. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # CatBoost Model
    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=100,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / skf.n_splits

    # LightGBM Model
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        random_state=42 + fold,
        objective="binary",
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # XGBoost Model
    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        early_stopping_rounds=100,
        eval_metric="auc",
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_xgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits

# 5. Rank Transformation and Averaging
oof_cat_rank = rankdata(oof_cat) / len(oof_cat)
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)

oof_blend = (oof_cat_rank + oof_lgb_rank + oof_xgb_rank) / 3.0

test_cat_rank = rankdata(test_cat) / len(test_cat)
test_lgb_rank = rankdata(test_lgb) / len(test_lgb)
test_xgb_rank = rankdata(test_xgb) / len(test_xgb)

test_blend = (test_cat_rank + test_lgb_rank + test_xgb_rank) / 3.0

cv_score = roc_auc_score(y, oof_blend)
print(f"5-Fold Cross Validation Rank-Averaged Ensemble ROC AUC Score: {cv_score:.5f}")

# 6. Export Predictions
submission = pd.DataFrame({"id": np.arange(len(test_blend)), "TARGET": test_blend})
submission.to_csv("./working/submission.csv", index=False)
