import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "decision_code"

# Encode string target to binary integers (0 and 1)
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# 2. Feature Engineering
# Identify date columns
date_cols = []
for col in combined.columns:
    if "date" in col.lower() or combined[col].dtype == "datetime64[ns]":
        date_cols.append(col)
    elif combined[col].dtype == "object":
        sample = combined[col].dropna().astype(str).head(100)
        if len(sample) > 0 and sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.7:
            date_cols.append(col)

# Extract date features
for col in date_cols:
    dt_series = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt_series.dt.year
    combined[f"{col}_month"] = dt_series.dt.month
    combined[f"{col}_day"] = dt_series.dt.day
    combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek
    combined[f"{col}_is_na"] = dt_series.isna().astype(int)

# Date differences
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        d1 = pd.to_datetime(combined[date_cols[i]], errors="coerce")
        d2 = pd.to_datetime(combined[date_cols[j]], errors="coerce")
        combined[f"diff_days_{date_cols[i]}_{date_cols[j]}"] = (d2 - d1).dt.days

# Drop raw date columns
combined = combined.drop(columns=date_cols)

# Process categorical and numerical features
cat_cols = []
num_cols = []

for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        combined[col] = combined[col].fillna("__MISSING__").astype(str)
        # Length feature
        combined[f"{col}_len"] = combined[col].str.len()
        # Frequency encoding
        freq = combined[col].value_counts()
        combined[f"{col}_freq"] = combined[col].map(freq)
        cat_cols.append(col)
    else:
        num_cols.append(col)
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].median())

# Prepare CatBoost features (using categorical features as string objects)
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

# 3. Stratified 5-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_cb = np.zeros(n_train)
test_preds_cb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_va, y_val, cat_features=cat_features_idx)
    test_pool = Pool(X_test, cat_features=cat_features_idx)

    model_cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=100,
        task_type="CPU",
    )

    model_cb.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_preds = model_cb.predict_proba(val_pool)[:, 1]
    oof_preds_cb[val_idx] = val_preds
    test_preds_cb += model_cb.predict_proba(test_pool)[:, 1] / kf.n_splits

    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"CatBoost Fold {fold + 1} ROC-AUC: {fold_auc:.6f}")

# Also build LightGBM model with label encoded categories
lgb_combined = combined.copy()
for col in cat_cols:
    lgb_combined[col] = lgb_combined[col].astype("category").cat.codes

lgb_X_train = lgb_combined.iloc[:n_train].copy()
lgb_X_test = lgb_combined.iloc[n_train:].copy()

oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(lgb_X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(lgb_X_train, y)):
    X_tr, y_tr = lgb_X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = lgb_X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=trn_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "seed": 42 + fold,
        "verbose": -1,
    }

    model_lgb = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_preds = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    oof_preds_lgb[val_idx] = val_preds
    test_preds_lgb += (
        model_lgb.predict(lgb_X_test, num_iteration=model_lgb.best_iteration)
        / kf.n_splits
    )

# Ensemble predictions (blend CatBoost + LightGBM)
oof_ensemble = 0.5 * oof_preds_cb + 0.5 * oof_preds_lgb
test_ensemble = 0.5 * test_preds_cb + 0.5 * test_preds_lgb

overall_auc = roc_auc_score(y, oof_ensemble)
print(f"Overall OOF ROC-AUC: {overall_auc:.6f}")

# 4. Save test predictions to submission.csv
submission = pd.DataFrame(
    {"id": np.arange(len(test_ensemble)), "decision_code": test_ensemble}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
