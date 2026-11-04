import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

os.makedirs("./working", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = train["TARGET"].values

for df in (train, test):
    if "SK_ID_CURR" in df.columns:
        df.drop(columns=["SK_ID_CURR"], inplace=True)
train.drop(columns=["TARGET"], inplace=True)

# ── Feature types ─────────────────────────────────────────────────────────
cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols]
print(
    f"Features: {len(train.columns)} total | {len(cat_cols)} cat | {len(num_cols)} num"
)

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# ── Stratified split ──────────────────────────────────────────────────────
X_tr, X_va, y_tr, y_va = train_test_split(
    train, y, test_size=0.2, random_state=42, stratify=y
)

# ── LightGBM ──────────────────────────────────────────────────────────────
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "seed": 42,
    "num_threads": -1,
    "feature_pre_filter": False,
}

dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

# ── Validation score ──────────────────────────────────────────────────────
val_preds = model.predict(X_va, num_iteration=model.best_iteration)
auc = roc_auc_score(y_va, val_preds)
print(f"\nBest iteration : {model.best_iteration}")
print(f"Validation ROC AUC: {auc:.6f}")

# ── Test predictions ──────────────────────────────────────────────────────
test_preds = model.predict(test, num_iteration=model.best_iteration)

sub = pd.DataFrame({"id": np.arange(len(test_preds)), "TARGET": test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
