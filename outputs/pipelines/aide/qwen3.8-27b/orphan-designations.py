import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os, warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "Status"

# ---- identify feature columns ------------------------------------------------
drop = {target} | {
    c
    for c in train.columns
    if c.lower() in ("id", "index", "row_id") or c.startswith("Unnamed")
}
features = [c for c in train.columns if c not in drop]

cat_cols, num_cols = [], []
for col in list(features):
    s = train[col]
    if s.dtype == object or str(s.dtype) == "category":
        cat_cols.append(col)
    elif np.issubdtype(s.dtype, np.datetime64):
        train[col + "_year"] = s.dt.year
        test[col + "_year"] = test[col].dt.year
        features = [f for f in features if f != col] + [col + "_year"]
        num_cols.append(col + "_year")
    elif np.issubdtype(s.dtype, np.number) or s.dtype == bool:
        num_cols.append(col)
    else:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        if train[col].dtype != object:
            num_cols.append(col)
        else:
            cat_cols.append(col)

# ---- label-encode categoricals (consistent across train/test) ---------------
for col in cat_cols:
    tr, te = train[col].astype(str), test[col].astype(str)
    mapping = {v: i for i, v in enumerate(sorted(set(tr) | set(te)))}
    train[col] = tr.map(mapping).astype("int32")
    test[col] = te.map(mapping).astype("int32")

# ---- binarise target ----------------------------------------------------------
train[target] = train[target].astype(str)
vals = sorted(train[target].unique())
pos = next(
    (
        v
        for v in vals
        if v.lower()
        in ("granted", "yes", "true", "1", "approved", "active", "positive")
    ),
    None,
)
if pos is None:
    pos = (
        vals[-1]
        if vals[-1].lower() not in ("0", "false", "no", "not_granted", "rejected")
        else vals[0]
    )
y = (train[target] == pos).astype(int)

print(f"Features ({len(features)}): {features}")
print(f"Categoricals ({len(cat_cols)}): {cat_cols}")
print(f"Numerical    ({len(num_cols)}): {num_cols}")
print(f"Positive class: '{pos}'  |  y distribution: {y.value_counts().to_dict()}")

# ---- stratified validation split ---------------------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(
    train[features], y, test_size=0.2, random_state=42, stratify=y
)

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=5,
    verbose=-1,
    seed=42,
)

model = lgb.train(
    params,
    lgb.Dataset(X_tr, y_tr),
    num_boost_round=2000,
    valid_sets=[lgb.Dataset(X_va, y_va)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

auc = roc_auc_score(y_va, model.predict(X_va))
print(f"\nValidation ROC AUC: {auc:.5f}   (best_iteration={model.best_iteration})")

# ---- retrain on full data & predict test ------------------------------------
model_full = lgb.train(
    params,
    lgb.Dataset(train[features], y),
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

pred = model_full.predict(test[features])

pd.DataFrame({"id": np.arange(len(test)), "Status": pred}).to_csv(
    "./working/submission.csv", index=False
)
print("submission.csv saved to ./working/")
