import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

# ── Load data ──────────────────────────────────────────────
train = pd.read_parquet('./input/train.parquet')
test  = pd.read_parquet('./input/test.parquet')

target_col = 'Status'
y_raw = train[target_col].copy()
X     = train.drop(columns=[target_col]).copy()

# Drop id columns
for df in (X, test):
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

# Align test columns to train
test = test.reindex(columns=X.columns, fill_value=np.nan)

# ── Encode target to 0/1 if needed ─────────────────────────
if y_raw.dtype == object or str(y_raw.dtype) == 'category':
    classes = sorted(y_raw.unique().tolist())
    y = (y_raw == classes[-1]).astype(int).values
else:
    y = y_raw.astype(int).values

# ── Column types ───────────────────────────────────────────
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# ── Split FIRST (prevents leakage) ─────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Preprocess (statistics fitted on train only) ───────────
num_medians = X_train[num_cols].median()
for df in (X_train, X_val, test):
    df[num_cols] = df[num_cols].fillna(num_medians)

for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col]   = X_val[col].astype('category')
    test[col]    = test[col].astype('category')

# ── Train with early stopping ──────────────────────────────
dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

params = dict(
    objective='binary', metric='auc',
    learning_rate=0.05, num_leaves=31, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    seed=42, verbose=-1
)

model = lgb.train(
    params, dtrain, num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# ── Validate (official metric: roc_auc) ────────────────────
val_pred = model.predict(X_val)
val_auc  = roc_auc_score(y_val, val_pred)
print(f"Final Validation Score: {val_auc:.6f}")

# ── Predict test & write submission ────────────────────────
test_pred = model.predict(test)
os.makedirs('./submission', exist_ok=True)
pd.DataFrame({'id': np.arange(len(test)), 'Status': test_pred}) \
  .to_csv('./submission/submission.csv', index=False)