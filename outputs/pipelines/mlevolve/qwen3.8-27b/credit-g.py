import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

target_col = 'class'

# ------------------------------------------------------------------
# 2. Encode target to 0/1 (roc_auc is label-symmetric, mapping is safe)
# ------------------------------------------------------------------
y_raw = train[target_col].astype(str)
class_map = {v: i for i, v in enumerate(sorted(y_raw.unique()))}
y = y_raw.map(class_map).astype(int)

# ------------------------------------------------------------------
# 3. Separate features (drop any id-like column)
# ------------------------------------------------------------------
X = train.drop(columns=[target_col]).copy().reset_index(drop=True)
X_test = test.copy().reset_index(drop=True)

id_like = [c for c in X_test.columns if c.lower() in ('id', 'index')]
for c in id_like:
    if c in X.columns:
        X = X.drop(columns=[c])
    if c in X_test.columns:
        X_test = X_test.drop(columns=[c])

# ------------------------------------------------------------------
# 4. Validation split FIRST (before any preprocessing)
# ------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 5. Preprocessing: fit encoders on the TRAIN split ONLY (no leakage)
# ------------------------------------------------------------------
cat_cols = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X_tr.columns if c not in cat_cols]

if cat_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc_train = pd.DataFrame(encoder.fit_transform(X_tr[cat_cols]),
                             columns=cat_cols, index=X_tr.index)
    enc_val = pd.DataFrame(encoder.transform(X_val[cat_cols]),
                           columns=cat_cols, index=X_val.index)
    enc_test = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                            columns=cat_cols, index=X_test.index)
    X_tr_enc = X_tr[num_cols].join(enc_train)
    X_val_enc = X_val[num_cols].join(enc_val)
    X_test_enc = X_test[num_cols].join(enc_test)
else:
    X_tr_enc = X_tr.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()

# ------------------------------------------------------------------
# 6. Train model with early stopping (best checkpoint used)
# ------------------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_tr_enc, y_tr,
    eval_set=[(X_val_enc, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

# ------------------------------------------------------------------
# 7. Validation score using the OFFICIAL metric (roc_auc)
# ------------------------------------------------------------------
val_pred = model.predict_proba(X_val_enc)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Final Validation Score: {val_auc:.5f}")

# ------------------------------------------------------------------
# 8. Test predictions: probability of positive class (for roc_auc)
# ------------------------------------------------------------------
test_pred = model.predict_proba(X_test_enc)[:, 1]

# ------------------------------------------------------------------
# 9. Write submission
# ------------------------------------------------------------------
os.makedirs('./submission', exist_ok=True)
submission = pd.DataFrame({
    'id': np.arange(len(test_pred)),
    'class': test_pred,
})
submission.to_csv('./submission/submission.csv', index=False)
print("Submission saved to ./submission/submission.csv")