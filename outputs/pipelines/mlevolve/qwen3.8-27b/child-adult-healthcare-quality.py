import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_parquet('./input/train.parquet')
test  = pd.read_parquet('./input/test.parquet')

target = 'State Rate'
y = train[target].values

# Feature columns: everything in train except the target
feature_cols = [c for c in train.columns if c != target]
# Keep only columns present in both train and test
feature_cols = [c for c in feature_cols if c in test.columns]

X      = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Cast object/category columns so LightGBM can treat them as categoricals
cat_cols = []
for col in feature_cols:
    if X[col].dtype == object or str(X[col].dtype) == 'category':
        X[col]      = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        cat_cols.append(col)

# ── Validation split (fit BEFORE any transform on full data) ───────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train with early stopping ──────────────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    categorical_feature=cat_cols if cat_cols else 'auto',
    callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=0)],
)

# ── Validation score (official metric: RMSE) ───────────────────────────────
val_preds = model.predict(X_val)
val_rmse  = float(np.sqrt(mean_squared_error(y_val, val_preds)))
print(f"Final Validation Score (RMSE): {val_rmse:.6f}")

# ── Retrain on FULL training data using best iteration count ───────────────
best_it = getattr(model, 'best_iteration_', None)
if best_it is None or best_it <= 0:
    best_it = 3000

final_model = lgb.LGBMRegressor(
    n_estimators=int(best_it),
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
final_model.fit(X, y, categorical_feature=cat_cols if cat_cols else 'auto')

# ── Test predictions ───────────────────────────────────────────────────────
test_preds = final_model.predict(X_test)

# ── Write submission ───────────────────────────────────────────────────────
submission = pd.DataFrame({
    'id':         np.arange(len(test), dtype=int),
    'State Rate': test_preds,
})

os.makedirs('./submission', exist_ok=True)
submission.to_csv('./submission/submission.csv', index=False)
print(f"Submission written: {len(submission)} rows → ./submission/submission.csv")