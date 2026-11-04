import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Load train and test data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify feature columns
cat_cols = [c for c in train.columns if c.startswith("cat")]
num_cols = [c for c in train.columns if c.startswith("cont")]

# Ensure categorical variables are clean string types
for col in cat_cols:
    train[col] = train[col].fillna("missing").astype(str)
    test[col] = test[col].fillna("missing").astype(str)

# Apply logarithmic transformation with shift to target
SHIFT = 200
y_train = np.log(train["loss"].values + SHIFT)

X_train = train[cat_cols + num_cols].copy()
X_test = test[cat_cols + num_cols].copy()

# Setup 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_trans = np.zeros(len(train))
test_preds_trans = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.08,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42 + fold,
        verbose=200,
        thread_count=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    oof_preds_trans[val_idx] = model.predict(X_va)
    test_preds_trans += model.predict(X_test) / kf.n_splits

# Reverse target transformation back to original scale
oof_preds = np.exp(oof_preds_trans) - SHIFT
test_preds = np.exp(test_preds_trans) - SHIFT

# Ensure non-negative claim predictions
oof_preds = np.clip(oof_preds, a_min=0, a_max=None)
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Compute validation MAE metric
oof_mae = mean_absolute_error(train["loss"].values, oof_preds)
print(f"Validation MAE: {oof_mae:.5f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test)), "loss": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
