import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "target"
id_cols = [c for c in ["ID", "id"] if c in train.columns]

y = train[target_col].values
X_train = train.drop(columns=[target_col] + id_cols)
X_test = test.drop(columns=[c for c in id_cols if c in test.columns])

# 2. Feature Engineering & Preprocessing
# Missing value count per row is a known strong signal for this dataset
X_train["null_count"] = X_train.isnull().sum(axis=1)
X_test["null_count"] = X_test.isnull().sum(axis=1)

# Identify categorical features
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    # Frequency encoding
    freq_map = pd.concat([X_train[col], X_test[col]]).value_counts().to_dict()
    X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0)
    X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0)

    # Label encoding for native categorical support
    le = LabelEncoder()
    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# 3. Model Training & 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 4. Evaluation
oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
cv_log_loss = log_loss(y, oof_preds_clipped)
print(f"5-Fold Cross-Validation Log Loss: {cv_log_loss:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test)), "target": np.clip(test_preds, 1e-15, 1 - 1e-15)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
