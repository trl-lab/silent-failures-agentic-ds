import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

TARGET = "Heart Disease"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

if train[TARGET].dtype == object:
    y = (train[TARGET] == "Presence").astype(int).to_numpy()
else:
    y = train[TARGET].astype(int).to_numpy()

train = train.drop(columns=[TARGET])

features = [c for c in train.columns if c in test.columns]
X = train[features].copy()
X_test = test[features].copy()

for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


def impute(train_df, val_df, medians=None):
    if medians is None:
        medians = train_df.median(numeric_only=True)
    train_df = train_df.fillna(medians)
    val_df = val_df.fillna(medians)
    return train_df, val_df, medians


model_params = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X), dtype=float)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train_raw = X.iloc[train_idx]
    X_val_raw = X.iloc[val_idx]

    X_train, X_val, _ = impute(X_train_raw, X_val_raw)

    model = HistGradientBoostingClassifier(**model_params)
    model.fit(X_train, y[train_idx])

    oof_predictions[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y[val_idx], oof_predictions[val_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof_predictions)
print(f"OOF ROC AUC: {oof_auc:.6f}")

X_train_final, _, medians = impute(X, X)
final_model = HistGradientBoostingClassifier(**model_params)
final_model.fit(X_train_final, y)

X_test_final = X_test.fillna(medians)
test_predictions = final_model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_predictions), dtype=int),
        "Heart Disease": test_predictions,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print(f"Saved {os.path.join(WORK_DIR, 'submission.csv')}")
