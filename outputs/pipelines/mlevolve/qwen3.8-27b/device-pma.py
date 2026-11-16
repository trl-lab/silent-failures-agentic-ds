import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
TARGET = "decision_code"

os.makedirs(SUBMISSION_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

y_raw = train[TARGET]
X_full = train.drop(columns=[TARGET])

# Keep only features present in both train and test
feature_cols = [c for c in X_full.columns if c in test.columns]
X_full = X_full[feature_cols].reset_index(drop=True)
X_test = test[feature_cols].reset_index(drop=True)

# Column roles determined from training data only
cat_cols = [c for c in feature_cols if X_full[c].dtype == object or str(X_full[c].dtype) == "category"]
num_cols = [c for c in feature_cols if c not in cat_cols]


def fit_preprocess(X_fit):
    """Fit imputation stats and category lists on training data ONLY."""
    stats = {}
    for c in num_cols:
        s = pd.to_numeric(X_fit[c], errors="coerce")
        stats[c] = float(s.median()) if s.notna().any() else 0.0
    cats = {}
    for c in cat_cols:
        cats[c] = list(pd.Series(X_fit[c]).astype(str).unique())
    return stats, cats


def apply_preprocess(X, stats, cats):
    X = X.copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(stats[c])
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
        X[c] = X[c].cat.set_categories(cats[c])
    return X


# Target encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)
is_multi = num_classes > 2

base_params = dict(
    objective="multiclass" if is_multi else "binary",
    learning_rate=0.05,
    num_leaves=63,
    n_estimators=3000,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
if is_multi:
    base_params["num_class"] = num_classes
eval_metric = "multi_logloss" if is_multi else "auc"

# Split FIRST, then fit preprocessing inside each fold on the training portion only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((len(X_full), num_classes)) if is_multi else np.zeros(len(X_full))
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
    stats, cats = fit_preprocess(X_full.iloc[tr_idx])
    X_tr = apply_preprocess(X_full.iloc[tr_idx], stats, cats)
    X_va = apply_preprocess(X_full.iloc[va_idx], stats, cats)

    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X_tr, y[tr_idx],
        eval_set=[(X_va, y[va_idx])],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    best_iters.append(int(model.best_iteration_ or base_params["n_estimators"]))

    proba = model.predict_proba(X_va)
    oof[va_idx] = proba if is_multi else proba[:, 1]

# Official metric: roc_auc
val_score = (roc_auc_score(y, oof, multi_class="ovr", average="macro") if is_multi
             else roc_auc_score(y, oof))
print(f"Final Validation Score (roc_auc): {val_score:.6f}")

# Final model trained on ALL training data using the median best iteration
stats, cats = fit_preprocess(X_full)
X_final = apply_preprocess(X_full, stats, cats)
X_test_pp = apply_preprocess(X_test, stats, cats)

final_n = int(np.median(best_iters))
final_model = lgb.LGBMClassifier(**{**base_params, "n_estimators": final_n})
final_model.fit(X_final, y, callbacks=[lgb.log_evaluation(period=0)])

test_proba = final_model.predict_proba(X_test_pp)

# Submission semantics: predicted values for decision_code
if is_multi:
    pred = le.inverse_transform(np.argmax(test_proba, axis=1))
else:
    # Probability of the positive (last-encoded) class, suitable for roc_auc
    pred = test_proba[:, 1]

submission = pd.DataFrame({"id": np.arange(len(test)), "decision_code": pred})
submission.to_csv(SUBMISSION_PATH, index=False)
print(submission.head())
print(submission.shape)