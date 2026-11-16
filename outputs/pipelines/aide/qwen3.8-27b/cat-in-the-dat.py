import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "target"

# Prepare target and features
y = train[TARGET].astype(int)
train = train.drop(columns=[TARGET], errors="ignore")

if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])

feature_cols = [c for c in train.columns if c not in [TARGET, "id"]]

# Ensure test has the same feature columns
for col in feature_cols:
    if col not in test.columns:
        test[col] = np.nan
test = test[feature_cols].copy()

# Identify categorical columns from training data
cat_cols = [
    col
    for col in feature_cols
    if train[col].dtype == object or train[col].dtype.name == "category"
]
num_cols = [col for col in feature_cols if col not in cat_cols]


def get_categories(df, cat_columns):
    """Fit category lists on a training dataframe only."""
    categories = {}
    for col in cat_columns:
        if col in df.columns:
            categories[col] = list(pd.unique(df[col].dropna()))
        else:
            categories[col] = []
    return categories


def encode_features(df, categories, cat_columns, num_columns):
    """
    Label-encode categorical columns using categories fitted on training data.
    Unknown or missing categories become -1, which LightGBM treats as missing.
    """
    out = df.copy()

    for col in cat_columns:
        if col not in out.columns:
            out[col] = -1
            continue

        if col in categories and len(categories[col]) > 0:
            out[col] = pd.Categorical(
                out[col], categories=categories[col]
            ).codes.astype("int32")
        else:
            out[col] = -1

    for col in num_columns:
        if col in out.columns:
            out[col] = out[col].astype("float32")

    return out


params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
    "verbosity": -1,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
best_iterations = []

# 5-fold cross-validation for generalization estimate
for fold, (train_idx, val_idx) in enumerate(cv.split(train, y), start=1):
    X_train_raw = train.iloc[train_idx]
    X_val_raw = train.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Fit categories on the fold's training data only
    fold_categories = get_categories(X_train_raw, cat_cols)

    X_train = encode_features(X_train_raw, fold_categories, cat_cols, num_cols)
    X_val = encode_features(X_val_raw, fold_categories, cat_cols, num_cols)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_cols,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_cols,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    auc_scores.append(auc)
    best_iterations.append(model.best_iteration)

    print(f"Fold {fold}: AUC = {auc:.6f}, best_iteration = {model.best_iteration}")

mean_auc = float(np.mean(auc_scores))
std_auc = float(np.std(auc_scores))
print(f"5-fold CV ROC AUC: {mean_auc:.6f} +/- {std_auc:.6f}")

# Retrain a single final model on the full training set
final_categories = get_categories(train, cat_cols)
X_full = encode_features(train, final_categories, cat_cols, num_cols)
X_test = encode_features(test, final_categories, cat_cols, num_cols)

final_rounds = max(1, int(np.mean(best_iterations)))

dtrain_full = lgb.Dataset(
    X_full,
    label=y,
    categorical_feature=cat_cols,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
)

test_pred = final_model.predict(X_test)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "target": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
