import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "TARGET"
id_col = "ID" if "ID" in train_df.columns else None

y = train_df[target_col].values
X_train_raw = train_df.drop(
    columns=[c for c in [target_col, id_col] if c in train_df.columns]
)
X_test_raw = test_df.drop(
    columns=[c for c in [target_col, id_col] if c in test_df.columns]
)

# 2. Remove constant and duplicate features
constant_cols = [
    c for c in X_train_raw.columns if X_train_raw[c].nunique(dropna=False) <= 1
]
X_train_raw = X_train_raw.drop(columns=constant_cols)
X_test_raw = X_test_raw.drop(columns=constant_cols)

# Remove duplicate columns based on values
transposed_train = X_train_raw.T
dup_mask = transposed_train.duplicated(keep="first")
dup_cols = transposed_train.index[dup_mask].tolist()
X_train_raw = X_train_raw.drop(columns=dup_cols)
X_test_raw = X_test_raw.drop(columns=dup_cols)


# 3. Feature engineering
def engineer_features(df):
    df_feat = df.copy()

    # Row-wise summary statistics
    df_feat["num_zeros"] = (df == 0).sum(axis=1)
    df_feat["num_non_zeros"] = (df != 0).sum(axis=1)
    df_feat["row_mean"] = df.mean(axis=1)
    df_feat["row_std"] = df.std(axis=1)
    df_feat["row_max"] = df.max(axis=1)

    # var38 specific transformations (117310.979 is the known default/median value)
    if "var38" in df.columns:
        df_feat["var38mc"] = np.isclose(df["var38"], 117310.97914835788).astype(int)
        df_feat["log_var38"] = np.log1p(np.maximum(0, df["var38"]))

    # var3 specific transformations (-999999 represents missing)
    if "var3" in df.columns:
        df_feat["var3_is_missing"] = (df["var3"] == -999999).astype(int)
        df_feat["var3_clean"] = df["var3"].replace(-999999, 2)

    return df_feat


X_train = engineer_features(X_train_raw)
X_test = engineer_features(X_test_raw)

# Align columns
feature_names = list(X_train.columns)
X_train = X_train[feature_names]
X_test = X_test[feature_names]

# 4. Stratified 5-Fold Cross Validation with XGBoost
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 2,
    "seed": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred = model.predict(dval)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(dtest) / n_splits

    fold_auc = roc_auc_score(y_va, val_pred)
    print(f"Fold {fold + 1} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF ROC AUC: {cv_auc:.6f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "TARGET": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
