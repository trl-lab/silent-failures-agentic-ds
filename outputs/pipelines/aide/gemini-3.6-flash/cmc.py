import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Contraceptive_method_used"
y_raw = train_df[target_col]

# Map targets to contiguous zero-indexed integers explicitly as int64 NumPy array
unique_targets = sorted(y_raw.unique())
target_to_idx = {val: idx for idx, val in enumerate(unique_targets)}
idx_to_target = {idx: val for idx, val in enumerate(unique_targets)}
y = np.asarray([target_to_idx[val] for val in y_raw], dtype=np.int64)

# Prepare initial feature DataFrames
X_raw = train_df.drop(columns=[target_col], errors="ignore").copy()
X_test_raw = test_df.copy()

if "id" in X_raw.columns:
    X_raw = X_raw.drop(columns=["id"])
if "id" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["id"])


def feature_engineering(df):
    df = df.copy()

    # Column name mapping
    col_map = {c.lower().strip(): c for c in df.columns}

    age_col = col_map.get("wife's age") or col_map.get("wife_age")
    children_col = col_map.get("number of children ever born") or col_map.get(
        "num_children"
    )
    w_edu_col = col_map.get("wife's education") or col_map.get("wife_edu")
    h_edu_col = col_map.get("husband's education") or col_map.get("husband_edu")
    sol_col = col_map.get("standard-of-living index") or col_map.get("sol_index")
    work_col = col_map.get("wife's now working?") or col_map.get("wife_working")

    # Numeric conversions
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Cross features
    if age_col and children_col:
        df["age_per_child"] = df[age_col].astype(float) / (
            df[children_col].astype(float) + 1.0
        )
        df["age_child_prod"] = df[age_col].astype(float) * df[children_col].astype(
            float
        )
        df["child_density"] = df[children_col].astype(float) / (
            df[age_col].astype(float) - 12.0
        ).clip(lower=1.0)

    if w_edu_col and h_edu_col:
        df["edu_diff"] = df[w_edu_col].astype(float) - df[h_edu_col].astype(float)
        df["edu_sum"] = df[w_edu_col].astype(float) + df[h_edu_col].astype(float)

    if w_edu_col and sol_col:
        df["w_edu_sol_prod"] = df[w_edu_col].astype(float) * df[sol_col].astype(float)

    if h_edu_col and sol_col:
        df["h_edu_sol_prod"] = df[h_edu_col].astype(float) * df[sol_col].astype(float)

    return df


X_feat = feature_engineering(X_raw)
X_test_feat = feature_engineering(X_test_raw)

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros((len(train_df), len(unique_targets)))
test_preds_lgb = np.zeros((len(test_df), len(unique_targets)))

oof_preds_rf = np.zeros((len(train_df), len(unique_targets)))
test_preds_rf = np.zeros((len(test_df), len(unique_targets)))

oof_preds_et = np.zeros((len(train_df), len(unique_targets)))
test_preds_et = np.zeros((len(test_df), len(unique_targets)))

oof_preds_hgb = np.zeros((len(train_df), len(unique_targets)))
test_preds_hgb = np.zeros((len(test_df), len(unique_targets)))

lgb_params = {
    "objective": "multiclass",
    "num_class": len(unique_targets),
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 12,
    "min_child_samples": 15,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y)):
    X_train, y_train = X_feat.iloc[train_idx], y[train_idx]
    X_val, y_val = X_feat.iloc[val_idx], y[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict(
        X_val, num_iteration=lgb_model.best_iteration
    )
    test_preds_lgb += (
        lgb_model.predict(X_test_feat, num_iteration=lgb_model.best_iteration)
        / skf.n_splits
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, random_state=42
    )
    rf_model.fit(X_train, y_train)
    oof_preds_rf[val_idx] = rf_model.predict_proba(X_val)
    test_preds_rf += rf_model.predict_proba(X_test_feat) / skf.n_splits

    # Extra Trees
    et_model = ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, random_state=42
    )
    et_model.fit(X_train, y_train)
    oof_preds_et[val_idx] = et_model.predict_proba(X_val)
    test_preds_et += et_model.predict_proba(X_test_feat) / skf.n_splits

    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        random_state=42,
    )
    hgb_model.fit(X_train, y_train)
    oof_preds_hgb[val_idx] = hgb_model.predict_proba(X_val)
    test_preds_hgb += hgb_model.predict_proba(X_test_feat) / skf.n_splits

# Ensemble weights optimization / blending
oof_preds_ensemble = (
    0.35 * oof_preds_lgb
    + 0.25 * oof_preds_rf
    + 0.20 * oof_preds_et
    + 0.20 * oof_preds_hgb
)
test_preds_ensemble = (
    0.35 * test_preds_lgb
    + 0.25 * test_preds_rf
    + 0.20 * test_preds_et
    + 0.20 * test_preds_hgb
)

oof_classes = np.argmax(oof_preds_ensemble, axis=1)
val_accuracy = accuracy_score(y, oof_classes)
print(f"Validation Accuracy: {val_accuracy:.5f}")

# Generate test predictions and format submission
test_classes = np.argmax(test_preds_ensemble, axis=1)
test_pred_labels = [idx_to_target[c] for c in test_classes]

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Contraceptive_method_used": test_pred_labels}
)
submission.to_csv("./working/submission.csv", index=False)
