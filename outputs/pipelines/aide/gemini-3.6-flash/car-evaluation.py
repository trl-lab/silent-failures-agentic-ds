import itertools
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ordinal mapping dictionary based on domain hierarchy
ordinal_maps = {
    "buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons": {"2": 2, "4": 4, "more": 6},
    "lug_boot": {"small": 1, "med": 2, "big": 3},
    "safety": {"low": 1, "med": 2, "high": 3},
}

orig_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


def extract_features(df):
    data = df.copy()

    # Apply ordinal numerical mappings
    for col, mapping in ordinal_maps.items():
        data[f"{col}_num"] = data[col].map(mapping).astype(float)

    # Hard decision rule indicator flags
    data["is_unacc_rule"] = (
        (data["persons_num"] == 2) | (data["safety_num"] == 1)
    ).astype(int)

    # Domain interaction features
    data["total_cost"] = data["buying_num"] + data["maint_num"]
    data["cost_product"] = data["buying_num"] * data["maint_num"]
    data["safety_persons"] = data["safety_num"] * data["persons_num"]
    data["safety_lug"] = data["safety_num"] * data["lug_boot_num"]
    data["capacity_prod"] = data["persons_num"] * data["lug_boot_num"]
    data["overall_utility"] = (data["safety_persons"] * data["lug_boot_num"]) / (
        data["total_cost"] + 1e-5
    )

    # Systematic extraction of all 15 pairwise categorical interaction features
    for col1, col2 in itertools.combinations(orig_cols, 2):
        data[f"{col1}_{col2}"] = data[col1].astype(str) + "_" + data[col2].astype(str)

    return data


# Process full dataset for consistent label encoding
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
combined_feat = extract_features(combined)

# Encode categorical columns as integer indicators
cat_cols = [
    c
    for c in combined_feat.columns
    if combined_feat[c].dtype == "object" or combined_feat[c].dtype.name == "category"
]
cat_cols = [c for c in cat_cols if c != "car"]

for c in cat_cols:
    le = LabelEncoder()
    combined_feat[c] = le.fit_transform(combined_feat[c].astype(str))

# Split back to train and test sets
train_processed = combined_feat.iloc[: len(train_df)].copy()
test_processed = combined_feat.iloc[len(train_df) :].copy()

feature_cols = [c for c in train_processed.columns if c not in ["car", "id", "index"]]

# Target label encoding
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df["car"])

# 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), len(le_target.classes_)))
test_preds = np.zeros((len(test_df), len(le_target.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, y_train)):
    X_tr, y_tr = (
        train_processed.iloc[train_idx][feature_cols],
        y_train[train_idx],
    )
    X_va, y_va = train_processed.iloc[val_idx][feature_cols], y_train[val_idx]
    X_te = test_processed[feature_cols]

    # Model 1: HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42 + fold,
    )
    hgb_model.fit(X_tr, y_tr)
    hgb_val_prob = hgb_model.predict_proba(X_va)
    hgb_test_prob = hgb_model.predict_proba(X_te)

    # Model 2: CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42 + fold,
    )
    cb_model.fit(X_tr, y_tr)
    cb_val_prob = cb_model.predict_proba(X_va)
    cb_test_prob = cb_model.predict_proba(X_te)

    # Model 3: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=6,
        eval_metric="mlogloss",
        random_state=42 + fold,
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_val_prob = xgb_model.predict_proba(X_va)
    xgb_test_prob = xgb_model.predict_proba(X_te)

    # Model 4: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        verbose=-1,
        random_state=42 + fold,
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_val_prob = lgb_model.predict_proba(X_va)
    lgb_test_prob = lgb_model.predict_proba(X_te)

    # Soft-voting probability ensemble per fold
    fold_val_prob = (hgb_val_prob + cb_val_prob + xgb_val_prob + lgb_val_prob) / 4.0
    fold_test_prob = (
        hgb_test_prob + cb_test_prob + xgb_test_prob + lgb_test_prob
    ) / 4.0

    oof_preds[val_idx] = fold_val_prob
    test_preds += fold_test_prob / skf.n_splits

# Compute Out-Of-Fold Accuracy
oof_pred_labels = np.argmax(oof_preds, axis=1)
oof_accuracy = accuracy_score(y_train, oof_pred_labels)
print(f"Validation Accuracy (OOF): {oof_accuracy:.5f}")

# Final test predictions
test_pred_labels = np.argmax(test_preds, axis=1)
test_pred_classes = le_target.inverse_transform(test_pred_labels)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "car": test_pred_classes})
submission.to_csv("./working/submission.csv", index=False)
