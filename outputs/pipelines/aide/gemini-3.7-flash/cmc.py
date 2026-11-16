import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Contraceptive_method_used"
features = [c for c in train_df.columns if c != target_col]

# Encode target cleanly to standard numpy integer array
le = LabelEncoder()
y_train = np.array(le.fit_transform(train_df[target_col]), dtype=np.int64)
unique_classes = le.classes_


def feature_engineering(df):
    data = df.copy()

    # Column name mapping helper
    col_map = {c.lower().replace(" ", "_").replace("'", ""): c for c in data.columns}

    age_col = next((col_map[k] for k in col_map if "age" in k and "wife" in k), None)
    children_col = next(
        (col_map[k] for k in col_map if "children" in k or "born" in k), None
    )
    w_edu_col = next(
        (
            col_map[k]
            for k in col_map
            if "education" in k and ("wife" in k or "w_" in k)
        ),
        None,
    )
    h_edu_col = next(
        (
            col_map[k]
            for k in col_map
            if "education" in k and ("husband" in k or "h_" in k)
        ),
        None,
    )
    sol_col = next(
        (col_map[k] for k in col_map if "living" in k or "standard" in k), None
    )
    media_col = next((col_map[k] for k in col_map if "media" in k), None)
    work_col = next(
        (col_map[k] for k in col_map if "work" in k or "now_working" in k), None
    )
    rel_col = next((col_map[k] for k in col_map if "religion" in k), None)

    if age_col and children_col:
        rep_years = np.maximum(data[age_col].astype(float) - 15.0, 1.0)
        data["children_per_rep_year"] = data[children_col].astype(float) / rep_years
        data["has_children"] = (data[children_col].astype(float) > 0).astype(int)
        data["large_family"] = (data[children_col].astype(float) >= 4).astype(int)
        data["age_x_children"] = data[age_col].astype(float) * data[
            children_col
        ].astype(float)

    if w_edu_col and h_edu_col:
        w_edu = data[w_edu_col].astype(float)
        h_edu = data[h_edu_col].astype(float)
        data["edu_diff"] = w_edu - h_edu
        data["edu_sum"] = w_edu + h_edu
        data["wife_more_educated"] = (w_edu > h_edu).astype(int)

    if sol_col and work_col:
        data["sol_work_inter"] = (
            data[sol_col].astype(str) + "_" + data[work_col].astype(str)
        )

    if rel_col and media_col:
        data["rel_media_inter"] = (
            data[rel_col].astype(str) + "_" + data[media_col].astype(str)
        )

    return data


X_train_df = feature_engineering(train_df[features])
X_test_df = feature_engineering(test_df[features])

# Identify categorical features
cat_cols = []
for col in X_train_df.columns:
    if (
        X_train_df[col].dtype == "object"
        or X_train_df[col].dtype.name == "category"
        or (
            X_train_df[col].nunique() <= 10
            and col not in ["children_per_rep_year", "age_x_children"]
        )
    ):
        cat_cols.append(col)

# Format categorical columns for CatBoost & LightGBM
X_train_cb = X_train_df.copy()
X_test_cb = X_test_df.copy()
for col in cat_cols:
    X_train_cb[col] = X_train_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

X_train_lgb = X_train_df.copy()
X_test_lgb = X_test_df.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# Stratified 5-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros((len(train_df), len(unique_classes)))
test_probs = np.zeros((len(test_df), len(unique_classes)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train)):
    # CatBoost Model
    cb_tr_x, cb_val_x = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    cb_model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=5,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42 + fold,
        cat_features=cat_cols,
        l2_leaf_reg=4.0,
        verbose=False,
    )
    cb_model.fit(cb_tr_x, y_tr, eval_set=(cb_val_x, y_val), early_stopping_rounds=60)

    cb_val_preds = cb_model.predict_proba(cb_val_x)
    cb_test_preds = cb_model.predict_proba(X_test_cb)

    # LightGBM Model
    lgb_tr_x, lgb_val_x = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    lgb_model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective="multiclass",
        num_class=len(unique_classes),
        verbose=-1,
    )
    lgb_model.fit(
        lgb_tr_x,
        y_tr,
        eval_set=[(lgb_val_x, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_val_preds = lgb_model.predict_proba(lgb_val_x)
    lgb_test_preds = lgb_model.predict_proba(X_test_lgb)

    # Blend fold predictions
    blend_val = 0.55 * cb_val_preds + 0.45 * lgb_val_preds
    blend_test = 0.55 * cb_test_preds + 0.45 * lgb_test_preds

    oof_probs[val_idx] = blend_val
    test_probs += blend_test / n_splits

oof_pred_labels = np.argmax(oof_probs, axis=1)
cv_accuracy = accuracy_score(y_train, oof_pred_labels)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# Inverse transform to original labels
test_pred_indices = np.argmax(test_probs, axis=1)
test_pred_labels = le.inverse_transform(test_pred_indices)

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
