import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Parse target variable
target_col = "Class"
target_raw = train_df[target_col]

if set(target_raw.unique()) == {"1", "2"}:
    y = (target_raw == "2").astype(int).values
elif set(target_raw.unique()) == {1, 2}:
    y = (target_raw == 2).astype(int).values
elif set(target_raw.unique()) == {"no", "yes"}:
    y = (target_raw == "yes").astype(int).values
else:
    y = pd.to_numeric(target_raw).astype(int).values

train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()


# 3. Feature Engineering Pipeline
def engineer_features(df_train, df_test):
    combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    # Month mapping
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    if "month" in combined.columns:
        combined["month_num"] = (
            combined["month"]
            .astype(str)
            .str.lower()
            .map(month_map)
            .fillna(0)
            .astype(int)
        )
        combined["month_sin"] = np.sin(2 * np.pi * combined["month_num"] / 12)
        combined["month_cos"] = np.cos(2 * np.pi * combined["month_num"] / 12)

    if "day" in combined.columns:
        combined["day_sin"] = np.sin(2 * np.pi * combined["day"].astype(float) / 31)
        combined["day_cos"] = np.cos(2 * np.pi * combined["day"].astype(float) / 31)

    # Financial indicators
    if "balance" in combined.columns:
        combined["is_negative_balance"] = (combined["balance"] < 0).astype(int)
        combined["is_zero_balance"] = (combined["balance"] == 0).astype(int)
        combined["log_balance"] = np.sign(combined["balance"]) * np.log1p(
            np.abs(combined["balance"])
        )

    if "housing" in combined.columns and "loan" in combined.columns:
        combined["has_any_loan"] = (
            (combined["housing"] == "yes") | (combined["loan"] == "yes")
        ).astype(int)
        combined["has_both_loans"] = (
            (combined["housing"] == "yes") & (combined["loan"] == "yes")
        ).astype(int)

    # Contact intensity and duration metrics
    if "duration" in combined.columns and "campaign" in combined.columns:
        combined["duration_per_campaign"] = combined["duration"] / (
            combined["campaign"] + 1
        )

    if "previous" in combined.columns and "campaign" in combined.columns:
        combined["total_contacts"] = combined["previous"] + combined["campaign"]
        combined["prev_to_curr_ratio"] = combined["previous"] / (
            combined["campaign"] + 1
        )

    if "pdays" in combined.columns:
        combined["was_previously_contacted"] = (combined["pdays"] != -1).astype(int)
        combined["pdays_cleaned"] = np.where(
            combined["pdays"] == -1, 999, combined["pdays"]
        )

    # Demographic & financial ratios
    if "balance" in combined.columns and "age" in combined.columns:
        combined["balance_to_age"] = combined["balance"] / (combined["age"] + 1)

    # Group aggregation features
    group_cols = ["job", "education", "marital"]
    for g_col in group_cols:
        if g_col in combined.columns:
            if "duration" in combined.columns:
                dur_mean = combined.groupby(g_col)["duration"].transform("mean")
                combined[f"{g_col}_dur_mean"] = dur_mean
                combined[f"{g_col}_dur_diff"] = combined["duration"] - dur_mean
            if "balance" in combined.columns:
                bal_mean = combined.groupby(g_col)["balance"].transform("mean")
                combined[f"{g_col}_bal_mean"] = bal_mean

    # Convert all categorical columns to 'category' dtype for XGBoost
    cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        combined[col] = combined[col].astype("category")

    train_out = combined.iloc[: len(df_train)].reset_index(drop=True)
    test_out = combined.iloc[len(df_train) :].reset_index(drop=True)

    return train_out, test_out


X_train, X_test = engineer_features(train_features, test_features)

# 4. 5-Fold Cross-Validation Modeling
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

xgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "enable_categorical": True,
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits

# 5. Evaluate and output metric
oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC-AUC: {oof_auc:.6f}")

# 6. Save final predictions
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "Class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
