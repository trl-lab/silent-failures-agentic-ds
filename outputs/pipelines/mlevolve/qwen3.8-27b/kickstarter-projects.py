import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import os
import re

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ STEP 1: DATA LOADING & FEATURE ENGINEERING ============

print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Target distribution:\n{train_df['status'].value_counts()}")

# Binarize target: successful = 1, else = 0
train_df["target"] = (train_df["status"] == "successful").astype(int)
print(f"Target distribution (binarized):\n{train_df['target'].value_counts()}")

# Separate target
y = train_df["target"].values
train_df = train_df.drop("target", axis=1)

# Create train/val split
train_split, val_split = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=(y)
)
y_train = y[train_split.index]
y_val = y[val_split.index]

print(f"Train split: {train_split.shape}, Val split: {val_split.shape}")

# Combine for feature engineering
all_df = pd.concat([train_split, val_split, test_df], axis=0, ignore_index=True)

# === FEATURE ENGINEERING ===

# 1. Text features
text_cols = [c for c in all_df.columns if all_df[c].dtype == object]
print(f"Object columns: {text_cols}")


def text_features(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(lambda x: len(str(x)))


def word_count(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(lambda x: len(str(x).split()))


def avg_word_length(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )


def exclamation_count(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(lambda x: str(x).count("!"))


def question_count(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(lambda x: str(x).count("?"))


def has_numbers(series):
    if series.dtype != object:
        return pd.Series(0, index=series.index)
    return series.apply(lambda x: int(bool(re.search(r"\d", str(x)))))


for col in text_cols:
    if col == "status":
        continue
    all_df[f"{col}_len"] = text_features(all_df[col])
    all_df[f"{col}_word_count"] = word_count(all_df[col])
    all_df[f"{col}_avg_word_len"] = avg_word_length(all_df[col])
    all_df[f"{col}_excl_count"] = exclamation_count(all_df[col])
    all_df[f"{col}_quest_count"] = question_count(all_df[col])
    all_df[f"{col}_has_numbers"] = has_numbers(all_df[col])

# 2. Temporal features
for col in all_df.columns:
    if all_df[col].dtype == object:
        try:
            parsed = pd.to_datetime(all_df[col], errors="coerce")
            if parsed.notna().sum() > len(all_df) * 0.5:
                all_df[f"{col}_year"] = parsed.dt.year
                all_df[f"{col}_month"] = parsed.dt.month
                all_df[f"{col}_day"] = parsed.dt.day
                all_df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                all_df[f"{col}_quarter"] = parsed.dt.quarter
                all_df[f"{col}_weekofyear"] = parsed.dt.isocalendar().week.astype(int)
                all_df[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
                all_df[f"{col}_is_holiday_season"] = parsed.dt.month.isin(
                    [11, 12]
                ).astype(int)
        except:
            pass

# 3. Numerical feature engineering
num_cols = [
    c
    for c in all_df.columns
    if all_df[c].dtype in [np.int64, np.float64, np.int32, np.float32]
]

for col in num_cols:
    if all_df[col].dtype in [np.float64, np.float32]:
        positive_mask = all_df[col] > 0
        if positive_mask.sum() > 0 and positive_mask.sum() < len(all_df):
            all_df[f"{col}_log"] = np.log1p(all_df[col].clip(lower=0))
        elif (all_df[col] > 0).all():
            all_df[f"{col}_log"] = np.log1p(all_df[col])

# 4. Category encoding - frequency encoding (fit on train only)
cat_cols = [
    c
    for c in all_df.columns
    if all_df[c].dtype == object
    and c != "status"
    and not any(
        s in c.lower()
        for s in [
            "_len",
            "_word",
            "_avg",
            "_excl",
            "_quest",
            "_has",
            "_year",
            "_month",
            "_day",
            "_quarter",
            "_week",
            "_is_",
        ]
    )
]
print(f"Categorical columns: {cat_cols}")

for col in cat_cols:
    if col not in train_split.columns:
        continue
    freq_map = train_split[col].value_counts(normalize=True)
    all_df[f"{col}_freq"] = all_df[col].map(freq_map).fillna(0)

# 5. Duration features
launch_col = None
end_col = None
for c in all_df.columns:
    if "launch" in c.lower() or "start" in c.lower() or "created" in c.lower():
        if "_year" in c or "_month" in c or all_df[c].dtype == object:
            launch_col = c
    if "end" in c.lower() or "deadline" in c.lower() or "close" in c.lower():
        if "_year" in c or "_month" in c or all_df[c].dtype == object:
            end_col = c

if launch_col and end_col:
    try:
        launch_dates = pd.to_datetime(all_df[launch_col], errors="coerce")
        end_dates = pd.to_datetime(all_df[end_col], errors="coerce")
        all_df["campaign_duration_days"] = (end_dates - launch_dates).dt.days
        all_df["campaign_duration_log"] = np.log1p(
            all_df["campaign_duration_days"].clip(lower=0)
        )
    except:
        pass

# 6. Category-specific target encoding (fit on train only)
cat_col = None
for c in all_df.columns:
    if (
        "category" in c.lower()
        or "sub_category" in c.lower()
        or "subcategory" in c.lower()
    ):
        cat_col = c
        break

if cat_col and cat_col in all_df.columns:
    target_enc = pd.DataFrame({"cat": train_split[cat_col], "target": y_train})
    cat_means = target_enc.groupby("cat")["target"].mean()
    all_df[f"{cat_col}_target_enc"] = (
        all_df[cat_col].map(cat_means).fillna(y_train.mean())
    )

# 7. Remove original text columns and status
drop_cols = [c for c in text_cols if c != "status"]
all_df = all_df.drop(drop_cols, axis=1)

if "status" in all_df.columns:
    all_df = all_df.drop("status", axis=1)

# 8. Split back and handle NaNs
train_features = all_df.iloc[: len(train_split)].copy()
val_features = all_df.iloc[len(train_split) : len(train_split) + len(val_split)].copy()
test_features = all_df.iloc[len(train_split) + len(val_split) :].copy()

# Impute using train statistics
for col in train_features.columns:
    if train_features[col].isnull().any():
        if train_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            impute_val = train_features[col].median()
        else:
            impute_val = (
                train_features[col].mode()[0]
                if not train_features[col].mode().empty
                else "unknown"
            )
        train_features[col] = train_features[col].fillna(impute_val)
        val_features[col] = val_features[col].fillna(impute_val)
        test_features[col] = test_features[col].fillna(impute_val)

# 9. Ensure all numeric, replace inf
for df in [train_features, val_features, test_features]:
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Final feature shapes:")
print(f"  Train: {train_features.shape}")
print(f"  Val: {val_features.shape}")
print(f"  Test: {test_features.shape}")

# ============ STEP 2: MODEL DESIGN & DATA PREPARATION ============

feature_names = train_features.columns.tolist()
n_features = len(feature_names)
print(f"Number of features: {n_features}")

X_train = train_features.values.astype(np.float32)
X_val = val_features.values.astype(np.float32)
X_test = test_features.values.astype(np.float32)

# Replace NaN/inf with 0 for tree-based models
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# ============ MODEL ARCHITECTURE: LightGBM + XGBoost Ensemble ============

# LightGBM parameters
lgbm_params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
}

# XGBoost parameters
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
}


# ============ STEP 3: TRAINING & EVALUATION ============

print("Training LightGBM...")
lgbm_model = lgb.LGBMClassifier(**lgbm_params)
lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
)

print(f"\nLightGBM best iteration: {lgbm_model.best_iteration_}")
lgbm_val_preds = lgbm_model.predict_proba(X_val)[:, 1]
lgbm_val_auc = roc_auc_score(y_val, lgbm_val_preds)
print(f"LightGBM Validation AUC: {lgbm_val_auc:.6f}")

print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=100)
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\nXGBoost best iteration: {xgb_model.best_iteration}")
xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_preds)
print(f"XGBoost Validation AUC: {xgb_val_auc:.6f}")

# Ensemble: average of LightGBM and XGBoost predictions
val_preds_final = 0.5 * lgbm_val_preds + 0.5 * xgb_val_preds
final_score = roc_auc_score(y_val, val_preds_final)
print(f"\nEnsemble Validation AUC: {final_score:.6f}")

# Test inference
lgbm_test_preds = lgbm_model.predict_proba(X_test)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]
test_preds = 0.5 * lgbm_test_preds + 0.5 * xgb_test_preds
test_preds = np.clip(test_preds, 0.0, 1.0)

print(
    f"Test predictions: shape={test_preds.shape}, mean={test_preds.mean():.4f}, std={test_preds.std():.4f}"
)

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "status": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")

print(f"Final Validation Score: {final_score:.6f}")
