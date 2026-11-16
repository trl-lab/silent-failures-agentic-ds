import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# ============================================================

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget distribution:\n{train['Company response to consumer'].value_counts()}")

# Identify column types
target_col = "Company response to consumer"
train_target = train[target_col].copy()
train_data = train.drop(columns=[target_col])

text_cols = [
    c
    for c in train_data.columns
    if train_data[c].dtype == "object" and train_data[c].str.len().mean() > 50
]
cat_cols = [
    c
    for c in train_data.columns
    if train_data[c].dtype == "object" and c not in text_cols
]
num_cols = [
    c
    for c in train_data.columns
    if train_data[c].dtype in ["int64", "float64", "int32", "float32"]
]
date_cols = [
    c for c in train_data.columns if "date" in c.lower() or "time" in c.lower()
]

# Check for date-like columns in object dtype
for c in train_data.columns:
    if train_data[c].dtype == "object":
        sample = train_data[c].dropna().iloc[0] if train_data[c].notna().any() else ""
        if re.match(r"\d{4}-\d{2}-\d{2}", str(sample)):
            if c not in date_cols:
                date_cols.append(c)
            if c in cat_cols:
                cat_cols.remove(c)

print(f"Text columns: {text_cols}")
print(f"Cat columns: {cat_cols}")
print(f"Num columns: {num_cols}")
print(f"Date columns: {date_cols}")

# Convert date columns to datetime
for c in date_cols:
    train_data[c] = pd.to_datetime(train_data[c], errors="coerce")
    test[c] = pd.to_datetime(test[c], errors="coerce")


# Feature Engineering Function
def engineer_features(df, is_train=False, train_df=None, freq_maps=None):
    """Engineer features from raw data."""
    df = df.copy()

    # --- Temporal Features ---
    for c in date_cols:
        if c in df.columns:
            df[f"{c}_year"] = df[c].dt.year
            df[f"{c}_month"] = df[c].dt.month
            df[f"{c}_day"] = df[c].dt.day
            df[f"{c}_dayofweek"] = df[c].dt.dayofweek
            df[f"{c}_quarter"] = df[c].dt.quarter
            df[f"{c}_is_weekend"] = (df[c].dt.dayofweek >= 5).astype(int)
            if is_train:
                df[f"{c}_days_since_min"] = (df[c] - df[c].min()).dt.days
            else:
                ref_min = train_df[c].min()
                df[f"{c}_days_since_min"] = (df[c] - ref_min).dt.days

    # --- Text Features ---
    for c in text_cols:
        if c in df.columns:
            text = df[c].fillna("")
            df[f"{c}_len"] = text.str.len()
            df[f"{c}_word_count"] = text.str.split().str.len()
            df[f"{c}_avg_word_len"] = df[f"{c}_len"] / (df[f"{c}_word_count"] + 1)
            df[f"{c}_excl_count"] = text.str.count("!")
            df[f"{c}_quest_count"] = text.str.count(r"\?")
            df[f"{c}_upper_ratio"] = text.str.count(r"[A-Z]") / (text.str.len() + 1)
            df[f"{c}_digit_count"] = text.str.count(r"\d")
            df[f"{c}_has_number"] = (text.str.count(r"\d") > 0).astype(int)
            df[f"{c}_has_refund"] = text.str.contains(
                r"refund|money back", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_scam"] = text.str.contains(
                r"scam|fraud|fraudulent", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_dispute"] = text.str.contains(
                r"dispute|disagreement|unauthorized", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_delay"] = text.str.contains(
                r"delay|waiting|never received|still waiting", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_apolog"] = text.str.contains(
                r"apolog|sorry|regret", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_threat"] = text.str.contains(
                r"threat|legal action|sue|lawyer", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_resolved"] = text.str.contains(
                r"resolved|fixed|corrected", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_denied"] = text.str.contains(
                r"denied|refused|declined", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_interest"] = text.str.contains(
                r"interest|fee|charge", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_credit"] = text.str.contains(
                r"credit|score|report", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_loan"] = text.str.contains(
                r"loan|mortgage|payment", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_bank"] = text.str.contains(
                r"bank|account|deposit|withdraw", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_insurance"] = text.str.contains(
                r"insurance|policy|claim", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_identity"] = text.str.contains(
                r"identity|stolen|theft", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_overdraft"] = text.str.contains(
                r"overdraft|negative|insufficient", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_automated"] = text.str.contains(
                r"automated|robot|call center", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_transfer"] = text.str.contains(
                r"transfer|wire|sent money", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_duplicate"] = text.str.contains(
                r"duplicate|double|twice", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_cancel"] = text.str.contains(
                r"cancel|terminate|close account", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_late"] = text.str.contains(
                r"late|overdue|past due", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_min_payment"] = text.str.contains(
                r"minimum payment|min payment", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_credit_limit"] = text.str.contains(
                r"credit limit|limit increase", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_statement"] = text.str.contains(
                r"statement|billing|invoice", case=False, regex=True
            ).astype(int)
            df[f"{c}_has_dispute_resolved"] = (
                df[f"{c}_has_dispute"] * df[f"{c}_has_resolved"]
            ).astype(int)
            df[f"{c}_has_scam_unresolved"] = (
                df[f"{c}_has_scam"] * (1 - df[f"{c}_has_resolved"])
            ).astype(int)

    # --- Categorical Features: Frequency Encoding ---
    if is_train:
        freq_maps = {}
        for c in cat_cols:
            if c in df.columns:
                freq = df[c].value_counts(normalize=True).to_dict()
                freq_maps[c] = freq
                df[f"{c}_freq"] = df[c].map(freq)
                count = df[c].value_counts().to_dict()
                df[f"{c}_count"] = df[c].map(count).astype(float)
    else:
        for c in cat_cols:
            if c in df.columns and c in freq_maps:
                df[f"{c}_freq"] = df[c].map(freq_maps[c]).fillna(0)
                df[f"{c}_count"] = 0.0
            elif c in df.columns:
                df[f"{c}_freq"] = 0.0
                df[f"{c}_count"] = 0.0
            else:
                df[f"{c}_freq"] = 0.0
                df[f"{c}_count"] = 0.0
        # Ensure _freq_missing and _count_missing columns exist for test (they are 0 since no NaNs)
        for c in cat_cols:
            if c in freq_maps:
                if f"{c}_freq_missing" not in df.columns:
                    df[f"{c}_freq_missing"] = 0
                if f"{c}_count_missing" not in df.columns:
                    df[f"{c}_count_missing"] = 0

    # --- Interaction Features ---
    if "Product" in df.columns and "Issue" in df.columns:
        df["Product_Issue"] = df["Product"].fillna("") + "_" + df["Issue"].fillna("")
    if "Product" in df.columns and "Sub-Issue" in df.columns:
        df["Product_SubIssue"] = (
            df["Product"].fillna("") + "_" + df["Sub-Issue"].fillna("")
        )
    if "Issue" in df.columns and "Sub-Issue" in df.columns:
        df["Issue_SubIssue"] = df["Issue"].fillna("") + "_" + df["Sub-Issue"].fillna("")

    # --- Numeric Feature Engineering ---
    for c in num_cols:
        if c in df.columns:
            df[f"{c}_log"] = np.log1p(df[c].clip(lower=0))

    # --- Missing Value Indicators ---
    for c in df.columns:
        if df[c].isnull().sum() > 0:
            df[f"{c}_missing"] = df[c].isnull().astype(int)
        else:
            # Ensure the column exists even if no NaN (for consistency between train/test)
            if f"{c}_missing" not in df.columns:
                df[f"{c}_missing"] = 0

    return df, freq_maps


# Apply feature engineering
train_data, freq_maps = engineer_features(train_data, is_train=True)
test_data, _ = engineer_features(
    test, is_train=False, train_df=train_data, freq_maps=freq_maps
)

# Identify final feature columns (exclude original text and date columns)
feature_cols = [c for c in train_data.columns if c not in text_cols + date_cols]
# Remove original categorical columns that we've encoded (keep interaction columns)
interaction_cols = [
    c
    for c in ["Product_Issue", "Product_SubIssue", "Issue_SubIssue"]
    if c in train_data.columns
]
feature_cols = [c for c in feature_cols if c not in cat_cols or c in interaction_cols]

print(f"\nNumber of features: {len(feature_cols)}")

# Label encode remaining object columns (fit on train only)
label_encoders = {}
remaining_obj = [c for c in feature_cols if train_data[c].dtype == "object"]
for c in remaining_obj:
    le = pd.Series(range(train_data[c].nunique()), index=train_data[c].unique())
    label_encoders[c] = le
    train_data[c] = train_data[c].map(le).fillna(-1).astype(int)
    test_data[c] = test_data[c].map(le).fillna(-1).astype(int)

# Final feature selection
feature_cols = [c for c in train_data.columns if c not in text_cols + date_cols]
feature_cols = [c for c in feature_cols if c not in cat_cols or c in interaction_cols]
print(f"Final feature count: {len(feature_cols)}")

# Create train/validation split (stratified)
X_train_full = train_data[feature_cols].copy()
y_train_full = train_target.copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

# Ensure test_data has all feature columns (fill missing ones with 0)
for c in feature_cols:
    if c not in test_data.columns:
        test_data[c] = 0

X_test = test_data[feature_cols].copy()

print(f"\nTrain split: {X_train.shape}")
print(f"Validation split: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Handle any remaining NaN/inf
for df in [X_train, X_val, X_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(-999, inplace=True)

# Baseline accuracy
majority_class = y_train.value_counts().idxmax()
baseline_acc = accuracy_score(y_val, [majority_class] * len(y_val))
print(f"\nBaseline (majority class) accuracy: {baseline_acc:.4f}")

# ============================================================
# STEP 2 & 3: MODEL DESIGN + TRAINING & EVALUATION
# ============================================================

target_classes = list(train_target.unique())
n_classes = len(target_classes)
print(f"Number of classes: {n_classes}")
print(f"Classes: {target_classes}")
print(f"Class distribution in train:\n{pd.Series(y_train).value_counts()}")

# Define LightGBM classifier
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Train with early stopping on logloss
early_stop = lgb.early_stopping(stopping_rounds=300, verbose=False)
metric_cb = lgb.log_evaluation(period=0)

print("Starting training...")
model.fit(
    X_train.values,
    y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_metric="multi_logloss",
    callbacks=[early_stop, metric_cb],
)

print(f"Early stopping at iteration: {model.best_iteration_}")
print(f"Best validation logloss: {model.best_score_['valid_0']['multi_logloss']:.4f}")

# --- Validation Inference (same pipeline as test) ---
# Use best_iteration_ to ensure optimal predictions
best_iter = model.best_iteration_
booster = model.booster_
booster.best_iteration = best_iter

val_probs = model.predict_proba(X_val.values)
val_preds_idx = np.argmax(val_probs, axis=1)
val_pred_labels = model.classes_[val_preds_idx]

val_accuracy = accuracy_score(y_val.values, val_pred_labels)
print(f"Validation Accuracy (final): {val_accuracy:.4f}")

# --- Test Inference (identical pipeline) ---
test_probs = model.predict_proba(X_test.values)
test_preds_idx = np.argmax(test_probs, axis=1)
test_pred_labels = model.classes_[test_preds_idx]

# --- Generate Submission ---
submission = pd.DataFrame(
    {
        "id": range(len(test_pred_labels)),
        "Company response to consumer": test_pred_labels,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Submission sample:\n{submission.head()}")
print(
    f"Prediction distribution:\n{submission['Company response to consumer'].value_counts(normalize=True)}"
)

# Save model
model.booster_.save_model("./working/lgbm_final_model.txt")

# Final output
score = val_accuracy
print(f"Final Validation Score: {score}")
