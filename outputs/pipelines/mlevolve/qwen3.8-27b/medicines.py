import pandas as pd
import numpy as np
import os
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Target distribution:\n{train['Medicine status'].value_counts()}")

# ============================================================
# 2. IDENTIFY COLUMN TYPES & CREATE SPLIT
# ============================================================
target_col = "Medicine status"
y = train[target_col].copy()
train = train.drop(columns=[target_col])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target classes: {le_target.classes_}")

train_idx, val_idx = train_test_split(
    range(len(train)), test_size=0.2, random_state=42, stratify=y_encoded
)

train_split = train.iloc[train_idx].copy()
val_split = train.iloc[val_idx].copy()
y_train = y_encoded[train_idx]
y_val = y_encoded[val_idx]

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ============================================================
# 3. FEATURE ENGINEERING PIPELINE
# ============================================================


def full_pipeline(df, is_train=False, ref_df=None):
    """Complete feature engineering pipeline."""
    df = df.copy()

    # --- Step 1: Parse dates ---
    date_feature_cols = []
    for col in list(df.columns):
        if df[col].dtype == object or str(df[col].dtype) == "datetime64[ns]":
            if df[col].dtype == object:
                sample = df[col].dropna().head(50)
                if len(sample) > 0:
                    parsed_sample = pd.to_datetime(
                        sample, errors="coerce", infer_datetime_format=True
                    )
                    if parsed_sample.notna().sum() > len(sample) * 0.5:
                        df[col] = pd.to_datetime(
                            df[col], errors="coerce", infer_datetime_format=True
                        )
                        date_feature_cols.append(col)
            else:
                date_feature_cols.append(col)

    for col in date_feature_cols:
        parsed = df[col]
        df[col + "_year"] = parsed.dt.year
        df[col + "_month"] = parsed.dt.month
        df[col + "_day"] = parsed.dt.day
        df[col + "_dayofweek"] = parsed.dt.dayofweek
        df[col + "_quarter"] = parsed.dt.quarter
        df[col + "_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
        df[col + "_dayofyear"] = parsed.dt.dayofyear
        df = df.drop(columns=[col])

    # --- Step 2: Text features ---
    remaining_object = [
        c
        for c in df.columns
        if df[c].dtype == object
        or (hasattr(df[c].dtype, "name") and df[c].dtype.name == "category")
    ]

    for col in remaining_object:
        str_col = (
            df[col].astype(str).replace("nan", "__NAN__").replace("None", "__NAN__")
        )
        df[col + "_len"] = str_col.str.len()
        df[col + "_word_count"] = str_col.str.split().str.len()
        df[col + "_has_digit"] = str_col.str.contains(r"\d").astype(int)
        df[col + "_all_upper"] = (str_col == str_col.str.upper()).astype(int)
        df[col + "_n_vowels"] = str_col.apply(
            lambda x: sum(c in "aeiouAEIOU" for c in x) if x != "__NAN__" else 0
        )
        df[col + "_n_consonants"] = str_col.apply(
            lambda x: (
                sum(c.isalpha() and c not in "aeiouAEIOU" for c in x)
                if x != "__NAN__"
                else 0
            )
        )
        df[col + "_n_spaces"] = str_col.str.count(" ")
        df[col + "_n_special"] = str_col.apply(
            lambda x: (
                sum(not c.isalnum() and c != " " for c in x) if x != "__NAN__" else 0
            )
        )
        df = df.drop(columns=[col])

    # --- Step 3: Handle remaining categoricals ---
    cat_cols = [
        c
        for c in df.columns
        if df[c].dtype == object
        or (hasattr(df[c].dtype, "name") and df[c].dtype.name == "category")
    ]

    for col in cat_cols:
        df[col] = df[col].fillna("__NAN__")
        df[col] = df[col].astype("category")

    # --- Step 4: Boolean to int ---
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # --- Step 5: Numeric missing values ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col + "_isna"] = df[col].isnull().astype(int)
            if is_train:
                fill_val = df[col].median()
            else:
                fill_val = ref_df[col].median() if col in ref_df.columns else 0
            df[col] = df[col].fillna(fill_val)

    # --- Step 6: Date interaction features ---
    year_cols = [c for c in df.columns if "_year" in c]
    if len(year_cols) >= 2:
        for i in range(len(year_cols)):
            for j in range(i + 1, len(year_cols)):
                diff_col = f'{year_cols[i].replace("_year","")}_diff_{year_cols[j].replace("_year","")}'
                df[diff_col] = df[year_cols[i]] - df[year_cols[j]]

    # --- Step 7: Frequency encoding (fit on train only) ---
    if is_train:
        for col in cat_cols:
            if col in df.columns and df[col].dtype.name == "category":
                freq = df[col].value_counts(normalize=True)
                df[col + "_freq"] = df[col].map(freq)
                df[col + "_freq"] = df[col + "_freq"].fillna(0)
    else:
        for col in cat_cols:
            if col in df.columns and df[col].dtype.name == "category":
                freq_col = col + "_freq"
                if freq_col in ref_df.columns:
                    # Rebuild mapping from train_processed: get unique cats and their freqs
                    # We need to map current df values to the train frequencies
                    # Build a lookup from the train data's category->freq mapping
                    # Since ref_df has the _freq column aligned row-by-row, we can't directly
                    # extract a mapping. Instead, we store the mapping at fit time.
                    pass
                # Use the stored mapping approach: map via the same categories
                if col in ref_df.columns and ref_df[col].dtype.name == "category":
                    # Build mapping using groupby to correctly handle repeated categories
                    tmp = pd.DataFrame({"cat": ref_df[col].values, "freq": ref_df[col + "_freq"].values})
                    cat_to_freq = tmp.groupby("cat")["freq"].first().to_dict()
                    df[col + "_freq"] = df[col].map(cat_to_freq).fillna(0)
                else:
                    df[col + "_freq"] = 0.0

    return df


# Apply to train split
train_processed = full_pipeline(train_split, is_train=True, ref_df=train_split)

# Store the category-to-frequency mapping from train for use in val/test
cat_freq_mappings = {}
for col in [c for c in train_processed.columns if c.endswith("_freq")]:
    base_col = col[:-5]  # remove "_freq"
    if base_col in train_processed.columns and train_processed[base_col].dtype.name == "category":
        mapping = {}
        for cat, freq_val in zip(train_processed[base_col], train_processed[col]):
            if cat not in mapping:
                mapping[cat] = freq_val
        cat_freq_mappings[base_col] = mapping

# Apply to val and test
val_processed = full_pipeline(val_split, is_train=False, ref_df=train_processed)
test_processed = full_pipeline(test, is_train=False, ref_df=train_processed)

# Fix frequency columns using train-derived mappings (avoid any val self-leakage)
for base_col, mapping in cat_freq_mappings.items():
    if base_col in val_processed.columns and val_processed[base_col].dtype.name == "category":
        val_processed[base_col + "_freq"] = val_processed[base_col].map(mapping).fillna(0)
    if base_col in test_processed.columns and test_processed[base_col].dtype.name == "category":
        test_processed[base_col + "_freq"] = test_processed[base_col].map(mapping).fillna(0)

# Ensure column alignment
all_cols = list(train_processed.columns)
for c in all_cols:
    if c not in val_processed.columns:
        val_processed[c] = 0
    if c not in test_processed.columns:
        test_processed[c] = 0

val_processed = val_processed[all_cols]
test_processed = test_processed[all_cols]

print(f"Processed train shape: {train_processed.shape}")
print(f"Processed val shape: {val_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")

# ============================================================
# 4. TRAIN MULTI-SEED LIGHTGBM ENSEMBLE
# ============================================================
# Sanitize column names: LightGBM does not support special JSON characters in feature names
import re

def sanitize_col_name(name):
    """Replace any non-alphanumeric character (except underscore) with underscore."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

train_processed.columns = [sanitize_col_name(c) for c in train_processed.columns]
val_processed.columns = [sanitize_col_name(c) for c in val_processed.columns]
test_processed.columns = [sanitize_col_name(c) for c in test_processed.columns]

# Rebuild all_cols from sanitized names
all_cols = list(train_processed.columns)

cat_cols = train_processed.select_dtypes(include=["category"]).columns.tolist()
cat_indices = [all_cols.index(c) for c in cat_cols if c in all_cols]
print(f"Categorical columns: {cat_cols}")
print(f"Total features: {len(all_cols)}")

seeds = [42, 123, 777]
val_preds_list = []
test_preds_list = []

for seed in seeds:
    model_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
        "min_child_weight": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "verbose": -1,
        "random_state": seed,
    }

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    if neg_count > 0 and pos_count > 0:
        model_params["scale_pos_weight"] = neg_count / pos_count

    dtrain = lgb.Dataset(
        train_processed[all_cols],
        label=y_train,
        categorical_feature=cat_indices,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_processed[all_cols],
        label=y_val,
        categorical_feature=cat_indices,
        free_raw_data=False,
        reference=dtrain,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ]

    model = lgb.train(
        model_params,
        dtrain,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    best_val_auc = model.best_score["val"]["auc"]
    print(f"Seed {seed}: best_iter={best_iter}, val_auc={best_val_auc:.4f}")

    val_pred = model.predict(val_processed[all_cols], num_iteration=best_iter)
    test_pred = model.predict(test_processed[all_cols], num_iteration=best_iter)

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

# ============================================================
# 5. ENSEMBLE & VALIDATION
# ============================================================
val_pred_ensemble = np.mean(val_preds_list, axis=0)
test_pred_ensemble = np.mean(test_preds_list, axis=0)

val_auc = roc_auc_score(y_val, val_pred_ensemble)
print(f"\nEnsemble Validation ROC AUC: {val_auc:.4f}")

# ============================================================
# 6. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": range(len(test_pred_ensemble)), "Medicine status": test_pred_ensemble}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
score = val_auc
print(f"Final Validation Score: {score}")