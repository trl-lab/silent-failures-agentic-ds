import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "dea_schedule"
features = [c for c in train_df.columns if c != target_col]

# Encode target
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
num_classes = len(le.classes_)

train_len = len(train_df)
combined_df = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Clean and normalize string/object representations
for col in features:
    if combined_df[col].dtype == "object":
        combined_df[col] = combined_df[col].astype(str).fillna("missing")

numeric_cols = []
cat_cols = []
text_cols = []

for col in features:
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        numeric_cols.append(col)
    else:
        avg_len = combined_df[col].apply(lambda x: len(x.split())).mean()
        unique_count = combined_df[col].nunique()
        if avg_len > 1.5 or unique_count > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Create a combined text column across all text and categorical features
combined_df["all_text_combined"] = combined_df[text_cols + cat_cols].apply(
    lambda row: " ".join(row.values.astype(str)), axis=1
)
text_processing_cols = list(text_cols) + ["all_text_combined"]

svd_dfs = []

# 1. Word-level TF-IDF + SVD
for col in text_processing_cols:
    tfidf_word = TfidfVectorizer(
        max_features=10000,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    mat_word = tfidf_word.fit_transform(combined_df[col])
    n_comp = min(
        30 if col == "all_text_combined" else 20, max(2, mat_word.shape[1] - 1)
    )
    if mat_word.shape[1] > 1:
        svd_word = TruncatedSVD(n_components=n_comp, random_state=42)
        feats_word = svd_word.fit_transform(mat_word)
        cols = [f"{col}_word_svd_{i}" for i in range(n_comp)]
        svd_dfs.append(pd.DataFrame(feats_word, columns=cols))

# 2. Character-level TF-IDF + SVD (captures chemical prefixes, stems, and suffixes)
for col in text_processing_cols:
    tfidf_char = TfidfVectorizer(
        max_features=10000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
    )
    mat_char = tfidf_char.fit_transform(combined_df[col])
    n_comp = min(
        30 if col == "all_text_combined" else 20, max(2, mat_char.shape[1] - 1)
    )
    if mat_char.shape[1] > 1:
        svd_char = TruncatedSVD(n_components=n_comp, random_state=42)
        feats_char = svd_char.fit_transform(mat_char)
        cols = [f"{col}_char_svd_{i}" for i in range(n_comp)]
        svd_dfs.append(pd.DataFrame(feats_char, columns=cols))

# Frequency encoding and categorical type conversion for categorical features
for col in cat_cols:
    freq = combined_df[col].value_counts()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq)
    combined_df[col] = combined_df[col].astype("category")

cols_to_keep = numeric_cols + cat_cols + [f"{c}_freq" for c in cat_cols]
processed_df = combined_df[cols_to_keep].copy()

if svd_dfs:
    processed_df = pd.concat([processed_df] + svd_dfs, axis=1)

X_train = processed_df.iloc[:train_len].reset_index(drop=True)
X_test = processed_df.iloc[train_len:].reset_index(drop=True)

# 5-Fold Stratified Cross-Validation with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((train_len, num_classes))
test_preds = np.zeros((len(test_df), num_classes))

seeds = [42, 2024]

for seed in seeds:
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": -1,
        "min_child_samples": 8,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": seed,
        "n_jobs": -1,
    }

    seed_oof = np.zeros((train_len, num_classes))
    seed_test = np.zeros((len(test_df), num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=1000,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        seed_oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        seed_test += (
            model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
        )

    oof_preds += seed_oof / len(seeds)
    test_preds += seed_test / len(seeds)

oof_class_preds = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_class_preds)
print(f"5-Fold CV Out-Of-Fold Accuracy: {cv_accuracy:.5f}")

# Generate and save test predictions
test_class_preds = np.argmax(test_preds, axis=1)
test_labels = le.inverse_transform(test_class_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "dea_schedule": test_labels})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
