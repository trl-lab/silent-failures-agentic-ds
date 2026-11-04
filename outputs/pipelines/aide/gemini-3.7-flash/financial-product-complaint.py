import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "Company response to consumer"
assert target_col in train_df.columns, f"Target column '{target_col}' not in train data"

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col].astype(str))
num_classes = len(le_target.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {le_target.classes_}")

# Combine datasets for consistent feature processing
n_train = len(train_df)
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], axis=0
).reset_index(drop=True)

# Drop high-cardinality identifier columns if present
id_cols = [c for c in ["Complaint ID", "id", "ID"] if c in df.columns]
df_features = df.drop(columns=id_cols, errors="ignore").copy()

# 2. Date features
date_cols = [c for c in df_features.columns if "date" in c.lower()]
for col in date_cols:
    df_features[col] = pd.to_datetime(df_features[col], errors="coerce")
    df_features[f"{col}_year"] = df_features[col].dt.year.fillna(-1).astype(int)
    df_features[f"{col}_month"] = df_features[col].dt.month.fillna(-1).astype(int)
    df_features[f"{col}_day"] = df_features[col].dt.day.fillna(-1).astype(int)
    df_features[f"{col}_dayofweek"] = (
        df_features[col].dt.dayofweek.fillna(-1).astype(int)
    )

if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            diff_col_name = f"diff_days_{col1}_{col2}"
            df_features[diff_col_name] = (
                df_features[col1] - df_features[col2]
            ).dt.days.fillna(-999)

df_features.drop(columns=date_cols, inplace=True)

# 3. Separate text and categorical/numerical columns
text_cols = [
    c
    for c in df_features.columns
    if any(
        keyword in c.lower()
        for keyword in [
            "narrative",
            "complaint",
            "description",
            "issue",
            "response",
            "public",
        ]
    )
]
cat_cols = [
    c
    for c in df_features.columns
    if df_features[c].dtype == "object" and c not in text_cols
]

# Build text features via TF-IDF + SVD
svd_dfs = []
for col in text_cols:
    text_data = df_features[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_mat = tfidf.fit_transform(text_data)

    n_components = min(30, max(2, tfidf_mat.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    svd_feats = svd.fit_transform(tfidf_mat)

    svd_df = pd.DataFrame(
        svd_feats, columns=[f"svd_{col}_{i}" for i in range(n_components)]
    )
    svd_dfs.append(svd_df)

if svd_dfs:
    svd_all = pd.concat(svd_dfs, axis=1)
    df_features = pd.concat([df_features.drop(columns=text_cols), svd_all], axis=1)

# Categorical column encoding & Frequency encoding
for col in cat_cols:
    df_features[col] = df_features[col].fillna("Missing").astype(str)
    # Frequency encoding
    freq = df_features[col].value_counts(normalize=True)
    df_features[f"{col}_freq"] = df_features[col].map(freq).astype(float)
    # Categorical type for LightGBM
    df_features[col] = df_features[col].astype("category")

# Split train and test
X_train = df_features.iloc[:n_train].reset_index(drop=True)
X_test = df_features.iloc[n_train:].reset_index(drop=True)

# Sanitize column names for LightGBM
clean_cols = [re.sub(r"[^\w_]", "_", str(c)) for c in X_train.columns]
X_train.columns = clean_cols
X_test.columns = clean_cols

# 4. Stratified K-Fold CV & LightGBM Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
    "n_estimators": 800,
    "verbose": -1,
    "n_jobs": -1,
}

print("Starting 5-fold cross validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test) / skf.n_splits

    fold_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.5f}")

# 5. Out-of-fold evaluation
oof_pred_labels = np.argmax(oof_preds, axis=1)
overall_accuracy = accuracy_score(y, oof_pred_labels)
print(f"Overall OOF Accuracy: {overall_accuracy:.5f}")

# 6. Generate Submission
final_test_labels = le_target.inverse_transform(np.argmax(test_preds, axis=1))
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), target_col: final_test_labels}
)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH} with shape {submission.shape}")
