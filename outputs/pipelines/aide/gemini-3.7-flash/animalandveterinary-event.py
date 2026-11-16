import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "serious_ae"

target_map = {
    "true": 1,
    "1": 1,
    "1.0": 1,
    "t": 1,
    "yes": 1,
    "false": 0,
    "0": 0,
    "0.0": 0,
    "f": 0,
    "no": 0,
}
y_raw = train_df[target_col].astype(str).str.strip().str.lower()
y = y_raw.map(target_map).fillna(0).astype(int).values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Advanced Feature Engineering
date_cols = []
text_corpus = pd.Series([""] * len(df_all), index=df_all.index)

for col in list(df_all.columns):
    # Detect complex / nested objects and extract structural stats
    sample_val = (
        df_all[col].dropna().iloc[0] if df_all[col].dropna().shape[0] > 0 else None
    )
    if isinstance(sample_val, (list, dict, tuple, np.ndarray)):
        df_all[f"{col}_len"] = df_all[col].apply(
            lambda x: len(x) if isinstance(x, (list, dict, tuple, np.ndarray)) else 0
        )
        df_all[col] = df_all[col].apply(lambda x: str(x) if x is not None else "")

    # Date parsing and temporal feature extraction
    if df_all[col].dtype == "object":
        is_date = False
        try:
            converted = pd.to_datetime(df_all[col], errors="raise", format="mixed")
            df_all[f"{col}_year"] = converted.dt.year
            df_all[f"{col}_month"] = converted.dt.month
            df_all[f"{col}_day"] = converted.dt.day
            df_all[f"{col}_dayofweek"] = converted.dt.dayofweek
            df_all[f"{col}_timestamp"] = converted.astype("int64") // 10**9
            date_cols.append(f"{col}_timestamp")
            df_all.drop(columns=[col], inplace=True)
            is_date = True
        except Exception:
            pass

        if is_date:
            continue

    # Categorical and string columns
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        str_series = df_all[col].astype(str).fillna("")
        df_all[f"{col}_char_len"] = str_series.str.len()
        df_all[f"{col}_word_cnt"] = str_series.apply(
            lambda s: len(s.split()) if s else 0
        )

        # Append to unified corpus for global TF-IDF representation
        text_corpus = (
            text_corpus + " " + str_series.str.replace(r"[^\w\s]", " ", regex=True)
        )

        # Frequency encoding
        freq = df_all[col].value_counts(normalize=True)
        df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(float)

        # Label encoding
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col].astype(str).fillna("__missing__"))
        df_all[col] = df_all[col].astype("category")

# Pairwise date intervals
if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col_a, col_b = date_cols[i], date_cols[j]
            diff_name = f"diff_{col_a}_{col_b}"
            df_all[diff_name] = (df_all[col_a] - df_all[col_b]).astype(float)

# TF-IDF + TruncatedSVD for textual clinical and reaction signals
tfidf = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
tfidf_mat = tfidf.fit_transform(text_corpus)

n_svd_components = 30
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
svd_features = svd.fit_transform(tfidf_mat)

for i in range(n_svd_components):
    df_all[f"svd_text_{i}"] = svd_features[:, i]

# Separate back to train and test sets
X = df_all.iloc[:n_train].reset_index(drop=True)
X_test = df_all.iloc[n_train:].reset_index(drop=True)

# 3. Model Training with 5-Fold Stratified CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    fold_auc = roc_auc_score(y_val_fold, val_pred)
    print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC AUC: {overall_auc:.5f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "serious_ae": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
