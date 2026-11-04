import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Rating"
y_raw = train_df[target_col]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Separate features
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Combine for uniform preprocessing
n_train = len(X_train)
df_all = pd.concat([X_train, X_test], ignore_index=True)


# Preprocessing helper
def clean_and_engineer_features(df):
    processed = pd.DataFrame(index=df.index)
    text_cols = []

    for col in df.columns:
        # Check if percentage string
        if df[col].dtype == "object":
            sample_val = (
                str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            )
            if "%" in sample_val:
                processed[f"{col}_num"] = (
                    df[col]
                    .astype(str)
                    .str.rstrip("%")
                    .replace("None", np.nan)
                    .astype(float)
                )
                continue

            # Check length/diversity to determine if text or categorical
            unique_ratio = df[col].nunique() / len(df)
            avg_len = (
                df[col].dropna().astype(str).map(len).mean()
                if not df[col].dropna().empty
                else 0
            )
            if avg_len > 15 or unique_ratio > 0.4:
                text_cols.append(col)
            else:
                # Categorical column - frequency encode and category code
                processed[f"{col}_cat"] = df[col].astype("category").cat.codes
                freq = df[col].value_counts(normalize=True)
                processed[f"{col}_freq"] = df[col].map(freq).fillna(0)
        else:
            processed[col] = df[col]

    return processed, text_cols


processed_features, text_cols = clean_and_engineer_features(df_all)

# Process text columns with TF-IDF
tfidf_features = []
for col in text_cols:
    text_series = df_all[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=50, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_mat = tfidf.fit_transform(text_series).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_mat,
        columns=[f"tfidf_{col}_{i}" for i in range(tfidf_mat.shape[1])],
        index=df_all.index,
    )
    tfidf_features.append(tfidf_df)

if tfidf_features:
    all_features = pd.concat([processed_features] + tfidf_features, axis=1)
else:
    all_features = processed_features

# Ensure all column names are clean strings
all_features.columns = [
    str(c).replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    for c in all_features.columns
]

X_train_proc = all_features.iloc[:n_train].reset_index(drop=True)
X_test_proc = all_features.iloc[n_train:].reset_index(drop=True)

# 5-fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(label_encoder.classes_)

oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test_proc), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 500,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_probs = model.predict_proba(X_va)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test_proc) / skf.n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# Final Test Predictions
final_test_labels = label_encoder.inverse_transform(np.argmax(test_preds, axis=1))

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "Rating": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
