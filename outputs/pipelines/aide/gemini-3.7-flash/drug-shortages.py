import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "availability"
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col].astype(str))

# Combine features for unified preprocessing
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0
).reset_index(drop=True)
n_train = len(train_df)

# 2. Feature Engineering & Preprocessing
features = []

# Identify column types
for col in combined_df.columns:
    # Check if datetime
    if (
        "date" in col.lower()
        or "time" in col.lower()
        or pd.api.types.is_datetime64_any_dtype(combined_df[col])
    ):
        try:
            dt = pd.to_datetime(combined_df[col], errors="coerce")
            if dt.notna().sum() > 0.3 * len(dt):
                combined_df[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
                combined_df[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
                combined_df[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
                combined_df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
                features.extend(
                    [
                        f"{col}_year",
                        f"{col}_month",
                        f"{col}_day",
                        f"{col}_dayofweek",
                    ]
                )
                continue
        except Exception:
            pass

    # Check for text vs categorical vs numerical
    if pd.api.types.is_numeric_dtype(combined_df[col]):
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        features.append(col)
    else:
        # Check text length characteristics
        sample_str = combined_df[col].astype(str).fillna("")
        mean_len = sample_str.apply(len).mean()
        unique_count = combined_df[col].nunique()

        if mean_len > 50 or unique_count > 500:
            # Text feature: TF-IDF + Truncated SVD
            tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
            tfidf_mat = tfidf.fit_transform(sample_str)
            n_components = min(16, tfidf_mat.shape[1] - 1)
            if n_components > 1:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd_mat = svd.fit_transform(tfidf_mat)
                for i in range(n_components):
                    col_name = f"{col}_svd_{i}"
                    combined_df[col_name] = svd_mat[:, i]
                    features.append(col_name)
        else:
            # Categorical feature: Label Encoding and Frequency Encoding
            combined_df[f"{col}_encoded"] = LabelEncoder().fit_transform(
                combined_df[col].astype(str)
            )
            freq = combined_df[col].value_counts(normalize=True).to_dict()
            combined_df[f"{col}_freq"] = (
                combined_df[col].map(freq).fillna(0).astype(float)
            )
            features.extend([f"{col}_encoded", f"{col}_freq"])

X_train = combined_df.iloc[:n_train][features].reset_index(drop=True)
X_test = combined_df.iloc[n_train:][features].reset_index(drop=True)

# 3. Stratified K-Fold Cross Validation with LightGBM
n_classes = len(target_encoder.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((n_train, n_classes))
test_preds = np.zeros((len(test_df), n_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 4. Evaluation
oof_pred_labels = np.argmax(oof_preds, axis=1)
oof_accuracy = accuracy_score(y_train, oof_pred_labels)
print(f"5-Fold CV Accuracy: {oof_accuracy:.5f}")

# 5. Save Submission
test_pred_labels = np.argmax(test_preds, axis=1)
predicted_classes = target_encoder.inverse_transform(test_pred_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "availability": predicted_classes}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
