import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Process target variable
target_col = "stars"
unique_targets = np.sort(train_df[target_col].unique())
target_map = {val: idx for idx, val in enumerate(unique_targets)}
inv_target_map = {idx: val for idx, val in enumerate(unique_targets)}
y_train = train_df[target_col].map(target_map).values

n_train = len(train_df)
n_test = len(test_df)
full_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# 2. Text Feature Construction
text_cols = ["name", "categories", "city", "state", "attributes"]
text_series = pd.Series([""] * len(full_df))
for col in text_cols:
    if col in full_df.columns:
        text_series = text_series + " " + full_df[col].astype(str).fillna("")

tfidf = TfidfVectorizer(
    max_features=12000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
)
tfidf_matrix = tfidf.fit_transform(text_series)

svd = TruncatedSVD(n_components=50, random_state=42)
svd_feats = svd.fit_transform(tfidf_matrix)

# 3. Tabular & Spatial Feature Engineering
tabular_feats = []

# Numeric features
num_cols = ["review_count", "latitude", "longitude", "is_open"]
for col in num_cols:
    if col in full_df.columns:
        val = (
            pd.to_numeric(full_df[col], errors="coerce")
            .fillna(full_df[col].median() if not full_df[col].isnull().all() else 0)
            .values
        )
        tabular_feats.append(val.reshape(-1, 1))

# Spatial aggregations
if (
    "latitude" in full_df.columns
    and "longitude" in full_df.columns
    and "city" in full_df.columns
):
    lat = pd.to_numeric(full_df["latitude"], errors="coerce")
    lon = pd.to_numeric(full_df["longitude"], errors="coerce")
    city_lat_mean = full_df.groupby("city")["latitude"].transform("mean")
    city_lon_mean = full_df.groupby("city")["longitude"].transform("mean")
    dist_center = (
        np.sqrt((lat - city_lat_mean) ** 2 + (lon - city_lon_mean) ** 2)
        .fillna(0)
        .values
    )
    tabular_feats.append(dist_center.reshape(-1, 1))

# Frequency Encodings
cat_cols = ["city", "state", "postal_code"]
for col in cat_cols:
    if col in full_df.columns:
        freq = full_df[col].value_counts(normalize=True).to_dict()
        col_freq = full_df[col].map(freq).fillna(0).values
        tabular_feats.append(col_freq.reshape(-1, 1))

tabular_matrix = (
    np.hstack(tabular_feats) if tabular_feats else np.zeros((len(full_df), 1))
)

# 4. Stratified K-Fold Split & Meta Feature Generation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tfidf_train = tfidf_matrix[:n_train]
tfidf_test = tfidf_matrix[n_train:]

meta_oof = np.zeros((n_train, len(unique_targets)))
meta_test = np.zeros((n_test, len(unique_targets)))

for train_idx, val_idx in skf.split(tfidf_train, y_train):
    X_tr_tfidf, y_tr = tfidf_train[train_idx], y_train[train_idx]
    X_va_tfidf = tfidf_train[val_idx]

    clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    clf.fit(X_tr_tfidf, y_tr)

    meta_oof[val_idx] = clf.predict_proba(X_va_tfidf)
    meta_test += clf.predict_proba(tfidf_test) / skf.n_splits

# Combine all engineered features
all_features = np.hstack([tabular_matrix, svd_feats, np.vstack([meta_oof, meta_test])])

X_train = all_features[:n_train]
X_test = all_features[n_train:]

# 5. XGBoost Model Training with Cross-Validation
oof_preds = np.zeros((n_train, len(unique_targets)))
test_preds = np.zeros((n_test, len(unique_targets)))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": len(unique_targets),
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 500,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_va)
    test_preds += model.predict_proba(X_test) / skf.n_splits

# Compute Out-of-Fold Accuracy
oof_pred_labels = np.argmax(oof_preds, axis=1)
acc = accuracy_score(y_train, oof_pred_labels)
print(f"5-Fold CV Accuracy: {acc:.5f}")

# 6. Generate Submission File
final_test_indices = np.argmax(test_preds, axis=1)
final_test_stars = [inv_target_map[idx] for idx in final_test_indices]

submission = pd.DataFrame({"id": np.arange(n_test), "stars": final_test_stars})
submission.to_csv("./working/submission.csv", index=False)
