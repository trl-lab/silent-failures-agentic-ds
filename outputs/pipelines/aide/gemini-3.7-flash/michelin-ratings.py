import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "awardFor"
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(label_encoder.classes_)

# Combine for unified preprocessing
train_len = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], axis=0
).reset_index(drop=True)

# 2. Feature Engineering
# Price features
for col in ["minPrice", "maxPrice"]:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

if "minPrice" in combined_df.columns and "maxPrice" in combined_df.columns:
    combined_df["price_mean"] = (
        combined_df["minPrice"] + combined_df["maxPrice"]
    ) / 2.0
    combined_df["price_diff"] = combined_df["maxPrice"] - combined_df["minPrice"]
    combined_df["price_ratio"] = (combined_df["maxPrice"] + 1.0) / (
        combined_df["minPrice"] + 1.0
    )
    combined_df["log_minPrice"] = np.log1p(combined_df["minPrice"].clip(lower=0))
    combined_df["log_maxPrice"] = np.log1p(combined_df["maxPrice"].clip(lower=0))

# Geographic features
if "latitude" in combined_df.columns and "longitude" in combined_df.columns:
    combined_df["latitude"] = pd.to_numeric(combined_df["latitude"], errors="coerce")
    combined_df["longitude"] = pd.to_numeric(combined_df["longitude"], errors="coerce")

    lat_median = combined_df["latitude"].median()
    lon_median = combined_df["longitude"].median()
    combined_df["lat_filled"] = combined_df["latitude"].fillna(lat_median)
    combined_df["lon_filled"] = combined_df["longitude"].fillna(lon_median)

    # Distance to major culinary capitals
    capitals = {
        "paris": (48.8566, 2.3522),
        "tokyo": (35.6762, 139.6503),
        "nyc": (40.7128, -74.0060),
        "london": (51.5074, -0.1278),
    }
    for cap_name, (clat, clon) in capitals.items():
        combined_df[f"dist_{cap_name}"] = np.sqrt(
            (combined_df["lat_filled"] - clat) ** 2
            + (combined_df["lon_filled"] - clon) ** 2
        )

    # Geo spatial clustering
    coords = combined_df[["lat_filled", "lon_filled"]].values
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=5)
    combined_df["geo_cluster"] = kmeans.fit_predict(coords)
    combined_df.drop(columns=["lat_filled", "lon_filled"], inplace=True)

# Text and Categorical Feature Engineering
text_cols = [
    col
    for col in [
        "description",
        "cuisine",
        "facilitiesServices",
        "name",
        "address",
        "location",
    ]
    if col in combined_df.columns
]
tfidf_svd_dfs = []

for t_col in text_cols:
    combined_df[f"{t_col}_len"] = combined_df[t_col].astype(str).str.len()
    combined_df[f"{t_col}_word_count"] = (
        combined_df[t_col].astype(str).apply(lambda s: len(s.split()))
    )

    # TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    text_data = combined_df[t_col].fillna("").astype(str)
    tfidf_mat = tfidf.fit_transform(text_data)

    n_comp = 16 if t_col in ["description", "cuisine"] else 8
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)

    svd_df = pd.DataFrame(
        svd_feats, columns=[f"{t_col}_svd_{i}" for i in range(n_comp)]
    )
    tfidf_svd_dfs.append(svd_df)

# Frequency encoding for remaining object/categorical columns
cat_cols = [
    col
    for col in combined_df.select_dtypes(include=["object", "category"]).columns
    if col not in ["id"]
]
for c in cat_cols:
    freq = combined_df[c].value_counts(normalize=True).to_dict()
    combined_df[f"{c}_freq"] = combined_df[c].map(freq).fillna(0).astype(float)
    combined_df[c] = combined_df[c].astype(str).astype("category")

# Merge SVD features
all_features = pd.concat([combined_df] + tfidf_svd_dfs, axis=1)

# Drop any non-numeric / non-categorical columns that can't be handled
drop_cols = ["id"] + [
    col for col in ["url", "websiteUrl", "phoneNumber"] if col in all_features.columns
]
all_features.drop(
    columns=[c for c in drop_cols if c in all_features.columns], inplace=True
)

X_train = all_features.iloc[:train_len].reset_index(drop=True)
X_test = all_features.iloc[train_len:].reset_index(drop=True)

cat_feature_indices = [
    i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == "category"
]

# 3. Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((train_len, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.06,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42 + fold,
        cat_features=cat_feature_indices,
        verbose=0,
        task_type="CPU",
    )

    model.fit(
        X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False
    )

    val_probs = model.predict_proba(X_val)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test) / n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_train, oof_pred_labels)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# 4. Submission
final_test_labels = label_encoder.inverse_transform(np.argmax(test_preds, axis=1))
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "awardFor": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
