import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

n_train = len(train_df)
n_test = len(test_df)

# Combine datasets for consistent vectorization
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Preprocess missing values
full_df["name"] = full_df["name"].fillna("").astype(str)
full_df["item_description"] = full_df["item_description"].fillna("").astype(str)
full_df["item_description"] = full_df["item_description"].replace(
    "No description yet", ""
)
full_df["brand_name"] = full_df["brand_name"].fillna("missing").astype(str)
full_df["category_name"] = full_df["category_name"].fillna("missing/missing/missing")

# Extract category hierarchy
cats = full_df["category_name"].astype(str).str.split("/", expand=True)
full_df["cat_0"] = cats[0].fillna("missing")
full_df["cat_1"] = cats[1].fillna("missing") if 1 in cats.columns else "missing"
full_df["cat_2"] = cats[2].fillna("missing") if 2 in cats.columns else "missing"

# Numerical text metadata features
full_df["name_len"] = full_df["name"].str.len()
full_df["desc_len"] = full_df["item_description"].str.len()
full_df["name_words"] = full_df["name"].str.split().str.len()
full_df["desc_words"] = full_df["item_description"].str.split().str.len()
full_df["has_brand"] = (full_df["brand_name"] != "missing").astype(int)
full_df["has_desc"] = (full_df["item_description"] != "").astype(int)

# TF-IDF Feature Extraction
vec_name_word = TfidfVectorizer(
    ngram_range=(1, 2), max_features=35000, token_pattern=r"\w+"
)
X_name_word = vec_name_word.fit_transform(full_df["name"])

vec_name_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), max_features=15000)
X_name_char = vec_name_char.fit_transform(full_df["name"])

vec_desc = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    stop_words="english",
    token_pattern=r"\w+",
)
X_desc = vec_desc.fit_transform(full_df["item_description"])

# Count Vectorization for Categoricals
vec_brand = CountVectorizer(min_df=2, token_pattern=r"(?u)\b\w+\b")
X_brand = vec_brand.fit_transform(full_df["brand_name"])

vec_cat0 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&/-]+\b")
X_cat0 = vec_cat0.fit_transform(full_df["cat_0"])

vec_cat1 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&/-]+\b")
X_cat1 = vec_cat1.fit_transform(full_df["cat_1"])

vec_cat2 = CountVectorizer(token_pattern=r"(?u)\b[\w\s&/-]+\b")
X_cat2 = vec_cat2.fit_transform(full_df["cat_2"])

# Categorical One-Hot Encoding
ohe = OneHotEncoder(handle_unknown="ignore")
X_cond_ship = ohe.fit_transform(full_df[["item_condition_id", "shipping"]])

# Scale Numerical Features
num_cols = [
    "name_len",
    "desc_len",
    "name_words",
    "desc_words",
    "has_brand",
    "has_desc",
]
scaler = StandardScaler()
X_num = scaler.fit_transform(full_df[num_cols])

# Concatenate Sparse Matrices
X_full = hstack(
    [
        X_name_word,
        X_name_char,
        X_desc,
        X_brand,
        X_cat0,
        X_cat1,
        X_cat2,
        X_cond_ship,
        X_num,
    ]
).tocsr()

X_train = X_full[:n_train]
X_test = X_full[n_train:]

# Log-transformed target values
y_train = np.log1p(np.maximum(0, train_df["price"].values))

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    model = Ridge(alpha=1.5, solver="auto", random_state=42)
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"5-Fold Cross-Validation RMSE (log scale): {cv_rmse:.5f}")

# Transform predictions back to raw price scale
final_test_prices = np.expm1(np.maximum(0, test_preds))

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(n_test), "price": final_test_prices})
sub.to_csv("./working/submission.csv", index=False)
