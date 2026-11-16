import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

n_train = len(train_df)
n_test = len(test_df)


# 2. Preprocess text, hierarchical categories, and interaction features
def extract_subcategories(series):
    def split_cat(text):
        try:
            cats = str(text).split("/")
            return cats[0], cats[1], "/".join(cats[2:])
        except Exception:
            return "missing", "missing", "missing"

    cats_split = [split_cat(val) for val in series]
    return zip(*cats_split)


def preprocess_df(df):
    df["name"] = df["name"].fillna("missing").astype(str)
    df["item_description"] = df["item_description"].fillna("missing").astype(str)
    df["brand_name"] = df["brand_name"].fillna("missing").astype(str)
    df["category_name"] = df["category_name"].fillna("missing").astype(str)

    # Subcategory decomposition
    c1, c2, c3 = extract_subcategories(df["category_name"])
    df["subcat_0"] = list(c1)
    df["subcat_1"] = list(c2)
    df["subcat_2"] = list(c3)

    # Contextual enriched name (brand + title)
    df["name_with_brand"] = df["brand_name"] + " " + df["name"]

    # Explicit cross-feature interactions
    df["brand_cat"] = df["brand_name"] + " __ " + df["category_name"]
    df["brand_subcat2"] = df["brand_name"] + " __ " + df["subcat_2"]
    df["cond_shipping"] = (
        df["item_condition_id"].astype(str) + "_" + df["shipping"].astype(str)
    )

    df["item_condition_id"] = df["item_condition_id"].fillna(1).astype(int)
    df["shipping"] = df["shipping"].fillna(0).astype(int)

    # Metadata indicator features
    df["brand_missing"] = (df["brand_name"] == "missing").astype(int)
    df["desc_missing"] = (
        (df["item_description"] == "missing")
        | (df["item_description"] == "No description yet")
    ).astype(int)
    df["desc_len"] = df["item_description"].str.len()
    df["name_len"] = df["name"].str.len()

    return df


full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
full_df = preprocess_df(full_df)

train_df = full_df.iloc[:n_train].copy()
test_df = full_df.iloc[n_train:].copy()

y = np.log1p(train_df["price"].values)

# 3. Vectorization of multi-modal features and interactions
print("Vectorizing enriched name word and char n-grams...")
name_word_vec = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 3),
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)
X_name_word = name_word_vec.fit_transform(full_df["name_with_brand"])

name_char_vec = TfidfVectorizer(
    max_features=60000,
    ngram_range=(2, 4),
    analyzer="char_wb",
    min_df=3,
    sublinear_tf=True,
)
X_name_char = name_char_vec.fit_transform(full_df["name"])

print("Vectorizing description text...")
desc_vec = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 2),
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)
X_desc = desc_vec.fit_transform(full_df["item_description"])

print("Vectorizing categorical hierarchies, brands, and interactions...")
brand_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_brand = brand_vec.fit_transform(full_df["brand_name"])

cat_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_cat = cat_vec.fit_transform(full_df["category_name"])

subcat0_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_subcat0 = subcat0_vec.fit_transform(full_df["subcat_0"])

subcat1_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_subcat1 = subcat1_vec.fit_transform(full_df["subcat_1"])

subcat2_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_subcat2 = subcat2_vec.fit_transform(full_df["subcat_2"])

# Interaction count vectorizers
brand_cat_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_brand_cat = brand_cat_vec.fit_transform(full_df["brand_cat"])

brand_subcat2_vec = CountVectorizer(token_pattern=r".+", min_df=2)
X_brand_subcat2 = brand_subcat2_vec.fit_transform(full_df["brand_subcat2"])

# One-hot encode condition, shipping, and cond_shipping interactions
ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe = ohe.fit_transform(full_df[["item_condition_id", "shipping", "cond_shipping"]])

# Numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(
    full_df[["brand_missing", "desc_missing", "desc_len", "name_len"]].values
)
X_num_sparse = csr_matrix(X_num)

# Combine all sparse feature representations
X_full = hstack(
    [
        X_name_word,
        X_name_char,
        X_desc,
        X_brand,
        X_cat,
        X_subcat0,
        X_subcat1,
        X_subcat2,
        X_brand_cat,
        X_brand_subcat2,
        X_ohe,
        X_num_sparse,
    ],
    format="csr",
)

X_train = X_full[:n_train]
X_test = X_full[n_train:]

# 4. 5-Fold Cross Validation with Ensemble of Ridge Regressors
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(n_test)

alphas = [3.0, 5.5, 9.0]
weights = [0.45, 0.35, 0.20]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr_f, y_tr_f = X_train[train_idx], y[train_idx]
    X_va_f, y_va_f = X_train[val_idx], y[val_idx]

    fold_val_preds = np.zeros(len(val_idx))
    fold_test_preds = np.zeros(n_test)

    for alpha, w in zip(alphas, weights):
        model = Ridge(
            alpha=alpha,
            solver="auto",
            fit_intercept=True,
            copy_X=False,
            random_state=42,
        )
        model.fit(X_tr_f, y_tr_f)

        pred_val = model.predict(X_va_f)
        pred_test = model.predict(X_test)

        fold_val_preds += w * pred_val
        fold_test_preds += w * pred_test

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / 5

# 5. Evaluate and output predictions
oof_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall OOF RMSE (log1p scale): {oof_rmse:.4f}")

os.makedirs("./working", exist_ok=True)
final_test_price = np.expm1(test_preds)
final_test_price = np.clip(final_test_price, 0.0, None)

submission = pd.DataFrame(
    {"id": np.arange(len(final_test_price)), "price": final_test_price}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
