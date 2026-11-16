import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_CATBOOST = False

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Meta Critic"
y = train_df[target_col].values

# Handle test id
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# Drop target from train for features
train_features = train_df.drop(columns=[target_col], errors="ignore")

# Combine train and test for uniform feature extraction
n_train = len(train_features)
combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)

# Distinguish column types
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

text_cols = []
short_cat_cols = []
for col in cat_cols:
    avg_len = combined[col].astype(str).apply(len).mean()
    if avg_len > 40:
        text_cols.append(col)
    else:
        short_cat_cols.append(col)

# 2. Feature Engineering
df_feat = pd.DataFrame(index=combined.index)

# Numerical features & transformations
for col in num_cols:
    vals = combined[col].astype(float)
    df_feat[col] = vals.fillna(vals.median())
    if (vals.dropna() >= 0).all():
        if abs(vals.skew()) > 0.75:
            df_feat[f"{col}_log1p"] = np.log1p(df_feat[col])

# Pairwise numeric interactions
if len(num_cols) >= 2:
    for i in range(min(len(num_cols), 5)):
        for j in range(i + 1, min(len(num_cols), 5)):
            c1, c2 = num_cols[i], num_cols[j]
            df_feat[f"{c1}_x_{c2}"] = df_feat[c1] * df_feat[c2]
            df_feat[f"{c1}_div_{c2}"] = df_feat[c1] / (df_feat[c2].abs() + 1e-4)

# Categorical features
for col in short_cat_cols:
    s = combined[col].astype(str).fillna("missing")
    freqs = s.value_counts(normalize=True)
    df_feat[f"{col}_freq"] = s.map(freqs).fillna(0)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_feat[f"{col}_ord"] = encoder.fit_transform(s.values.reshape(-1, 1))

# Text features via TF-IDF + SVD
for col in text_cols:
    text_series = combined[col].astype(str).fillna("")
    df_feat[f"{col}_char_len"] = text_series.apply(len)
    df_feat[f"{col}_word_count"] = text_series.apply(lambda x: len(x.split()))

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(text_series)
    n_comp = min(12, tfidf_mat.shape[1] - 1)
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for comp in range(n_comp):
            df_feat[f"{col}_svd_{comp}"] = svd_mat[:, comp]

X_all = df_feat.values
X_train = X_all[:n_train]
X_test = X_all[n_train:]

# 3. Model Training & 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]

    if HAS_CATBOOST:
        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=0,
        )
        model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.03, random_state=42 + fold
        )
        model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

rmse = root_mean_squared_error(y, oof_preds)
print(f"OOF RMSE: {rmse:.4f}")

# 4. Save Submission File
os.makedirs("./working", exist_ok=True)
sub_df = pd.DataFrame({"id": test_ids, "Meta Critic": test_preds})
sub_df.to_csv("./working/submission.csv", index=False)
