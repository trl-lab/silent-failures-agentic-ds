import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor

os.makedirs("./working", exist_ok=True)

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Contract Award Amount (USD)"
y_raw = train_df[target_col].values.astype(np.float64)

# Target log1p transform
use_log = np.max(y_raw) > 50.0
if use_log:
    y = np.log1p(np.maximum(y_raw, 0))
else:
    y = y_raw

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

n_train = len(train_features)
df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# 2. Extract Datetime features
date_cols = [c for c in df_all.columns if "date" in c.lower() or "time" in c.lower()]

for c in date_cols:
    parsed = pd.to_datetime(df_all[c], errors="coerce")
    df_all[f"{c}_year"] = parsed.dt.year.fillna(-1).astype(int)
    df_all[f"{c}_month"] = parsed.dt.month.fillna(-1).astype(int)
    df_all[f"{c}_day"] = parsed.dt.day.fillna(-1).astype(int)
    df_all[f"{c}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
    df_all[f"{c}_dayofyear"] = parsed.dt.dayofyear.fillna(-1).astype(int)
    df_all[f"{c}_epoch"] = (parsed.astype("int64") // 10**9).fillna(-1).astype(int)
    df_all.drop(columns=[c], inplace=True)

# 3. Categorical and Text features
object_cols = df_all.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

cat_cols = []
text_cols = []
for c in object_cols:
    str_series = df_all[c].fillna("").astype(str)
    avg_words = str_series.apply(lambda x: len(x.split())).mean()
    if avg_words > 3:
        text_cols.append(c)
    else:
        cat_cols.append(c)

# Frequency encoding for categoricals
for c in cat_cols:
    str_series = df_all[c].fillna("missing").astype(str)
    freq = str_series.value_counts(normalize=True).to_dict()
    df_all[f"{c}_freq"] = str_series.map(freq).fillna(0).astype(np.float32)
    df_all[c] = str_series.astype(str)

# Text feature engineering (metadata + TF-IDF with expanded SVD)
for c in text_cols:
    str_series = df_all[c].fillna("").astype(str)
    df_all[f"{c}_char_len"] = str_series.apply(len).astype(np.float32)
    df_all[f"{c}_word_cnt"] = str_series.apply(lambda x: len(x.split())).astype(
        np.float32
    )

    texts = str_series.values
    tfidf = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_mat = tfidf.fit_transform(texts)

    n_comp = min(24, max(2, tfidf_mat.shape[1] - 1))
    if n_comp > 1:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_comp):
            df_all[f"{c}_svd_{i}"] = svd_mat[:, i].astype(np.float32)
    df_all.drop(columns=[c], inplace=True)

# Missing count feature
df_all["missing_count"] = df_all.isnull().sum(axis=1)

# Split back into train and test
X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

cat_features = [c for c in cat_cols if c in X_train.columns]

# 4. Cross-Validation and Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train, dtype=np.float64)
test_preds = np.zeros(len(X_test), dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        cat_features=cat_features,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# 5. Compute Validation Metric
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation Metric (RMSE): {cv_rmse:.4f}")

# 6. Generate Submission
if use_log:
    final_test_preds = np.expm1(np.maximum(test_preds, 0))
else:
    final_test_preds = np.maximum(test_preds, 0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds)),
        "Contract Award Amount (USD)": final_test_preds,
    }
)

submission.to_csv("./working/submission.csv", index=False)
