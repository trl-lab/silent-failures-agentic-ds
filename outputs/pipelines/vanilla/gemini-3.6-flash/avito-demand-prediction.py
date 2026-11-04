import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y_train = train["deal_probability"].values
y_train = np.clip(y_train, 0.0, 1.0)

len_train = len(train)
df = pd.concat([train.drop(columns=["deal_probability"], errors="ignore"), test], axis=0, ignore_index=True)

title_text = df["title"].fillna("").astype(str)
desc_text = df["description"].fillna("").astype(str)

df["title_len"] = title_text.str.len()
df["desc_len"] = desc_text.str.len()
df["title_words"] = title_text.str.split().str.len()
df["desc_words"] = desc_text.str.split().str.len()

if "price" in df.columns:
    df["price_log"] = np.log1p(df["price"].clip(lower=0))
    if "category_name" in df.columns:
        cat_price_mean = df.groupby("category_name")["price_log"].transform("mean")
        df["price_diff_cat"] = df["price_log"] - cat_price_mean

if "image" in df.columns:
    df["has_image"] = df["image"].notnull().astype(int)
    df.drop(columns=["image"], inplace=True)

if "activation_date" in df.columns:
    dates = pd.to_datetime(df["activation_date"])
    df["dayofweek"] = dates.dt.dayofweek
    df["day"] = dates.dt.day
    df.drop(columns=["activation_date"], inplace=True)

if "user_id" in df.columns:
    if "item_seq_number" in df.columns:
        df["user_count"] = df.groupby("user_id")["item_seq_number"].transform("count")
    df.drop(columns=["user_id"], inplace=True)

if "item_id" in df.columns:
    df.drop(columns=["item_id"], inplace=True)

cat_cols = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("missing").astype(str).astype("category")

p1_text = df["param_1"].fillna("").astype(str) if "param_1" in df.columns else ""
combined_text = title_text + " " + desc_text + " " + p1_text

if "title" in df.columns:
    df.drop(columns=["title"], inplace=True)
if "description" in df.columns:
    df.drop(columns=["description"], inplace=True)

vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3)
X_text = vec.fit_transform(combined_text)

X_train_text = X_text[:len_train]
X_test_text = X_text[len_train:]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ridge = np.zeros(len_train)
test_ridge = np.zeros(len(df) - len_train)

for tr_idx, va_idx in kf.split(X_train_text):
    r = Ridge(alpha=5.0, solver="auto", random_state=42)
    r.fit(X_train_text[tr_idx], y_train[tr_idx])
    oof_ridge[va_idx] = r.predict(X_train_text[va_idx])
    test_ridge += r.predict(X_test_text) / kf.n_splits

df["ridge_pred"] = np.concatenate([oof_ridge, test_ridge])

X_train = df.iloc[:len_train].copy()
X_test = df.iloc[len_train:].copy()

preds_lgb = np.zeros(len(X_test))

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1200,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for tr_idx, va_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    preds_lgb += model.predict(X_test) / kf.n_splits

preds_final = np.clip(preds_lgb, 0.0, 1.0)

sub = pd.DataFrame({"id": np.arange(len(preds_final)), "deal_probability": preds_final})
sub.to_csv("predictions.csv", index=False)