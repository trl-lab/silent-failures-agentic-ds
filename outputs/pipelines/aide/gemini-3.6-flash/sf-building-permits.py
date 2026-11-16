import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "time_to_approve"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

n_train = len(train_df)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

if "id" in df_all.columns:
    df_all = df_all.drop(columns=["id"])

# 2. Datetime Processing
date_cols = []
for col in list(df_all.columns):
    if df_all[col].dtype == "object" or "date" in col.lower() or "time" in col.lower():
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notna().sum() > 0.1 * len(df_all):
                date_cols.append(col)
                df_all[col + "_parsed"] = converted
        except Exception:
            pass

new_features = pd.DataFrame(index=df_all.index)

parsed_ts_cols = []
for col in date_cols:
    dt_series = df_all[col + "_parsed"]
    new_features[col + "_year"] = dt_series.dt.year
    new_features[col + "_month"] = dt_series.dt.month
    new_features[col + "_day"] = dt_series.dt.day
    new_features[col + "_dayofweek"] = dt_series.dt.dayofweek
    new_features[col + "_dayofyear"] = dt_series.dt.dayofyear
    new_features[col + "_quarter"] = dt_series.dt.quarter

    ts_col = col + "_timestamp"
    new_features[ts_col] = dt_series.astype("int64") // 10**9
    parsed_ts_cols.append(ts_col)

    df_all = df_all.drop(columns=[col, col + "_parsed"])

for i in range(len(parsed_ts_cols)):
    for j in range(i + 1, len(parsed_ts_cols)):
        c1, c2 = parsed_ts_cols[i], parsed_ts_cols[j]
        new_features[f"{c1}_diff_{c2}"] = new_features[c1] - new_features[c2]

# 3. Text Processing
text_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object":
        mean_len = df_all[col].astype(str).str.len().mean()
        if mean_len > 25:
            text_cols.append(col)

for col in text_cols:
    tfidf = TfidfVectorizer(max_features=2500, stop_words="english")
    text_data = df_all[col].fillna("").astype(str)
    tfidf_mat = tfidf.fit_transform(text_data)
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd_feats = svd.fit_transform(tfidf_mat)
    for i in range(10):
        new_features[f"{col}_svd_{i}"] = svd_feats[:, i]
    df_all = df_all.drop(columns=[col])

# 4. Categorical Processing & Frequency Encoding
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    freq = df_all[col].value_counts(normalize=True)
    new_features[col + "_freq"] = df_all[col].map(freq).fillna(0)
    df_all[col] = df_all[col].astype("category").cat.codes

# 5. Group Aggregations
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
main_cats = [c for c in cat_cols if df_all[c].nunique() < 500][:3]
for cat in main_cats:
    for num in num_cols[:5]:
        grp = df_all.groupby(cat)[num].agg(["mean", "std"]).reset_index()
        grp.columns = [cat, f"{num}_grp_{cat}_mean", f"{num}_grp_{cat}_std"]
        df_all = df_all.merge(grp, on=cat, how="left")

df_all = pd.concat([df_all, new_features], axis=1)

# 6. Train/Test Split & Out-Of-Fold Target Encoding
X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for cat in main_cats:
    X_train[f"{cat}_te"] = 0.0
    X_test[f"{cat}_te"] = 0.0
    global_mean = y.mean()
    te_test_folds = np.zeros(len(X_test))

    for train_idx, val_idx in kf.split(X_train):
        y_tr_fold = y[train_idx]
        cat_tr_fold = X_train.iloc[train_idx][cat]
        means_df = (
            pd.DataFrame({cat: cat_tr_fold, "target": y_tr_fold})
            .groupby(cat)["target"]
            .agg(["mean", "count"])
        )
        smooth = (means_df["count"] * means_df["mean"] + 10 * global_mean) / (
            means_df["count"] + 10
        )

        X_train.loc[val_idx, f"{cat}_te"] = (
            X_train.loc[val_idx, cat].map(smooth).fillna(global_mean)
        )
        te_test_folds += X_test[cat].map(smooth).fillna(global_mean) / 5.0

    X_test[f"{cat}_te"] = te_test_folds

# 7. Model Cross-Validation with XGBoost
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / 5.0

if (y >= 0).all():
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# 8. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "time_to_approve": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
