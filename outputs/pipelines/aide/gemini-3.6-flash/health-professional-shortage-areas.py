import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "HPSA Score"
y = train_df[target_col].values.astype(float)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

features = [c for c in X_train_raw.columns if c in X_test_raw.columns]
X_train_raw = X_train_raw[features]
X_test_raw = X_test_raw[features]

n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

# 2. Datetime Processing
datetime_cols = []
for col in combined.columns:
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        datetime_cols.append(col)
    elif combined[col].dtype == "object":
        sample = combined[col].dropna().head(100)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() > len(sample) * 0.8:
                    combined[col] = pd.to_datetime(combined[col], errors="coerce")
                    datetime_cols.append(col)
            except Exception:
                pass

for col in datetime_cols:
    dt = combined[col].dt
    combined[f"{col}_year"] = dt.year
    combined[f"{col}_month"] = dt.month
    combined[f"{col}_day"] = dt.day
    combined[f"{col}_dayofweek"] = dt.dayofweek
    combined[f"{col}_dayofyear"] = dt.dayofyear
    combined[f"{col}_timestamp"] = (
        combined[col] - pd.Timestamp("1970-01-01")
    ).dt.total_seconds()
    combined.drop(columns=[col], inplace=True)

# 3. Categorical and Text Feature Identification
cat_cols = []
text_cols = []

for col in combined.columns:
    if combined[col].dtype == "object" or isinstance(
        combined[col].dtype, pd.CategoricalDtype
    ):
        unique_cnt = combined[col].nunique()
        avg_len = combined[col].dropna().astype(str).str.len().mean()
        if (
            avg_len is not None
            and not np.isnan(avg_len)
            and avg_len > 40
            and unique_cnt > 100
        ):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# Text Feature Extraction with TF-IDF + SVD
for col in text_cols:
    text_data = combined[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_mat = tfidf.fit_transform(text_data)
    n_comp = min(10, tfidf_mat.shape[1] - 1)
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_comp):
            combined[f"{col}_svd_{i}"] = svd_mat[:, i]
    combined.drop(columns=[col], inplace=True)

# Fill and format string categoricals
for col in cat_cols:
    combined[col] = combined[col].fillna("missing").astype(str)

# Frequency Encoding
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)

# Group Numerical Aggregations
num_cols = [
    c
    for c in combined.select_dtypes(include=[np.number]).columns
    if combined[c].nunique() > 1 and not c.endswith("_freq")
]
group_cat_cols = [c for c in cat_cols if 2 <= combined[c].nunique() <= 300]

for cat in group_cat_cols[:5]:
    for num in num_cols[:4]:
        grp_mean = combined.groupby(cat, observed=False)[num].transform("mean")
        grp_std = combined.groupby(cat, observed=False)[num].transform("std")
        combined[f"{cat}_{num}_grp_mean"] = grp_mean
        combined[f"{cat}_{num}_grp_std"] = grp_std.fillna(0)

# 4. Train/Test Split & Out-of-fold Multi-Statistic Target Encoding
X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y.mean())
global_std = float(y.std())

for col in cat_cols:
    if X_train[col].nunique() > 5:
        te_train_mean = np.zeros(len(X_train))
        te_train_std = np.zeros(len(X_train))
        te_test_mean = np.zeros(len(X_test))
        te_test_std = np.zeros(len(X_test))

        for train_idx, val_idx in kf.split(X_train):
            tr_cat = X_train.iloc[train_idx][col]
            tr_y = y[train_idx]

            fold_df = pd.DataFrame({"cat": tr_cat, "y": tr_y})
            stats = fold_df.groupby("cat", observed=False)["y"].agg(
                ["mean", "std", "count"]
            )

            smooth_means = (stats["count"] * stats["mean"] + 10 * global_mean) / (
                stats["count"] + 10
            )
            smooth_stds = (
                stats["count"] * stats["std"].fillna(global_std) + 10 * global_std
            ) / (stats["count"] + 10)

            val_cat = X_train.iloc[val_idx][col]
            te_train_mean[val_idx] = (
                val_cat.map(smooth_means).astype(float).fillna(global_mean).values
            )
            te_train_std[val_idx] = (
                val_cat.map(smooth_stds).astype(float).fillna(global_std).values
            )

            te_test_mean += (
                X_test[col].map(smooth_means).astype(float).fillna(global_mean).values
                / 5.0
            )
            te_test_std += (
                X_test[col].map(smooth_stds).astype(float).fillna(global_std).values
                / 5.0
            )

        X_train[f"{col}_te_mean"] = te_train_mean
        X_train[f"{col}_te_std"] = te_train_std
        X_test[f"{col}_te_mean"] = te_test_mean
        X_test[f"{col}_te_std"] = te_test_std

# Prepare datasets for LightGBM and CatBoost
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# 5. Model Training (LightGBM + CatBoost Ensemble)
oof_lgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
test_lgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 2500,
}

cb_params = {
    "iterations": 2500,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # --- LightGBM ---
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_lgb[val_idx] = model_lgb.predict(X_va_lgb)
    test_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

    # --- CatBoost ---
    X_tr_cb = X_train.iloc[train_idx]
    X_va_cb = X_train.iloc[val_idx]

    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols,
        use_best_model=True,
    )

    oof_cb[val_idx] = model_cb.predict(X_va_cb)
    test_cb += model_cb.predict(X_test) / kf.n_splits

oof_blend = 0.5 * oof_lgb + 0.5 * oof_cb
test_preds = 0.5 * test_lgb + 0.5 * test_cb

rmse_score = root_mean_squared_error(y, oof_blend)
print(f"Validation Metric (RMSE): {rmse_score:.5f}")

# 6. Save Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "HPSA Score": test_preds})
sub.to_csv("./working/submission.csv", index=False)
