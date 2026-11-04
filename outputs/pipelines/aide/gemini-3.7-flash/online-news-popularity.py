import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# 1. Load Data and Clean Column Names
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Strip leading/trailing whitespace from column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()


# 2. Feature Engineering
def extract_features(df):
    data = df.copy()

    # URL parsing if URL exists
    if "url" in data.columns:

        def parse_url(url):
            if not isinstance(url, str):
                return 0, 0, 0, 0
            match = re.search(r"mashable\.com/(\d{4})/(\d{2})/(\d{2})/([^/]+)/?", url)
            if match:
                year, month, day, slug = match.groups()
                slug_words = len(slug.replace("-", " ").split())
                slug_len = len(slug)
                return int(month), int(day), slug_words, slug_len
            return 0, 0, 0, 0

        url_features = [parse_url(u) for u in data["url"]]
        url_df = pd.DataFrame(
            url_features,
            columns=["url_month", "url_day", "url_slug_words", "url_slug_len"],
            index=data.index,
        )
        data = pd.concat([data, url_df], axis=1)
        data = data.drop(columns=["url"])

    # Ratios and Interactions
    content_words = data["n_tokens_content"].clip(lower=1)
    data["imgs_per_word"] = data["num_imgs"] / content_words
    data["videos_per_word"] = data["num_videos"] / content_words
    data["hrefs_per_word"] = data["num_hrefs"] / content_words
    data["self_hrefs_ratio"] = data["num_self_hrefs"] / (data["num_hrefs"] + 1)

    # Keyword interactions
    data["kw_avg_diff"] = data["kw_max_avg"] - data["kw_min_avg"]
    data["kw_spread"] = data["kw_max_max"] - data["kw_min_min"]
    data["self_ref_spread"] = (
        data["self_reference_max_shares"] - data["self_reference_min_shares"]
    )

    # LDA dominant topic
    lda_cols = ["LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04"]
    if all(c in data.columns for c in lda_cols):
        data["lda_max"] = data[lda_cols].max(axis=1)
        data["lda_argmax"] = data[lda_cols].values.argmax(axis=1)

    return data


# Process datasets
y_train = train_df["shares"].values
y_train_log = np.log1p(y_train)

X_train = extract_features(train_df.drop(columns=["shares"]))
X_test = extract_features(test_df.copy())

# Ensure feature alignment
feature_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

# 3. Model Training & Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_log = np.zeros(len(X_train))
test_preds_log = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1500,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train_log[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred = model.predict(X_va)
    oof_preds_log[val_idx] = val_pred
    test_preds_log += model.predict(X_test) / kf.n_splits

# 4. Evaluation
oof_preds = np.expm1(oof_preds_log)
oof_preds = np.clip(oof_preds, 0, None)
cv_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 5. Save Submission
test_preds = np.expm1(test_preds_log)
test_preds = np.clip(test_preds, 0, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "shares": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
