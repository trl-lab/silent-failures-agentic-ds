import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


def preprocess_features(df):
    data = df.copy()

    # Extract string length from url if available and drop non-numeric column
    if "url" in data.columns:
        data["url_length"] = data["url"].astype(str).str.len()
        data = data.drop(columns=["url"])

    # Feature interactions and domain ratios
    if "n_tokens_content" in data.columns and "n_tokens_title" in data.columns:
        data["content_title_ratio"] = data["n_tokens_content"] / (
            data["n_tokens_title"] + 1e-5
        )

    if "num_hrefs" in data.columns and "n_tokens_content" in data.columns:
        data["href_per_content"] = data["num_hrefs"] / (data["n_tokens_content"] + 1.0)

    if "num_imgs" in data.columns and "n_tokens_content" in data.columns:
        data["img_per_content"] = data["num_imgs"] / (data["n_tokens_content"] + 1.0)

    if "num_videos" in data.columns and "n_tokens_content" in data.columns:
        data["video_per_content"] = data["num_videos"] / (
            data["n_tokens_content"] + 1.0
        )

    # Summary statistics across keyword features
    kw_cols = [c for c in data.columns if c.startswith("kw_")]
    if kw_cols:
        data["kw_summary_mean"] = data[kw_cols].mean(axis=1)
        data["kw_summary_std"] = data[kw_cols].std(axis=1)
        data["kw_summary_max"] = data[kw_cols].max(axis=1)
        data["kw_summary_min"] = data[kw_cols].min(axis=1)

    # Dominant topic assignment from LDA distributions
    lda_cols = [c for c in data.columns if c.startswith("LDA_")]
    if lda_cols:
        data["lda_max"] = data[lda_cols].max(axis=1)
        data["lda_argmax"] = data[lda_cols].values.argmax(axis=1)

    # Handle infinite and missing values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)

    return data


# Target and features setup
target_col = "shares"
y = train_df[target_col].values

X = preprocess_features(train_df.drop(columns=[target_col]))
X_test = preprocess_features(test_df.drop(columns=[target_col], errors="ignore"))

# Align feature schemas between train and test
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# 5-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1000,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "shares": np.clip(test_preds, 0, None)}
)
submission.to_csv("./working/submission.csv", index=False)
