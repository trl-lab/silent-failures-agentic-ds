import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total interior livable area"

# Remove rows with missing target in train
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Combine for uniform preprocessing
n_train = len(train_df)
df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# Feature engineering & preprocessing
drop_cols = ["id", "Id", "ID"]
features_to_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=features_to_drop)

# Identify text, categorical, and datetime columns
text_cols = []
categorical_cols = []

for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        mean_len = df[col].dropna().astype(str).map(len).mean()
        if mean_len > 50:
            text_cols.append(col)
            df[f"{col}_len"] = df[col].astype(str).str.len().fillna(0)
            df[f"{col}_word_count"] = (
                df[col]
                .astype(str)
                .str.split()
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
            )
        else:
            df[col] = df[col].astype("category")
            categorical_cols.append(col)
    elif df[col].dtype.name.startswith("datetime"):
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df = df.drop(columns=[col])

# Extract TF-IDF + TruncatedSVD features from text columns
if text_cols:
    combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(combined_text)

    n_components = 30
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    svd_feats = svd.fit_transform(tfidf_matrix)

    for i in range(n_components):
        df[f"text_svd_{i}"] = svd_feats[:, i]

    # Drop raw text columns to avoid high cardinality overhead
    df = df.drop(columns=text_cols)

# Separate back into train and test
X = df.iloc[:n_train].copy()
y = train_df[target_col].values
X_test = df.iloc[n_train:].copy()

# Setup 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, categorical_feature=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        reference=train_data,
        categorical_feature=categorical_cols,
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# Compute CV RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Total interior livable area": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
