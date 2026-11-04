import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_path = "./working/submission.csv"

os.makedirs("./working", exist_ok=True)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Price"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
n_train = len(train_df)
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Identify feature types
cat_cols = []
text_cols = []
num_cols = []

for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        # Check average string length to distinguish text from categorical
        avg_len = (
            df[col].dropna().astype(str).map(len).mean()
            if len(df[col].dropna()) > 0
            else 0
        )
        nunique = df[col].nunique()
        if avg_len > 50 and nunique > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        num_cols.append(col)

# 3. Feature Engineering
features_train_list = []
features_test_list = []

# Process Numeric Features
if num_cols:
    num_df = df[num_cols].copy()
    for c in num_cols:
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
    features_train_list.append(num_df.iloc[:n_train].values)
    features_test_list.append(num_df.iloc[n_train:].values)

# Process Categorical Features (Frequency encoding + Category codes)
cat_feature_names = []
if cat_cols:
    cat_df_encoded = pd.DataFrame()
    for c in cat_cols:
        series_str = df[c].astype(str).fillna("__MISSING__")
        # Frequency encoding
        freq = series_str.value_counts(normalize=True)
        cat_df_encoded[f"{c}_freq"] = series_str.map(freq).astype(float)
        # Factorize / Category encoding
        cat_df_encoded[f"{c}_cat"] = pd.factorize(series_str)[0]
    features_train_list.append(cat_df_encoded.iloc[:n_train].values)
    features_test_list.append(cat_df_encoded.iloc[n_train:].values)

# Process Text Features with TF-IDF + TruncatedSVD
for col in text_cols:
    text_data = df[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        max_features=10000, stop_words="english", ngram_range=(1, 2)
    )
    svd = TruncatedSVD(n_components=25, random_state=42)

    tfidf_mat = tfidf.fit_transform(text_data)
    svd_feats = svd.fit_transform(tfidf_mat)

    features_train_list.append(svd_feats[:n_train])
    features_test_list.append(svd_feats[n_train:])

# Combine all processed features
X_train = np.hstack(features_train_list)
X_test = np.hstack(features_test_list)

# 4. 5-Fold Cross-Validation Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_val = X_train[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# 5. Evaluate and Print Metric
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV Out-Of-Fold RMSE: {overall_rmse:.4f}")

# 6. Save Submission File
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Price": test_preds})
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} with shape {submission.shape}")
