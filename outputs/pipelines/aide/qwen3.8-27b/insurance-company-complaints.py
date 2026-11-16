import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os


def preprocess(df):
    df = df.copy()

    # Drop high-cardinality text columns (heuristic: mean string length > 50)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dtype == "object":
            mean_len = df[col].dropna().astype(str).str.len().mean()
            if mean_len > 50:
                df = df.drop(columns=[col])

    # Extract temporal features from date-like columns
    date_cols = [
        col for col in df.columns if "date" in col.lower() or "time" in col.lower()
    ]
    date_cols += [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    date_cols = list(dict.fromkeys(date_cols))

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df = df.drop(columns=[col])

    # Convert categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df, cat_cols


def align_categories(train_df, test_df, cat_cols):
    """Align test categorical levels to training levels to avoid code mismatch."""
    for col in cat_cols:
        if col in train_df.columns and col in test_df.columns:
            if train_df[col].dtype == "category" and test_df[col].dtype == "category":
                train_cats = train_df[col].cat.categories
                train_df[col] = train_df[col].cat.set_categories(train_cats)
                test_df[col] = test_df[col].cat.set_categories(train_cats)
    return train_df, test_df


def add_frequency_encoding(X_train, X_other, cat_cols):
    """
    Add leakage-safe frequency encoding.

    Frequencies are computed from X_train only and applied to both X_train and X_other.
    In CV, X_train is the fold training split and X_other is the fold validation split.
    For final inference, X_train is the full training set and X_other is the test set.
    """
    X_train = X_train.copy()
    X_other = X_other.copy()

    for col in cat_cols:
        if col not in X_train.columns or col not in X_other.columns:
            continue

        freq = X_train[col].value_counts(normalize=True)

        if freq.empty:
            X_train[f"{col}_freq"] = 0.0
            X_other[f"{col}_freq"] = 0.0
        else:
            X_train[f"{col}_freq"] = X_train[col].map(freq).astype(float).fillna(0.0)
            X_other[f"{col}_freq"] = X_other[col].map(freq).astype(float).fillna(0.0)

    return X_train, X_other


# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Handle target
target = "Recovery"
train = train.dropna(subset=[target])
y = train[target]
train = train.drop(columns=[target])

if target in test.columns:
    test = test.drop(columns=[target])

# Preprocess
train_processed, cat_cols = preprocess(train)
test_processed, _ = preprocess(test)

# Align columns between train and test
common_cols = [col for col in train_processed.columns if col in test_processed.columns]
train_processed = train_processed[common_cols]
test_processed = test_processed[common_cols]
cat_cols = [col for col in cat_cols if col in common_cols]

# Align categorical categories between train and test
train_processed, test_processed = align_categories(
    train_processed, test_processed, cat_cols
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
    X_train = train_processed.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = train_processed.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # Add fold-wise frequency encoding
    X_train, X_val = add_frequency_encoding(X_train, X_val, cat_cols)
    X_val = X_val[X_train.columns]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    cv_scores.append(rmse)
    best_iterations.append(model.best_iteration)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

mean_cv_rmse = np.mean(cv_scores)
print(f"Mean CV RMSE: {mean_cv_rmse:.4f}")

# Train final model on full training data
avg_best_iter = max(1, int(np.mean(best_iterations)))

X_train_full, X_test = add_frequency_encoding(train_processed, test_processed, cat_cols)
X_test = X_test[X_train_full.columns]

dtrain_full = lgb.Dataset(X_train_full, label=y, categorical_feature=cat_cols)
model_final = lgb.train(params, dtrain_full, num_boost_round=avg_best_iter)

# Generate test predictions
test_preds = model_final.predict(X_test)

# Create and save submission
submission = pd.DataFrame(
    {
        "id": range(len(X_test)),
        "Recovery": test_preds,
    }
)

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
