import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Financial Assistance"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Ensure same columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]


def preprocess_features(train, test):
    df_all = pd.concat([train, test], axis=0).reset_index(drop=True)
    n_train = len(train)

    datetime_cols = []
    for col in df_all.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(df_all[col])
            or "date" in col.lower()
            or "year" in col.lower()
            or "time" in col.lower()
        ):
            try:
                parsed = pd.to_datetime(df_all[col], errors="coerce")
                if parsed.notna().sum() > 0.3 * len(df_all):
                    df_all[col + "_dt"] = parsed
                    datetime_cols.append(col + "_dt")
            except Exception:
                pass

    # Extract date features
    for col in datetime_cols:
        df_all[f"{col}_year"] = df_all[col].dt.year
        df_all[f"{col}_month"] = df_all[col].dt.month
        df_all[f"{col}_day"] = df_all[col].dt.day
        df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
        df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear
        df_all.drop(columns=[col], inplace=True)

    extracted_years = [c for c in df_all.columns if c.endswith("_year")]
    if len(extracted_years) >= 2:
        for i in range(len(extracted_years)):
            for j in range(i + 1, len(extracted_years)):
                c1, c2 = extracted_years[i], extracted_years[j]
                df_all[f"{c1}_minus_{c2}"] = df_all[c1] - df_all[c2]

    # Process categorical columns
    cat_cols = []
    for col in df_all.columns:
        if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
            freq = df_all[col].value_counts(dropna=False)
            df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(int)
            df_all[col] = df_all[col].fillna("__MISSING__").astype("category")
            cat_cols.append(col)
        elif not pd.api.types.is_numeric_dtype(df_all[col]):
            df_all[col] = df_all[col].astype(str).astype("category")
            cat_cols.append(col)

    train_feat = df_all.iloc[:n_train].copy().reset_index(drop=True)
    test_feat = df_all.iloc[n_train:].copy().reset_index(drop=True)
    return train_feat, test_feat, cat_cols


X_train_proc, X_test_proc, cat_features = preprocess_features(X_train, X_test)

# Setup 5-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_proc))
test_preds = np.zeros(len(X_test_proc))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    val_data = lgb.Dataset(
        X_va, label=y_va, reference=trn_data, categorical_feature=cat_features
    )

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_rmse = root_mean_squared_error(y_va, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    test_preds += (
        model.predict(X_test_proc, num_iteration=model.best_iteration) / n_splits
    )

overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall Out-of-Fold RMSE: {overall_rmse:.5f}")

# Save submission
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds)),
        "Financial Assistance": test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved with shape:", submission.shape)
