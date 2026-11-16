import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "number_of_officers"
features = [col for col in train_df.columns if col != target_col]

# 2. Preprocess datetime and categorical columns
for col in features:
    if pd.api.types.is_datetime64_any_dtype(
        train_df[col]
    ) or pd.api.types.is_datetime64_any_dtype(test_df[col]):
        for df in [train_df, test_df]:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_timestamp"] = df[col].astype("int64") // 10**9
            df.drop(columns=[col], inplace=True)

# Update feature list after datetime transformations
features = [col for col in train_df.columns if col != target_col]

categorical_cols = []
for col in features:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        categorical_cols.append(col)
        # Harmonize categories between train and test
        categories = sorted(
            list(
                set(train_df[col].dropna().astype(str)).union(
                    set(test_df[col].dropna().astype(str))
                )
            )
        )
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=categories)
        test_df[col] = pd.Categorical(test_df[col].astype(str), categories=categories)

# 3. 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

# 4. Train LightGBM across folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    X_train = train_df.iloc[train_idx][features]
    y_train = train_df.iloc[train_idx][target_col]
    X_val = train_df.iloc[val_idx][features]
    y_val = train_df.iloc[val_idx][target_col]

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    test_preds += (
        model.predict(test_df[features], num_iteration=model.best_iteration)
        / kf.n_splits
    )

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Post-processing: predictions should not be negative
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(train_df[target_col], oof_preds)
print(f"\nOverall Out-of-Fold RMSE: {cv_rmse:.4f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "number_of_officers": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
