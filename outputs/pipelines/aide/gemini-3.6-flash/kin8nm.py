import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "y"
drop_cols = [target_col, "id"] if "id" in train_df.columns else [target_col]
feature_cols = [c for c in train_df.columns if c not in drop_cols]


# 2. Feature Engineering Function
def create_features(df, base_features):
    df_feat = df[base_features].copy()

    # Trigonometric features for kinematic joint angles
    for col in base_features:
        df_feat[f"{col}_sin"] = np.sin(df_feat[col])
        df_feat[f"{col}_cos"] = np.cos(df_feat[col])

    # Pairwise sin/cos products for key interaction terms
    for i in range(len(base_features)):
        for j in range(i + 1, len(base_features)):
            f1, f2 = base_features[i], base_features[j]
            df_feat[f"sin_{f1}_{f2}"] = np.sin(df[f1] + df[f2])
            df_feat[f"cos_{f1}_{f2}"] = np.cos(df[f1] + df[f2])

    return df_feat


X = create_features(train_df, feature_cols)
y = train_df[target_col].values
X_test = create_features(test_df, feature_cols)

# 3. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1500,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# 4. Evaluate Metric
oof_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV OOF RMSE: {oof_rmse:.5f}")

# 5. Save Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {
        "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
        "y": test_preds,
    }
)
sub.to_csv("./working/submission.csv", index=False)
