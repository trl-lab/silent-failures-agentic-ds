import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Enrollment_FT_Teachers_Ratio"

# Identify feature columns
drop_cols = [target_col]
if "id" in train_df.columns:
    drop_cols.append("id")

features = [c for c in train_df.columns if c not in drop_cols]

# Preprocess categorical features
cat_cols = (
    train_df[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
for c in cat_cols:
    train_df[c] = train_df[c].astype("category")
    test_df[c] = test_df[c].astype("category")

X = train_df[features]
y = train_df[target_col].values
X_test = test_df[features]

# Setup Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
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
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Compute Out-of-Fold RMSE
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Out-of-Fold RMSE: {oof_rmse:.5f}")

# Prepare submission file
sub = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_preds})

sub.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
