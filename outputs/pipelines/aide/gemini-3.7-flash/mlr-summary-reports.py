import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Paths
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, "submission.csv")

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Adjusted MLR"
raw_features = [c for c in train_df.columns if c != target_col]


# Clean column names to remove special characters unsupported by LightGBM
def clean_col_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


rename_mapping = {col: clean_col_name(col) for col in raw_features}
train_df = train_df.rename(columns=rename_mapping)
test_df = test_df.rename(columns=rename_mapping)
features = [rename_mapping[col] for col in raw_features]

# Preprocess categorical features
for col in features:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        le = LabelEncoder()
        full_vals = (
            pd.concat([train_df[col], test_df[col]], axis=0)
            .astype(str)
            .fillna("missing")
        )
        le.fit(full_vals)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("missing"))
        test_df[col] = le.transform(test_df[col].astype(str).fillna("missing"))

X = train_df[features].copy()
y = train_df[target_col].copy()
X_test = test_df[features].copy()

# Setup 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
    "n_estimators": 1500,
    "verbosity": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], callbacks=callbacks)

    val_pred = model.predict(X_val_f)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Prepare submission
submission = pd.DataFrame({"id": np.arange(len(test_df)), target_col: test_preds})

submission.to_csv(submission_path, index=False)
print(f"Saved test predictions to {submission_path}")
