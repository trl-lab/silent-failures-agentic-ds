import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Preprocessing & Feature Selection
target_col = "ViolentCrimesPerPop"
drop_cols = ["communityname", "fold", target_col]
feature_cols = [col for col in train_df.columns if col not in drop_cols]

# Convert features to numeric
X = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = train_df[target_col].values
X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

# 3. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 1000,
}

# 4. Model Training & Evaluation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model.fit(
        X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False
    )

    val_preds = model.predict(X_val_fold)
    oof_predictions[val_idx] = val_preds
    test_predictions += model.predict(X_test) / kf.n_splits

# 5. Calculate validation metric
oof_rmse = root_mean_squared_error(y, oof_predictions)
print(f"Validation RMSE: {oof_rmse:.5f}")

# 6. Generate submission file
test_predictions_clipped = np.clip(test_predictions, 0.0, 1.0)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "ViolentCrimesPerPop": test_predictions_clipped}
)
submission.to_csv("./working/submission.csv", index=False)
