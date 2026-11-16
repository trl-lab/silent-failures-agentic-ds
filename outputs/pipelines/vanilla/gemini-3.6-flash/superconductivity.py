import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "critical_temp"
features = [c for c in train_df.columns if c != target_col]

X = train_df[features].values
y = train_df[target_col].values
X_test = test_df[features].values

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test_df))
xgb_preds = np.zeros(len(test_df))
cat_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_train, y_train)
    lgb_preds += model_lgb.predict(X_test) / n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist"
    )
    model_xgb.fit(X_train, y_train)
    xgb_preds += model_xgb.predict(X_test) / n_splits

    model_cat = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    model_cat.fit(X_train, y_train)
    cat_preds += model_cat.predict(X_test) / n_splits

final_preds = lgb_preds * 0.40 + xgb_preds * 0.35 + cat_preds * 0.25
final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "critical_temp": final_preds
})

submission.to_csv("predictions.csv", index=False)