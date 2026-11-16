import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Appliances"
y_train = train_df[target_col].values
min_y = train_df[target_col].min()

def process_data(df):
    data = df.copy()
    if "date" in data.columns:
        dt = pd.to_datetime(data["date"])
        data["hour"] = dt.dt.hour
        data["minute"] = dt.dt.minute
        data["dayofweek"] = dt.dt.dayofweek
        data["day"] = dt.dt.day
        data["month"] = dt.dt.month
        data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

        nsm = data["hour"] * 3600 + data["minute"] * 60
        data["sin_nsm"] = np.sin(2 * np.pi * nsm / 86400)
        data["cos_nsm"] = np.cos(2 * np.pi * nsm / 86400)

        data = data.drop(columns=["date"])

    t_cols = [c for c in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"] if c in data.columns]
    if t_cols:
        data["T_mean"] = data[t_cols].mean(axis=1)
        data["T_std"] = data[t_cols].std(axis=1)
        data["T_min"] = data[t_cols].min(axis=1)
        data["T_max"] = data[t_cols].max(axis=1)

    rh_cols = [c for c in ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"] if c in data.columns]
    if rh_cols:
        data["RH_mean"] = data[rh_cols].mean(axis=1)
        data["RH_std"] = data[rh_cols].std(axis=1)
        data["RH_min"] = data[rh_cols].min(axis=1)
        data["RH_max"] = data[rh_cols].max(axis=1)

    if "T_mean" in data.columns and "T_out" in data.columns:
        data["T_diff_out"] = data["T_mean"] - data["T_out"]
    if "RH_mean" in data.columns and "RH_out" in data.columns:
        data["RH_diff_out"] = data["RH_mean"] - data["RH_out"]

    if target_col in data.columns:
        data = data.drop(columns=[target_col])

    return data

X_train = process_data(train_df)
X_test = process_data(test_df)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
        n_jobs=-1,
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=7,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1,
    )
    cat_model.fit(X_tr, y_tr)
    cat_preds += cat_model.predict(X_test) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds
final_preds = np.clip(final_preds, min_y, None)

sub = pd.DataFrame({"id": np.arange(len(final_preds)), "Appliances": final_preds})
sub.to_csv("predictions.csv", index=False)