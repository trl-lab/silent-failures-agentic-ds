import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/concrete-compressive-strength/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "strength"
y_train = train[target_col].values
X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.drop(columns=[target_col], errors="ignore")


def engineer_features(df):
  df = df.copy()
  df["binder"] = df["cement"] + df["blast_furnace_slag"] + df["fly_ash"]
  df["water_cement_ratio"] = df["water"] / (df["cement"] + 1e-5)
  df["water_binder_ratio"] = df["water"] / (df["binder"] + 1e-5)
  df["superplasticizer_binder_ratio"] = df["superplasticizer"] / (
      df["binder"] + 1e-5
  )
  df["total_aggregates"] = df["coarse_aggregate"] + df["fine_aggregate"]
  df["fine_coarse_ratio"] = df["fine_aggregate"] / (
      df["coarse_aggregate"] + 1e-5
  )
  df["binder_aggregate_ratio"] = df["binder"] / (df["total_aggregates"] + 1e-5)
  df["log_age"] = np.log1p(df["age"])
  df["age_binder"] = df["log_age"] * df["binder"]
  return df


X_train = engineer_features(X_train_raw)
X_test = engineer_features(X_test_raw)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

test_preds = np.zeros(len(X_test))
models_count = 0

seeds = [42, 2024]

for seed in seeds:
  kf = KFold(n_splits=10, shuffle=True, random_state=seed)

  for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed + fold,
        verbose=-1,
    )
    model_lgb.fit(X_tr, y_tr)
    test_preds += model_lgb.predict(X_test)
    models_count += 1

    model_xgb = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed + fold,
        verbosity=0,
    )
    model_xgb.fit(X_tr, y_tr)
    test_preds += model_xgb.predict(X_test)
    models_count += 1

    model_cb = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        random_seed=seed + fold,
        verbose=0,
    )
    model_cb.fit(X_tr, y_tr)
    test_preds += model_cb.predict(X_test)
    models_count += 1

    model_et = ExtraTreesRegressor(
        n_estimators=300, max_depth=12, random_state=seed + fold, n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)
    test_preds += model_et.predict(X_test)
    models_count += 1

    model_hgb = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_depth=6, random_state=seed + fold
    )
    model_hgb.fit(X_tr, y_tr)
    test_preds += model_hgb.predict(X_test)
    models_count += 1

test_preds /= models_count

sub = pd.DataFrame({"id": np.arange(len(test)), "strength": test_preds})

sub.to_csv("predictions.csv", index=False)