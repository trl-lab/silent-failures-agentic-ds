import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "item_cnt_day"
ID = "id"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


def clean_and_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "item_price" in df.columns:
        df["item_price"] = (
            pd.to_numeric(df["item_price"], errors="coerce").clip(lower=0.0).fillna(0.0)
        )
        df["log_price"] = np.log1p(df["item_price"])

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
        df["month"] = d.dt.month
        df["year"] = d.dt.year
        df["day"] = d.dt.day
        df["dayofweek"] = d.dt.dayofweek

    if "date_block_num" in df.columns:
        df["date_block_num"] = (
            pd.to_numeric(df["date_block_num"], errors="coerce").fillna(0).astype(int)
        )
        df["dbn_sin"] = np.sin(2 * np.pi * df["date_block_num"] / 12)
        df["dbn_cos"] = np.cos(2 * np.pi * df["date_block_num"] / 12)

    return df


train = clean_and_feature(train)
test = clean_and_feature(test)

if TARGET in train.columns:
    train[TARGET] = (
        pd.to_numeric(train[TARGET], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

exclude_cols = {TARGET, ID, "date"}
feature_cols = [c for c in train.columns if c in test.columns and c not in exclude_cols]

if not feature_cols:
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    for c in feature_cols:
        if c not in test.columns:
            test[c] = np.nan

cat_cols = [c for c in ["shop_id", "item_id", "item_category_id"] if c in feature_cols]

for c in cat_cols:
    if train[c].dtype == object or test[c].dtype == object:
        all_values = pd.concat([train[c], test[c]], ignore_index=True).astype(str)
        categories = sorted(all_values.unique())
        mapping = {v: i for i, v in enumerate(categories)}
        train[c] = train[c].astype(str).map(mapping).fillna(-1).astype(int)
        test[c] = test[c].astype(str).map(mapping).fillna(-1).astype(int)
    else:
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(-1).astype(int)
        test[c] = pd.to_numeric(test[c], errors="coerce").fillna(-1).astype(int)

num_cols = [c for c in feature_cols if c not in cat_cols]
for c in num_cols:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    test[c] = pd.to_numeric(test[c], errors="coerce")

if "date_block_num" in train.columns and train["date_block_num"].nunique() > 1:
    max_block = train["date_block_num"].max()
    val_mask = train["date_block_num"] == max_block
    if val_mask.sum() < max(1000, len(train) // 10):
        val_mask = train["date_block_num"] >= max_block - 1
else:
    rng = np.random.RandomState(42)
    val_idx = rng.choice(len(train), size=int(len(train) * 0.1), replace=False)
    val_mask = np.zeros(len(train), dtype=bool)
    val_mask[val_idx] = True

train_part = train.loc[~val_mask].reset_index(drop=True)
val_part = train.loc[val_mask].reset_index(drop=True)

if len(train_part) == 0 or len(val_part) == 0:
    rng = np.random.RandomState(42)
    val_idx = rng.choice(len(train), size=int(len(train) * 0.1), replace=False)
    val_mask = np.zeros(len(train), dtype=bool)
    val_mask[val_idx] = True
    train_part = train.loc[~val_mask].reset_index(drop=True)
    val_part = train.loc[val_mask].reset_index(drop=True)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

dtrain = lgb.Dataset(
    train_part[feature_cols],
    label=train_part[TARGET],
    categorical_feature=cat_cols,
)
dval = lgb.Dataset(
    val_part[feature_cols],
    label=val_part[TARGET],
    categorical_feature=cat_cols,
    reference=dtrain,
)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    valid_sets=[dval],
    valid_names=["val"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

best_iter = getattr(model, "best_iteration", None)
if not best_iter or best_iter <= 0:
    best_iter = 1000

val_pred = model.predict(val_part[feature_cols], num_iteration=best_iter)
val_pred = np.clip(val_pred, 0.0, None)
val_rmse = float(np.sqrt(mean_squared_error(val_part[TARGET], val_pred)))
print(f"Validation RMSE: {val_rmse:.6f}")

dtrain_full = lgb.Dataset(
    train[feature_cols],
    label=train[TARGET],
    categorical_feature=cat_cols,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(test[feature_cols], num_iteration=best_iter)
test_pred = np.clip(test_pred, 0.0, None)

if ID in test.columns:
    submission_ids = test[ID].astype(int)
else:
    submission_ids = np.arange(len(test))

submission = pd.DataFrame(
    {
        ID: submission_ids,
        TARGET: test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
