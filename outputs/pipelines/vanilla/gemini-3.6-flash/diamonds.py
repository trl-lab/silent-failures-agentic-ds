import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from scipy.optimize import minimize

# Load datasets
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target
y_train = train_df["price"].values
train_df = train_df.drop(columns=["price"])

if "price" in test_df.columns:
    test_df = test_df.drop(columns=["price"])

if "id" in train_df.columns:
    train_df = train_df.drop(columns=["id"])
if "id" in test_df.columns:
    test_df = test_df.drop(columns=["id"])

# Combine for feature engineering
df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Ordinal mappings for diamond specifics
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_map = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}

df_all["cut_ord"] = df_all["cut"].map(cut_map).fillna(-1)
df_all["color_ord"] = df_all["color"].map(color_map).fillna(-1)
df_all["clarity_ord"] = df_all["clarity"].map(clarity_map).fillna(-1)

# Domain feature engineering
df_all["volume"] = df_all["x"] * df_all["y"] * df_all["z"]
df_all["carat_density"] = df_all["carat"] / (df_all["volume"] + 1e-5)
df_all["xy_mean"] = (df_all["x"] + df_all["y"]) / 2.0
df_all["calc_depth"] = (
    200.0 * df_all["z"] / (df_all["x"] + df_all["y"] + 1e-5)
)
df_all["depth_diff"] = df_all["depth"] - df_all["calc_depth"]

df_all["surface_area"] = 2 * (
    df_all["x"] * df_all["y"]
    + df_all["y"] * df_all["z"]
    + df_all["x"] * df_all["z"]
)

# Dimension ratios
df_all["x_y_ratio"] = df_all["x"] / (df_all["y"] + 1e-5)
df_all["x_z_ratio"] = df_all["x"] / (df_all["z"] + 1e-5)
df_all["y_z_ratio"] = df_all["y"] / (df_all["z"] + 1e-5)

# Interactive quality metrics
df_all["overall_grade"] = (
    df_all["cut_ord"] + df_all["color_ord"] + df_all["clarity_ord"]
)
df_all["grade_prod"] = (
    (df_all["cut_ord"] + 1)
    * (df_all["color_ord"] + 1)
    * (df_all["clarity_ord"] + 1)
)
df_all["carat_x_grade"] = df_all["carat"] * df_all["overall_grade"]

# Group aggregations
group_cols = ["cut", "color", "clarity"]
for stat_feat in ["carat", "volume", "table", "depth"]:
    grouped = df_all.groupby(group_cols)[stat_feat]
    mean_val = grouped.transform("mean")
    df_all[f"{stat_feat}_grp_mean"] = mean_val
    df_all[f"{stat_feat}_grp_diff"] = df_all[stat_feat] - mean_val
    df_all[f"{stat_feat}_grp_ratio"] = df_all[stat_feat] / (mean_val + 1e-5)

# Convert string categories
cat_cols = ["cut", "color", "clarity"]
for col in cat_cols:
    df_all[col] = df_all[col].astype("category")

# Separate train and test sets
X_train = df_all.iloc[: len(train_df)].copy()
X_test = df_all.iloc[len(train_df) :].copy()

# Setup K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb = np.zeros(len(y_train))
oof_preds_xgb = np.zeros(len(y_train))
oof_preds_cb = np.zeros(len(y_train))
oof_preds_lgb_log = np.zeros(len(y_train))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_lgb_log = np.zeros(len(X_test))

xgb_cols = [c for c in X_train.columns if c not in cat_cols]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # 1. LightGBM (RMSE on Raw Target)
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / 5.0

    # 2. XGBoost (RMSE on Raw Target)
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(
        X_tr[xgb_cols], y_tr, eval_set=[(X_va[xgb_cols], y_va)], verbose=False
    )
    oof_preds_xgb[val_idx] = model_xgb.predict(X_va[xgb_cols])
    test_preds_xgb += model_xgb.predict(X_test[xgb_cols]) / 5.0

    # 3. CatBoost (RMSE on Raw Target)
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_preds_cb[val_idx] = model_cb.predict(X_va)
    test_preds_cb += model_cb.predict(X_test) / 5.0

    # 4. LightGBM (Log-Transformed Target)
    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)
    model_lgb_log = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        random_state=142 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb_log.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds_lgb_log[val_idx] = np.expm1(model_lgb_log.predict(X_va))
    test_preds_lgb_log += np.expm1(model_lgb_log.predict(X_test)) / 5.0

# Optimize blending weights on Out-Of-Fold predictions
OOF = np.column_stack(
    [oof_preds_lgb, oof_preds_xgb, oof_preds_cb, oof_preds_lgb_log]
)
TEST = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_lgb_log]
)


def loss_func(weights):
    w = weights / np.sum(weights)
    pred = OOF @ w
    return np.sqrt(np.mean((pred - y_train) ** 2))


init_weights = np.array([0.35, 0.25, 0.25, 0.15])
bounds = [(0, 1)] * 4
res = minimize(loss_func, init_weights, bounds=bounds, method="SLSQP")

best_weights = res.x / np.sum(res.x)
final_test_preds = TEST @ best_weights

# Ensure prices are non-negative / physically meaningful
final_test_preds = np.clip(final_test_preds, 300, None)

submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "price": final_test_preds}
)

submission.to_csv("predictions.csv", index=False)