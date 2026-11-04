import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

test_ids = list(range(len(test_df)))


# 2. Feature Engineering
def extract_baseline_features(df):
    df = df.copy()

    # Identify baseline record per patient (earliest visit)
    baseline_records = {}
    for patient, group in df.groupby("Patient"):
        min_week_idx = group["Weeks"].idxmin()
        min_row = group.loc[min_week_idx]
        baseline_records[patient] = {
            "base_Weeks": min_row["Weeks"],
            "base_Percent": min_row["Percent"] if "Percent" in min_row else np.nan,
            "base_Age": min_row["Age"] if "Age" in min_row else np.nan,
            "base_Sex": min_row["Sex"] if "Sex" in min_row else "Unknown",
            "base_SmokingStatus": (
                min_row["SmokingStatus"] if "SmokingStatus" in min_row else "Unknown"
            ),
            "base_FVC": (
                min_row["FVC"]
                if ("FVC" in min_row and pd.notnull(min_row["FVC"]))
                else np.nan
            ),
        }

    base_data = pd.DataFrame.from_dict(baseline_records, orient="index").reset_index()
    base_data.rename(columns={"index": "Patient"}, inplace=True)

    # Merge baseline back into the dataframe
    df = df.merge(base_data, on="Patient", how="left")

    # Temporal and physiological features
    df["delta_Weeks"] = df["Weeks"] - df["base_Weeks"]
    df["delta_Weeks_abs"] = df["delta_Weeks"].abs()
    df["delta_Weeks_sq"] = np.sign(df["delta_Weeks"]) * (df["delta_Weeks"] ** 2) / 100.0

    df["base_FVC_to_Age"] = df["base_FVC"] / (df["base_Age"] + 1.0)
    df["estimated_total_capacity"] = df["base_FVC"] / (
        (df["base_Percent"] / 100.0) + 1e-4
    )
    df["delta_Weeks_lung_capacity"] = (
        df["delta_Weeks"] * df["estimated_total_capacity"] / 1000.0
    )
    df["delta_Weeks_base_Percent"] = df["delta_Weeks"] * df["base_Percent"] / 100.0
    df["delta_Weeks_base_FVC"] = df["delta_Weeks"] * df["base_FVC"] / 1000.0
    df["delta_Weeks_Age"] = df["delta_Weeks"] * df["base_Age"] / 100.0

    # Categorical string formatting
    df["Sex"] = df["Sex"].fillna("Unknown").astype(str)
    df["SmokingStatus"] = df["SmokingStatus"].fillna("Unknown").astype(str)
    df["base_Sex"] = df["base_Sex"].fillna("Unknown").astype(str)
    df["base_SmokingStatus"] = df["base_SmokingStatus"].fillna("Unknown").astype(str)

    return df


train_fe = extract_baseline_features(train_df)
test_fe = extract_baseline_features(test_df)

# Target: relative decline from baseline FVC
train_fe["target_delta_FVC"] = train_fe["FVC"] - train_fe["base_FVC"]

cat_cols = ["Sex", "SmokingStatus", "base_Sex", "base_SmokingStatus"]
num_cols = [
    "Weeks",
    "Percent",
    "Age",
    "base_Weeks",
    "base_Percent",
    "base_Age",
    "base_FVC",
    "delta_Weeks",
    "delta_Weeks_abs",
    "delta_Weeks_sq",
    "base_FVC_to_Age",
    "estimated_total_capacity",
    "delta_Weeks_lung_capacity",
    "delta_Weeks_base_Percent",
    "delta_Weeks_base_FVC",
    "delta_Weeks_Age",
]

features = num_cols + cat_cols

# Ordinal encode categoricals for LightGBM
for col in cat_cols:
    train_fe[col + "_cat"] = train_fe[col].astype("category")
    test_fe[col + "_cat"] = test_fe[col].astype("category")

lgb_features = num_cols + [col + "_cat" for col in cat_cols]
cat_indices = [features.index(c) for c in cat_cols]

# 3. Model Training & Cross-Validation
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_preds_cb = np.zeros(len(train_fe))
test_preds_cb = np.zeros(len(test_fe))

oof_preds_lgb = np.zeros(len(train_fe))
test_preds_lgb = np.zeros(len(test_fe))

X_cb = train_fe[features]
X_cb_test = test_fe[features]

X_lgb = train_fe[lgb_features]
X_lgb_test = test_fe[lgb_features]

y_delta = train_fe["target_delta_FVC"].values
y_true = train_fe["FVC"].values
base_FVC_tr = train_fe["base_FVC"].values
base_FVC_te = test_fe["base_FVC"].values
groups = train_fe["Patient"].values

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_fe, y_delta, groups)):
    # CatBoost
    X_tr_cb, y_tr_delta = X_cb.iloc[train_idx], y_delta[train_idx]
    X_va_cb, y_va_delta = X_cb.iloc[val_idx], y_delta[val_idx]

    train_pool = Pool(X_tr_cb, y_tr_delta, cat_features=cat_indices)
    val_pool = Pool(X_va_cb, y_va_delta, cat_features=cat_indices)
    test_pool = Pool(X_cb_test, cat_features=cat_indices)

    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42 + fold,
        verbose=False,
    )
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=60, verbose=False)

    oof_delta_cb = cb_model.predict(val_pool)
    oof_preds_cb[val_idx] = base_FVC_tr[val_idx] + oof_delta_cb
    test_preds_cb += (base_FVC_te + cb_model.predict(test_pool)) / n_splits

    # LightGBM
    X_tr_lgb, y_tr_lgb = X_lgb.iloc[train_idx], y_delta[train_idx]
    X_va_lgb, y_va_lgb = X_lgb.iloc[val_idx], y_delta[val_idx]

    trn_data = lgb.Dataset(X_tr_lgb, label=y_tr_lgb)
    val_data = lgb.Dataset(X_va_lgb, label=y_va_lgb, reference=trn_data)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 10,
        "random_state": 42 + fold,
        "verbosity": -1,
    }

    lgb_model = lgb.train(
        params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(-1)],
    )

    oof_delta_lgb = lgb_model.predict(X_va_lgb, num_iteration=lgb_model.best_iteration)
    oof_preds_lgb[val_idx] = base_FVC_tr[val_idx] + oof_delta_lgb
    test_preds_lgb += (
        base_FVC_te
        + lgb_model.predict(X_lgb_test, num_iteration=lgb_model.best_iteration)
    ) / n_splits

# 4. Ensemble & Evaluation
oof_ensemble = 0.55 * oof_preds_cb + 0.45 * oof_preds_lgb
test_ensemble = 0.55 * test_preds_cb + 0.45 * test_preds_lgb

val_mae = mean_absolute_error(y_true, oof_ensemble)
print(f"Validation MAE: {val_mae:.4f}")

# 5. Save Submission
submission = pd.DataFrame({"id": test_ids, "FVC": test_ensemble})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
