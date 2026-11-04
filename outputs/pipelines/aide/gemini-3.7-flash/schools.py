import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor, Pool

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Enrollment_FT_Teachers_Ratio"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Ensure matching columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]


# 2. Feature Engineering
def preprocess_features(train, test):
    df_all = pd.concat([train, test], axis=0, ignore_index=True)
    n_train = len(train)

    # Standardize grade mappings if available
    grade_map = {
        "PK": -1,
        "KG": 0,
        "01": 1,
        "02": 2,
        "03": 3,
        "04": 4,
        "05": 5,
        "06": 6,
        "07": 7,
        "08": 8,
        "09": 9,
        "10": 10,
        "11": 11,
        "12": 12,
        "13": 13,
        "UG": 14,
        "N": np.nan,
    }

    for col in ["START_GRADE", "END_GRADE", "LOW_GRADE", "HIGH_GRADE"]:
        if col in df_all.columns:
            mapped = df_all[col].astype(str).str.strip().map(grade_map)
            df_all[f"{col}_NUM"] = pd.to_numeric(mapped, errors="coerce")

    if "START_GRADE_NUM" in df_all.columns and "END_GRADE_NUM" in df_all.columns:
        df_all["GRADE_SPAN"] = df_all["END_GRADE_NUM"] - df_all["START_GRADE_NUM"] + 1
    elif "LOW_GRADE_NUM" in df_all.columns and "HIGH_GRADE_NUM" in df_all.columns:
        df_all["GRADE_SPAN"] = df_all["HIGH_GRADE_NUM"] - df_all["LOW_GRADE_NUM"] + 1

    # Spatial coordinates & rotation features
    lat_col = next(
        (c for c in df_all.columns if c.upper() in ["LATITUDE", "LAT", "Y"]), None
    )
    lon_col = next(
        (c for c in df_all.columns if c.upper() in ["LONGITUDE", "LON", "X"]), None
    )

    if lat_col and lon_col:
        lat = pd.to_numeric(df_all[lat_col], errors="coerce").fillna(0)
        lon = pd.to_numeric(df_all[lon_col], errors="coerce").fillna(0)
        df_all["ROT_45_X"] = 0.7071 * lon - 0.7071 * lat
        df_all["ROT_45_Y"] = 0.7071 * lon + 0.7071 * lat
        df_all["RAD_DIST"] = np.sqrt(lat**2 + lon**2)

    # Enrollment-related aggregations if available
    enroll_col = next(
        (c for c in df_all.columns if "ENROLL" in c.upper() or "POP" in c.upper()), None
    )
    group_col = next(
        (c for c in ["COUNTY", "STATE", "CITY", "TYPE"] if c in df_all.columns), None
    )

    if enroll_col and group_col:
        enroll_numeric = pd.to_numeric(df_all[enroll_col], errors="coerce")
        group_mean = enroll_numeric.groupby(df_all[group_col]).transform("mean")
        group_std = enroll_numeric.groupby(df_all[group_col]).transform("std").fillna(0)
        df_all[f"{enroll_col}_DIFF_GROUP_MEAN"] = enroll_numeric - group_mean
        df_all[f"{enroll_col}_GROUP_Z"] = (enroll_numeric - group_mean) / (
            group_std + 1e-5
        )

    # Identify categorical and numeric columns
    cat_cols = []
    for col in df_all.columns:
        if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
            df_all[col] = df_all[col].astype(str).fillna("missing")
            # Frequency encoding
            freq = df_all[col].value_counts(normalize=True)
            df_all[f"{col}_FREQ"] = df_all[col].map(freq)
            cat_cols.append(col)
        elif np.issubdtype(df_all[col].dtype, np.number):
            df_all[col] = df_all[col].fillna(df_all[col].median())

    train_processed = df_all.iloc[:n_train].copy()
    test_processed = df_all.iloc[n_train:].copy()

    return train_processed, test_processed, cat_cols


X_train_proc, X_test_proc, cat_features = preprocess_features(X_train, X_test)

# 3. 5-Fold Cross Validation Modeling with CatBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train_proc))
test_preds = np.zeros(len(X_test_proc))

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_va, y_va, cat_features=cat_features)
    test_pool = Pool(X_test_proc, cat_features=cat_features)

    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    oof_preds[val_idx] = model.predict(val_pool)
    test_preds += model.predict(test_pool) / kf.n_splits

# Post-processing: Ratios cannot be negative
oof_preds = np.clip(oof_preds, a_min=0, a_max=None)
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Validation Metric: {cv_rmse:.5f}")

# 5. Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Enrollment_FT_Teachers_Ratio": test_preds}
)
sub.to_csv("./working/submission.csv", index=False)
