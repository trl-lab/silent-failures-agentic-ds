import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "WATERDEPTH"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

# Drop identifier columns if present
drop_cols = [
    col
    for col in ["id", "ID", "OBJECTID", "FID", "GLOBALID"]
    if col in combined.columns
]
if drop_cols:
    combined = combined.drop(columns=drop_cols)

# Process datetime columns
for col in list(combined.columns):
    if (
        np.issubdtype(combined[col].dtype, np.datetime64)
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            dt_s = pd.to_datetime(combined[col], errors="coerce")
            combined[f"{col}_year"] = dt_s.dt.year
            combined[f"{col}_month"] = dt_s.dt.month
            combined[f"{col}_day"] = dt_s.dt.day
            combined[f"{col}_dayofweek"] = dt_s.dt.dayofweek
            combined = combined.drop(columns=[col])
        except Exception:
            pass

# Identify coordinate columns and generate directional/spatial features
coord_candidates = [
    ("LATITUDE", "LONGITUDE"),
    ("LAT", "LON"),
    ("Y", "X"),
    ("latitude", "longitude"),
    ("lat", "lon"),
    ("y", "x"),
]
lat_col, lon_col = None, None
for lat_c, lon_c in coord_candidates:
    if lat_c in combined.columns and lon_c in combined.columns:
        lat_col, lon_col = lat_c, lon_c
        break

if lat_col and lon_col:
    lat_rad = np.radians(combined[lat_col])
    lon_rad = np.radians(combined[lon_col])

    # Spherical / polar coordinates
    combined["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    combined["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    combined["coord_z"] = np.sin(lat_rad)

    # Rotated coordinate axes (capturing shelf slope orientations)
    for angle in [15, 30, 45, 60, 75]:
        theta = np.radians(angle)
        combined[f"rot_{angle}"] = combined[lat_col] * np.cos(theta) + combined[
            lon_col
        ] * np.sin(theta)
        combined[f"rot_ortho_{angle}"] = -combined[lat_col] * np.sin(theta) + combined[
            lon_col
        ] * np.cos(theta)

# Identify categorical features and add frequency encoding
cat_cols = []
for col in list(combined.columns):
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
        combined[col] = combined[col].astype(str)
    elif combined[col].dtype == "bool":
        combined[col] = combined[col].astype(int)

# Separate train and test sets
X = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

# Categorical label encoding for XGBoost/LightGBM
for col in cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[col] = oe.fit_transform(X[[col]]).astype(int)
    X_test[col] = oe.transform(X_test[[col]]).astype(int)
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))

oof_cb = np.zeros(n_train)
test_cb = np.zeros(len(X_test))

oof_xgb = np.zeros(n_train)
test_xgb = np.zeros(len(X_test))

# LightGBM Parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": -1,
    "min_child_samples": 15,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Numeric versions of features for CatBoost / XGBoost
X_num = X.copy()
X_test_num = X_test.copy()
for c in cat_cols:
    X_num[c] = X_num[c].astype(int)
    X_test_num[c] = X_test_num[c].astype(int)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Data splits
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # --- 1. LightGBM ---
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols
    )

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    test_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- 2. CatBoost ---
    if HAS_CATBOOST:
        cb_model = CatBoostRegressor(
            iterations=1800,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=0,
        )
        cb_model.fit(
            X_num.iloc[train_idx],
            y_tr,
            eval_set=(X_num.iloc[val_idx], y_val),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof_cb[val_idx] = cb_model.predict(X_num.iloc[val_idx])
        test_cb += cb_model.predict(X_test_num) / n_splits

    # --- 3. XGBoost ---
    if HAS_XGBOOST:
        xgb_model = XGBRegressor(
            n_estimators=1800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=60,
            eval_metric="rmse",
        )
        xgb_model.fit(
            X_num.iloc[train_idx],
            y_tr,
            eval_set=[(X_num.iloc[val_idx], y_val)],
            verbose=False,
        )
        oof_xgb[val_idx] = xgb_model.predict(X_num.iloc[val_idx])
        test_xgb += xgb_model.predict(X_test_num) / n_splits

# Ensemble weights optimization
models_oof = [oof_lgb]
models_test = [test_lgb]

if HAS_CATBOOST:
    models_oof.append(oof_cb)
    models_test.append(test_cb)

if HAS_XGBOOST:
    models_oof.append(oof_xgb)
    models_test.append(test_xgb)

if len(models_oof) > 1:
    M_oof = np.column_stack(models_oof)
    M_test = np.column_stack(models_test)

    # Non-negative least squares / constrained linear blending
    from scipy.optimize import minimize

    def obj(w):
        w_norm = np.array(w) / np.sum(w)
        pred = M_oof @ w_norm
        return root_mean_squared_error(y, pred)

    init_w = np.ones(len(models_oof)) / len(models_oof)
    bnds = [(0, 1) for _ in range(len(models_oof))]
    res = minimize(obj, init_w, bounds=bnds, method="L-BFGS-B")
    best_w = res.x / np.sum(res.x)

    oof_final = M_oof @ best_w
    test_final = M_test @ best_w
else:
    oof_final = oof_lgb
    test_final = test_lgb

cv_rmse = root_mean_squared_error(y, oof_final)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_final)), "WATERDEPTH": test_final})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
