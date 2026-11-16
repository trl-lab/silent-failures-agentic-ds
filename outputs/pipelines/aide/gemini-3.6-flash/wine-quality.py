import os
import numpy as np
import pandas as pd
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Suppress numpy warnings
np.seterr(all="ignore")

# Set random seed
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature Engineering Function
def create_domain_features(df):
    data = df.copy()

    # Binary encode color if present
    if "color" in data.columns:
        data["color_code"] = (data["color"] == "white").astype(int)
        data = data.drop(columns=["color"])

    # Acidity features
    data["total_acidity"] = (
        data["fixed_acidity"] + data["volatile_acidity"] + data["citric_acid"]
    )
    data["bound_sulfur_dioxide"] = (
        data["total_sulfur_dioxide"] - data["free_sulfur_dioxide"]
    )
    data["free_to_total_so2"] = data["free_sulfur_dioxide"] / (
        data["total_sulfur_dioxide"] + 1e-5
    )
    data["volatile_to_fixed_acidity"] = data["volatile_acidity"] / (
        data["fixed_acidity"] + 1e-5
    )
    data["acid_to_pH"] = data["total_acidity"] * data["pH"]

    # Alcohol and Density ratios
    data["sugar_to_alcohol"] = data["residual_sugar"] / (data["alcohol"] + 1e-5)
    data["alcohol_to_density"] = data["alcohol"] / (data["density"] + 1e-5)
    data["sulphates_to_chlorides"] = data["sulphates"] / (data["chlorides"] + 1e-5)
    data["density_sugar_ratio"] = data["density"] / (data["residual_sugar"] + 1e-5)

    # Skewness transformations
    skewed_cols = [
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "sulphates",
    ]
    for col in skewed_cols:
        if col in data.columns:
            data[f"log_{col}"] = np.log1p(np.maximum(0, data[col]))

    return data


# Process raw features
y = train_df["quality"].values
X_train_raw = create_domain_features(train_df.drop(columns=["quality"]))
X_test_raw = create_domain_features(test_df)

# Stratified K-Fold setup using ordinal target values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y)):
    X_tr = X_train_raw.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train_raw.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test_raw.copy()

    # Imputation and Scaling per fold to prevent leakage
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)
    X_te_imp = imputer.transform(X_te)

    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_va_scaled = scaler.transform(X_va_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    # KMeans clustering feature generation
    kmeans = KMeans(n_clusters=5, random_state=SEED + fold, n_init=10)
    cluster_tr = kmeans.fit_transform(X_tr_scaled)
    cluster_va = kmeans.transform(X_va_scaled)
    cluster_te = kmeans.transform(X_te_scaled)

    for i in range(5):
        X_tr[f"dist_cluster_{i}"] = cluster_tr[:, i]
        X_va[f"dist_cluster_{i}"] = cluster_va[:, i]
        X_te[f"dist_cluster_{i}"] = cluster_te[:, i]

    # 1. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_te) / 5.0

    # 2. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        early_stopping_rounds=50,
        eval_metric="rmse",
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_te) / 5.0

    # 3. CatBoost Regressor
    cat_model = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        subsample=0.8,
        random_seed=SEED + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[val_idx] = cat_model.predict(X_va)
    test_cat += cat_model.predict(X_te) / 5.0

    # 4. ExtraTrees Regressor
    et_model = ExtraTreesRegressor(
        n_estimators=300, max_depth=12, random_state=SEED + fold, n_jobs=-1
    )
    et_model.fit(X_tr_imp, y_tr)
    oof_et[val_idx] = et_model.predict(X_va_imp)
    test_et += et_model.predict(X_te_imp) / 5.0


# Optimize blending weights via SLSQP
def rmse_obj(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_et
    return np.sqrt(mean_squared_error(y, blend))


init_weights = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
res = minimize(rmse_obj, init_weights, bounds=bounds, method="SLSQP")
opt_weights = res.x / np.sum(res.x)

# Compute final Out-Of-Fold RMSE
oof_blend = (
    opt_weights[0] * oof_lgb
    + opt_weights[1] * oof_xgb
    + opt_weights[2] * oof_cat
    + opt_weights[3] * oof_et
)
final_cv_rmse = np.sqrt(mean_squared_error(y, oof_blend))

# Compute Test Predictions Blend
test_preds = (
    opt_weights[0] * test_lgb
    + opt_weights[1] * test_xgb
    + opt_weights[2] * test_cat
    + opt_weights[3] * test_et
)

# Save submission file
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "quality": test_preds})
sub.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSE: {final_cv_rmse:.5f}")
