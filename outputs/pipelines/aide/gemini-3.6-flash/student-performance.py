import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature Engineering with Linear Trajectory and Absence Interactions
def create_features(df):
    data = df.copy()

    # Grade statistics and non-linear interactions
    if "G1" in data.columns and "G2" in data.columns:
        data["G1"] = pd.to_numeric(data["G1"], errors="coerce").fillna(0)
        data["G2"] = pd.to_numeric(data["G2"], errors="coerce").fillna(0)
        data["G_mean"] = (data["G1"] + data["G2"]) / 2.0
        data["G_diff"] = data["G2"] - data["G1"]
        data["weighted_G"] = 0.3 * data["G1"] + 0.7 * data["G2"]
        data["G2_sq"] = data["G2"] ** 2
        data["G1_G2_prod"] = data["G1"] * data["G2"]
        data["G_ratio"] = (data["G2"] + 1.0) / (data["G1"] + 1.0)
        data["G1_is_zero"] = (data["G1"] == 0).astype(int)
        data["G2_is_zero"] = (data["G2"] == 0).astype(int)

        # Grade trajectory projections
        data["G_proj_linear"] = 2.0 * data["G2"] - data["G1"]
        data["G_proj_dampened"] = 1.5 * data["G2"] - 0.5 * data["G1"]

    # Parent education aggregates
    if "Medu" in data.columns and "Fedu" in data.columns:
        data["Medu"] = pd.to_numeric(data["Medu"], errors="coerce").fillna(0)
        data["Fedu"] = pd.to_numeric(data["Fedu"], errors="coerce").fillna(0)
        data["par_edu_mean"] = (data["Medu"] + data["Fedu"]) / 2.0
        data["par_edu_max"] = np.maximum(data["Medu"], data["Fedu"])
        data["par_edu_diff"] = data["Medu"] - data["Fedu"]

    # Alcohol consumption features
    if "Dalc" in data.columns and "Walc" in data.columns:
        data["Dalc"] = pd.to_numeric(data["Dalc"], errors="coerce").fillna(0)
        data["Walc"] = pd.to_numeric(data["Walc"], errors="coerce").fillna(0)
        data["alc_sum"] = data["Dalc"] + data["Walc"]
        data["alc_ratio"] = data["Dalc"] / (data["Walc"] + 1e-5)

    # Time and interaction metrics
    if "studytime" in data.columns and "traveltime" in data.columns:
        data["studytime"] = pd.to_numeric(data["studytime"], errors="coerce").fillna(0)
        data["traveltime"] = pd.to_numeric(data["traveltime"], errors="coerce").fillna(
            0
        )
        data["study_travel_ratio"] = data["studytime"] / (data["traveltime"] + 1e-5)

    if "freetime" in data.columns and "goout" in data.columns:
        data["freetime"] = pd.to_numeric(data["freetime"], errors="coerce").fillna(0)
        data["goout"] = pd.to_numeric(data["goout"], errors="coerce").fillna(0)
        data["free_goout_diff"] = data["freetime"] - data["goout"]

    if "failures" in data.columns and "age" in data.columns:
        data["failures"] = pd.to_numeric(data["failures"], errors="coerce").fillna(0)
        data["age"] = pd.to_numeric(data["age"], errors="coerce").fillna(0)
        data["fail_age_ratio"] = data["failures"] / (data["age"] + 1e-5)

    if "absences" in data.columns:
        data["absences"] = pd.to_numeric(data["absences"], errors="coerce").fillna(0)
        data["absences_log"] = np.log1p(data["absences"])
        if "G2" in data.columns:
            data["grade_absence_risk"] = data["absences"] * (20.0 - data["G2"])
        if "studytime" in data.columns:
            data["absences_per_study"] = data["absences"] / (data["studytime"] + 0.5)

    return data


train_feat = create_features(train_df)
test_feat = create_features(test_df)

target_col = "G3"
y = pd.to_numeric(train_feat[target_col], errors="coerce").values
X_train_raw = train_feat.drop(columns=[target_col], errors="ignore")
X_test_raw = test_feat.drop(columns=[target_col], errors="ignore")

# Distinguish numerical and categorical columns
cat_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]

# Handle missing values
for c in num_cols:
    med = X_train_raw[c].median()
    X_train_raw[c] = X_train_raw[c].fillna(med)
    X_test_raw[c] = X_test_raw[c].fillna(med)

for c in cat_cols:
    X_train_raw[c] = X_train_raw[c].astype(str).fillna("missing")
    X_test_raw[c] = X_test_raw[c].astype(str).fillna("missing")

# Encode and scale features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
X_test_cat = encoder.transform(X_test_raw[cat_cols])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_raw[num_cols])
X_test_num = scaler.transform(X_test_raw[num_cols])

X_train_proc = np.hstack([X_train_num, X_train_cat])
X_test_proc = np.hstack([X_test_num, X_test_cat])

# Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

num_models = 5
oof_model_preds = np.zeros((len(train_df), num_models))
test_model_preds = np.zeros((len(test_df), num_models))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc[val_idx], y[val_idx]

    # Model 0: XGBoost
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=30,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Model 1: CatBoost
    cat = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=4,
        random_seed=42 + fold,
        early_stopping_rounds=30,
        verbose=False,
    )
    cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    # Model 2: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Model 3: Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, y_tr)

    # Model 4: Support Vector Regressor (SVR)
    svr = SVR(C=5.0, epsilon=0.1)
    svr.fit(X_tr, y_tr)

    # Store individual predictions
    oof_model_preds[val_idx, 0] = xgb.predict(X_va)
    oof_model_preds[val_idx, 1] = cat.predict(X_va)
    oof_model_preds[val_idx, 2] = lgb_model.predict(X_va)
    oof_model_preds[val_idx, 3] = ridge.predict(X_va)
    oof_model_preds[val_idx, 4] = svr.predict(X_va)

    test_model_preds[:, 0] += xgb.predict(X_test_proc) / kf.n_splits
    test_model_preds[:, 1] += cat.predict(X_test_proc) / kf.n_splits
    test_model_preds[:, 2] += lgb_model.predict(X_test_proc) / kf.n_splits
    test_model_preds[:, 3] += ridge.predict(X_test_proc) / kf.n_splits
    test_model_preds[:, 4] += svr.predict(X_test_proc) / kf.n_splits


# Optimize blending weights on OOF predictions
def ensemble_loss(weights):
    weights = np.array(weights)
    blend = np.dot(oof_model_preds, weights)
    blend_clipped = np.clip(blend, 0, 20)
    return root_mean_squared_error(y, blend_clipped)


initial_weights = [0.2] * num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x

# Final OOF and Test predictions using optimal weights
oof_preds = np.clip(np.dot(oof_model_preds, best_weights), 0, 20)
test_preds = np.clip(np.dot(test_model_preds, best_weights), 0, 20)

# Compute CV RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "G3": test_preds})
submission.to_csv("./working/submission.csv", index=False)
