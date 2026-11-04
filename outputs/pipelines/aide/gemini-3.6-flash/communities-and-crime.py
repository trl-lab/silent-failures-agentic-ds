import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "ViolentCrimesPerPop"
drop_cols = ["state", "county", "community", "communityname", "fold", target_col]
feature_cols = [c for c in train_df.columns if c not in drop_cols]


# 2. Feature Engineering Function
def create_features(df, features):
    data = df[features].copy()

    # Coerce all to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)

    # Row missingness count
    data["missing_count"] = data.isna().sum(axis=1).astype(np.float32)

    # Socio-economic interaction features
    eps = 1e-5
    if "medFamInc" in data.columns and "medIncome" in data.columns:
        data["fam_to_med_inc"] = data["medFamInc"] / (data["medIncome"] + eps)
    if "perCapInc" in data.columns and "medIncome" in data.columns:
        data["percap_to_med_inc"] = data["perCapInc"] / (data["medIncome"] + eps)
    if "PctPopUnderPov" in data.columns and "PctUnemployed" in data.columns:
        data["pov_unemp_idx"] = data["PctPopUnderPov"] * data["PctUnemployed"]
    if "PctFam2Par" in data.columns and "PctKids2Par" in data.columns:
        data["family_stability"] = data["PctFam2Par"] * data["PctKids2Par"]
    if "PctVacantBoarded" in data.columns and "PctVacMore6Mos" in data.columns:
        data["blight_index"] = data["PctVacantBoarded"] * data["PctVacMore6Mos"]
    if "racePctWhite" in data.columns and "racepctblack" in data.columns:
        data["white_black_ratio"] = data["racePctWhite"] / (data["racepctblack"] + eps)

    # Aggregate stats over income-related continuous features if present
    inc_cols = [
        c
        for c in [
            "medIncome",
            "medFamInc",
            "perCapInc",
            "whitePerCap",
            "blackPerCap",
            "HispPerCap",
        ]
        if c in data.columns
    ]
    if inc_cols:
        data["inc_mean"] = data[inc_cols].mean(axis=1)
        data["inc_std"] = data[inc_cols].std(axis=1)

    return data


X = create_features(train_df, feature_cols)
y = pd.to_numeric(train_df[target_col], errors="coerce").values.astype(np.float32)
X_test = create_features(test_df, feature_cols)

# Ensure columns alignment
all_cols = X.columns.tolist()
X_test = X_test[all_cols]

# 3. Model Setup & 5-Fold Stacking Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Out-of-fold and test prediction containers for base models
base_model_names = ["lgb", "xgb", "cat", "et"]
oof_preds = {name: np.zeros(len(X)) for name in base_model_names}
test_preds = {name: np.zeros(len(X_test)) for name in base_model_names}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # 3.1 LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds["lgb"][val_idx] = lgb_model.predict(X_va)
    test_preds["lgb"] += lgb_model.predict(X_test) / n_splits

    # 3.2 XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        random_state=SEED,
        verbosity=0,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb"][val_idx] = xgb_model.predict(X_va)
    test_preds["xgb"] += xgb_model.predict(X_test) / n_splits

    # 3.3 CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_preds["cat"][val_idx] = cat_model.predict(X_va)
    test_preds["cat"] += cat_model.predict(X_test) / n_splits

    # 3.4 ExtraTrees (Imputed & Scaled)
    et_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ExtraTreesRegressor(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
        ),
    )
    et_pipeline.fit(X_tr, y_tr)
    oof_preds["et"][val_idx] = et_pipeline.predict(X_va)
    test_preds["et"] += et_pipeline.predict(X_test) / n_splits

# 4. Stacking Meta-Learner (Ridge Regression)
oof_matrix = np.column_stack([oof_preds[name] for name in base_model_names])
test_matrix = np.column_stack([test_preds[name] for name in base_model_names])

meta_model = Ridge(alpha=1.0, random_state=SEED)
meta_oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(oof_matrix, y):
    meta_model.fit(oof_matrix[train_idx], y[train_idx])
    meta_oof[val_idx] = meta_model.predict(oof_matrix[val_idx])

# Fit meta-learner on full OOF predictions
meta_model.fit(oof_matrix, y)
final_oof_preds = meta_model.predict(oof_matrix)
final_test_preds = meta_model.predict(test_matrix)

# Target Domain Bounding [0.0, 1.0]
final_oof_preds = np.clip(final_oof_preds, 0.0, 1.0)
final_test_preds = np.clip(final_test_preds, 0.0, 1.0)

# Calculate final OOF RMSE
cv_rmse = np.sqrt(mean_squared_error(y, final_oof_preds))
print(f"5-Fold Cross-Validation Stacked Ensemble RMSE: {cv_rmse:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "ViolentCrimesPerPop": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
