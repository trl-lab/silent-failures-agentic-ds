import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor

    has_catboost = True
except ImportError:
    has_catboost = False

warnings.filterwarnings("ignore")

# 1. Setup paths and directories
input_dir = "./input"
output_dir = "./working"
submission_dir = "./submission"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Population for whom broadband available, 2019 (%)"

# Identify or create index-based ID column
id_col = None
for col in ["id", "ID", "Id", "row_id"]:
    if col in test_df.columns:
        id_col = col
        break

if id_col is None:
    test_df["id"] = np.arange(len(test_df))
    id_col = "id"
    if id_col not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

y_train = train_df[target_col].astype(float).values
feature_cols = [c for c in train_df.columns if c != target_col and c != id_col]

sanitized_mapping = {}
seen = set()
for c in feature_cols:
    clean = re.sub(r'[^\w]', '_', str(c))
    if clean in seen:
        idx = 1
        while f"{clean}_{idx}" in seen:
            idx += 1
        clean = f"{clean}_{idx}"
    seen.add(clean)
    sanitized_mapping[c] = clean

X_train = train_df[feature_cols].copy().rename(columns=sanitized_mapping)
X_test = test_df[feature_cols].copy().rename(columns=sanitized_mapping)

# 2. Process Feature Engineering strictly preventing data leakage
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

missing_cols = [
    col
    for col in numeric_cols
    if X_train[col].isnull().any() or X_test[col].isnull().any()
]
for col in missing_cols:
    X_train[f"{col}_isna"] = X_train[col].isnull().astype(float)
    X_test[f"{col}_isna"] = X_test[col].isnull().astype(float)

if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

for col in categorical_cols:
    X_train[col] = X_train[col].fillna("Missing").astype(str)
    X_test[col] = X_test[col].fillna("Missing").astype(str)

    freq_map = X_train[col].value_counts(normalize=True).to_dict()
    X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0.0)
    X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0.0)

if len(categorical_cols) > 0:
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[categorical_cols] = ord_encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = ord_encoder.transform(X_test[categorical_cols])

updated_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
skewed_cols = []
for col in updated_numeric_cols:
    if not col.endswith("_isna") and not col.endswith("_freq"):
        skew_val = X_train[col].skew()
        if abs(skew_val) > 1.5 and (X_train[col] >= 0).all():
            skewed_cols.append(col)

for col in skewed_cols:
    X_train[f"{col}_log1p"] = np.log1p(X_train[col].clip(lower=0))
    X_test[f"{col}_log1p"] = np.log1p(X_test[col].clip(lower=0))

variances = X_train[numeric_cols].var()
top_var_cols = variances.nlargest(min(6, len(numeric_cols))).index.tolist()

ratio_count = 0
for i in range(len(top_var_cols)):
    for j in range(i + 1, len(top_var_cols)):
        col1, col2 = top_var_cols[i], top_var_cols[j]
        denom_tr = X_train[col2].replace(0, np.nan)
        denom_te = X_test[col2].replace(0, np.nan)

        ratio_tr = X_train[col1] / (denom_tr.abs() + 1e-5)
        ratio_te = X_test[col1] / (denom_te.abs() + 1e-5)

        X_train[f"ratio_{col1}_{col2}"] = ratio_tr.fillna(0.0)
        X_test[f"ratio_{col1}_{col2}"] = ratio_te.fillna(0.0)
        ratio_count += 1
        if ratio_count >= 10:
            break
    if ratio_count >= 10:
        break

all_num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
n_components = min(5, len(all_num_cols))

# 3. K-Fold Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train.loc[val_idx, "fold"] = fold

final_features = [c for c in X_train.columns if c != "fold"]

folds = X_train["fold"].values
unique_folds = np.unique(folds)
n_splits = len(unique_folds)

X = X_train[final_features]
y = y_train
X_te = X_test[final_features]

# 4. Model Ensemble Cross-Validation
models_dict = {
    "lgb": "LightGBM Regressor",
    "xgb": "XGBoost Regressor",
    "hgb": "HistGradientBoosting Regressor",
    "et": "ExtraTrees Regressor",
}
if has_catboost:
    models_dict["cb"] = "CatBoost Regressor"

oof_preds = {m: np.zeros(len(X_train)) for m in models_dict}
test_preds = {m: np.zeros(len(X_test)) for m in models_dict}

for fold_idx in sorted(unique_folds):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_tr, y_tr = X.iloc[train_mask].copy(), y[train_mask]
    X_va, y_va = X.iloc[val_mask].copy(), y[val_mask]
    X_te_fold = X_te.copy()

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr[all_num_cols])
    X_va_scaled = scaler.transform(X_va[all_num_cols])
    X_te_scaled = scaler.transform(X_te_fold[all_num_cols])

    if n_components > 1:
        pca = PCA(n_components=n_components, random_state=42)
        pca_tr = pca.fit_transform(X_tr_scaled)
        pca_va = pca.transform(X_va_scaled)
        pca_te = pca.transform(X_te_scaled)

        for k in range(n_components):
            X_tr[f"pca_{k}"] = pca_tr[:, k]
            X_va[f"pca_{k}"] = pca_va[:, k]
            X_te_fold[f"pca_{k}"] = pca_te[:, k]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(-1),
    ]
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=lgb_callbacks)
    oof_preds["lgb"][val_mask] = lgb_model.predict(X_va)
    test_preds["lgb"] += lgb_model.predict(X_te_fold) / n_splits

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb"][val_mask] = xgb_model.predict(X_va)
    test_preds["xgb"] += xgb_model.predict(X_te_fold) / n_splits

    # Model 3: HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.025,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42 + fold_idx,
        early_stopping=True,
        n_iter_no_change=30,
    )
    hgb_model.fit(X_tr, y_tr)
    oof_preds["hgb"][val_mask] = hgb_model.predict(X_va)
    test_preds["hgb"] += hgb_model.predict(X_te_fold) / n_splits

    # Model 4: ExtraTrees
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_split=4,
        random_state=42 + fold_idx,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_preds["et"][val_mask] = et_model.predict(X_va)
    test_preds["et"] += et_model.predict(X_te_fold) / n_splits

    # Model 5: CatBoost (if available)
    if has_catboost:
        cb_model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.025,
            depth=6,
            random_seed=42 + fold_idx,
            verbose=0,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds["cb"][val_mask] = cb_model.predict(X_va)
        test_preds["cb"] += cb_model.predict(X_te_fold) / n_splits

# 5. Optimization & Ensembling
model_keys = list(models_dict.keys())
oof_matrix = np.column_stack([oof_preds[m] for m in model_keys])
test_matrix = np.column_stack([test_preds[m] for m in model_keys])


def objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = np.clip(oof_matrix @ w, 0.0, 100.0)
    return np.sqrt(mean_squared_error(y, blend))


n_models = len(model_keys)
init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0)] * n_models
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x / np.sum(opt_res.x)

final_oof = np.clip(oof_matrix @ opt_weights, 0.0, 100.0)
final_test = np.clip(test_matrix @ opt_weights, 0.0, 100.0)

final_score = np.sqrt(mean_squared_error(y, final_oof))

submission_df = pd.DataFrame({id_col: test_df[id_col].values, target_col: final_test})
submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

print(f"Final Validation Score: {final_score}")