import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1. Environment & Reproducibility Setup
# -------------------------------------------------------------------------
np.random.seed(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------------------
# 2. Data Loading & Feature Engineering
# -------------------------------------------------------------------------
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "ISE"
if target_col not in train_raw.columns:
    candidates = [c for c in train_raw.columns if "ise" in c.lower()]
    target_col = candidates[0] if candidates else train_raw.columns[-1]


def engineer_features(df_input: pd.DataFrame, target_name: str) -> pd.DataFrame:
    df = df_input.copy()

    # Standardize column naming except target
    col_map = {
        c: c.strip().upper() for c in df.columns if c.strip().upper() != target_name
    }
    df = df.rename(columns=col_map)

    # Date and temporal features
    date_col = None
    for c in df.columns:
        if "DATE" in c.upper():
            date_col = c
            break

    if date_col is not None:
        try:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed", dayfirst=True)
        except Exception:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
        if parsed_dates.isna().sum() > 0.5 * len(df):
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

        df["day"] = parsed_dates.dt.day.fillna(15).astype(int)
        df["dayofweek"] = parsed_dates.dt.dayofweek.fillna(2).astype(int)
        df["month"] = parsed_dates.dt.month.fillna(6).astype(int)
        df["quarter"] = parsed_dates.dt.quarter.fillna(2).astype(int)
        df["dayofyear"] = parsed_dates.dt.dayofyear.fillna(180).astype(int)
        df["is_month_start"] = parsed_dates.dt.is_month_start.fillna(False).astype(int)
        df["is_month_end"] = parsed_dates.dt.is_month_end.fillna(False).astype(int)

        # Cyclical calendar encodings
        df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 5.0)
        df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 5.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

        df = df.drop(columns=[date_col])

    # Identify international market index feature columns
    excluded = [
        target_name,
        "ID",
        "id",
        "Id",
        "day",
        "dayofweek",
        "month",
        "quarter",
        "dayofyear",
        "is_month_start",
        "is_month_end",
        "sin_dayofweek",
        "cos_dayofweek",
        "sin_month",
        "cos_month",
        "sin_dayofyear",
        "cos_dayofyear",
    ]
    index_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Contemporaneous cross-index statistics (domain financial moments)
    if len(index_cols) > 1:
        idx_vals = df[index_cols].values
        df["global_mean_return"] = np.nanmean(idx_vals, axis=1)
        df["global_median_return"] = np.nanmedian(idx_vals, axis=1)
        df["global_std_return"] = np.nanstd(idx_vals, axis=1)
        df["global_min_return"] = np.nanmin(idx_vals, axis=1)
        df["global_max_return"] = np.nanmax(idx_vals, axis=1)
        df["global_return_range"] = df["global_max_return"] - df["global_min_return"]
        df["global_abs_sum"] = np.nansum(np.abs(idx_vals), axis=1)
        df["global_positive_count"] = np.sum(idx_vals > 0, axis=1)

        diffs = idx_vals - df["global_mean_return"].values[:, None]
        m3 = np.nanmean(diffs**3, axis=1)
        m2 = np.nanmean(diffs**2, axis=1) + 1e-8
        df["global_skew_return"] = m3 / (m2**1.5)

    # Regional composites & inter-market relative return spreads
    cols_set = set(index_cols)
    eu_cols = [
        c for c in index_cols if any(k in c for k in ["DAX", "FTSE", "EU", "MSCE"])
    ]
    if eu_cols:
        df["europe_composite_mean"] = df[eu_cols].mean(axis=1)
        df["europe_composite_std"] = df[eu_cols].std(axis=1).fillna(0)

    us_cols = [c for c in index_cols if any(k in c for k in ["SP", "BOVESPA"])]
    if us_cols:
        df["americas_composite_mean"] = df[us_cols].mean(axis=1)

    em_cols = [c for c in index_cols if any(k in c for k in ["EM", "BOVESPA"])]
    if em_cols:
        df["em_composite_mean"] = df[em_cols].mean(axis=1)

    # Linear inter-market spreads capturing regional risk premia
    if "EM" in cols_set and "SP" in cols_set:
        df["spread_EM_SP"] = df["EM"] - df["SP"]
    if "EM" in cols_set and "EU" in cols_set:
        df["spread_EM_EU"] = df["EM"] - df["EU"]
    if "EU" in cols_set and "DAX" in cols_set:
        df["spread_EU_DAX"] = df["EU"] - df["DAX"]
    if "DAX" in cols_set and "FTSE" in cols_set:
        df["spread_DAX_FTSE"] = df["DAX"] - df["FTSE"]
    if "BOVESPA" in cols_set and "EM" in cols_set:
        df["spread_BOVESPA_EM"] = df["BOVESPA"] - df["EM"]
    if "BOVESPA" in cols_set and "SP" in cols_set:
        df["spread_BOVESPA_SP"] = df["BOVESPA"] - df["SP"]
    if "SP" in cols_set and "EU" in cols_set:
        df["spread_SP_EU"] = df["SP"] - df["EU"]
    if "DAX" in cols_set and "SP" in cols_set:
        df["spread_DAX_SP"] = df["DAX"] - df["SP"]
    if "NIKKEI" in cols_set and "SP" in cols_set:
        df["spread_NIKKEI_SP"] = df["NIKKEI"] - df["SP"]
    if "NIKKEI" in cols_set and "europe_composite_mean" in df.columns:
        df["spread_NIKKEI_EUROP"] = df["NIKKEI"] - df["europe_composite_mean"]

    return df


train_fe = engineer_features(train_raw, target_name=target_col)
test_fe = engineer_features(test_raw, target_name=target_col)

# Align feature columns
feature_cols = [c for c in train_fe.columns if c != target_col]
for col in feature_cols:
    if col not in test_fe.columns:
        test_fe[col] = 0.0

test_fe = test_fe[feature_cols].copy()

# Leak-free imputation using training medians
train_medians = train_fe[feature_cols].median()
train_fe[feature_cols] = train_fe[feature_cols].fillna(train_medians)
test_fe[feature_cols] = test_fe[feature_cols].fillna(train_medians)

# Drop any target NaNs in train
if train_fe[target_col].isna().sum() > 0:
    train_fe = train_fe.dropna(subset=[target_col]).reset_index(drop=True)

# 5-fold cross-validation split
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
train_fe["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(kf.split(train_fe)):
    train_fe.loc[val_idx, "fold"] = fold_idx

n_samples = len(train_fe)
n_test = len(test_fe)
n_features = len(feature_cols)


# -------------------------------------------------------------------------
# 3. Model Architectures & Configurations
# -------------------------------------------------------------------------
def create_linear_model_dict():
    return {
        "LinearRegression": LinearRegression(fit_intercept=True),
        "BayesianRidge": BayesianRidge(max_iter=5000, tol=1e-12),
        "RidgeCV": RidgeCV(alphas=np.logspace(-12, 4, 200)),
        "LassoCV": LassoCV(
            alphas=np.logspace(-12, 2, 200), max_iter=50000, tol=1e-10, cv=5, random_state=42
        ),
        "ElasticNetCV": ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=np.logspace(-12, 2, 100),
            max_iter=50000,
            tol=1e-10,
            cv=5,
            random_state=42,
        ),
        "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=5000, tol=1e-10, alpha=1.0),
        "ARDRegression": ARDRegression(max_iter=5000, tol=1e-10),
    }


# -------------------------------------------------------------------------
# 4. Cross-Validation Training & Inference
# -------------------------------------------------------------------------
sample_models = create_linear_model_dict()
model_names = list(sample_models.keys())
oof_preds = {m: np.zeros(n_samples) for m in model_names}
test_preds_folds = {m: np.zeros((n_splits, n_test)) for m in model_names}
y_true = train_fe[target_col].values

for fold_idx in range(n_splits):
    train_mask = train_fe["fold"] != fold_idx
    val_mask = train_fe["fold"] == fold_idx
    val_indices = np.where(val_mask)[0]

    X_tr_raw = train_fe.loc[train_mask, feature_cols].values
    y_tr = train_fe.loc[train_mask, target_col].values
    X_val_raw = train_fe.loc[val_mask, feature_cols].values
    y_val = train_fe.loc[val_mask, target_col].values
    X_te_raw = test_fe[feature_cols].values

    # Strict isolation: Standardize continuous features fitted ONLY on training split
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te = scaler.transform(X_te_raw)

    models = create_linear_model_dict()

    fold_scores = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val).ravel()
        te_pred = model.predict(X_te).ravel()

        oof_preds[name][val_indices] = val_pred
        test_preds_folds[name][fold_idx] = te_pred

        score = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores.append(f"{name}: {score:.5f}")

    print(f"Fold {fold_idx+1}/{n_splits} Validation RMSE -> " + " | ".join(fold_scores))

# -------------------------------------------------------------------------
# 5. Optimal Blending & Submission Generation
# -------------------------------------------------------------------------
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])

# Automated pruning of suboptimal estimators (RMSE > 2x minimum)
model_cv_scores = {m: np.sqrt(mean_squared_error(y_true, oof_preds[m])) for m in model_names}
min_score = min(model_cv_scores.values())
active_models = [m for m in model_names if model_cv_scores[m] <= 2.0 * min_score]
active_indices = [model_names.index(m) for m in active_models]

A_active = oof_matrix[:, active_indices]
lsq_res = lsq_linear(A_active, y_true, bounds=(0, 1), tol=1e-15)

if lsq_res.success and np.sum(lsq_res.x) > 0:
    active_weights = lsq_res.x / np.sum(lsq_res.x)
else:
    active_weights = np.ones(len(active_models)) / len(active_models)

best_weights = np.zeros(len(model_names))
for act_idx, orig_idx in enumerate(active_indices):
    best_weights[orig_idx] = active_weights[act_idx]

for m, w in zip(model_names, best_weights):
    cv_score = model_cv_scores[m]
    print(f"Model {m:15s} | Weight: {w:8.4f} | CV RMSE: {cv_score:.8f}")

final_oof_pred = np.dot(oof_matrix, best_weights)
final_val_score = np.sqrt(mean_squared_error(y_true, final_oof_pred))

test_pred_matrix = np.zeros((n_test, len(model_names)))
for i, m in enumerate(model_names):
    test_pred_matrix[:, i] = np.mean(test_preds_folds[m], axis=0)

final_test_pred = np.dot(test_pred_matrix, best_weights)

sub_id_col = None
for col in ["id", "ID", "Id"]:
    if col in test_raw.columns:
        sub_id_col = col
        break

if sub_id_col is not None:
    submission_df = pd.DataFrame(
        {sub_id_col: test_raw[sub_id_col], target_col: final_test_pred}
    )
else:
    submission_df = pd.DataFrame(
        {"id": np.arange(len(final_test_pred), dtype=int), target_col: final_test_pred}
    )

submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
