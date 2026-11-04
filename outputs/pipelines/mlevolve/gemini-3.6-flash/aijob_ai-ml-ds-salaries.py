import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    has_catboost = True
except ImportError:
    has_catboost = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Directories & Data Loading
# ---------------------------------------------------------
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_raw = pd.read_parquet(f"{input_dir}/train.parquet")
test_raw = pd.read_parquet(f"{input_dir}/test.parquet")

if "id" in test_raw.columns:
    test_ids = test_raw["id"].values
else:
    test_ids = np.arange(len(test_raw))
    test_raw["id"] = test_ids

# Split train strictly BEFORE feature engineering to prevent data leakage
tr_raw, va_raw = train_test_split(train_raw, test_size=0.20, random_state=42)
tr_raw = tr_raw.copy().reset_index(drop=True)
va_raw = va_raw.copy().reset_index(drop=True)
test_raw = test_raw.copy().reset_index(drop=True)


# ---------------------------------------------------------
# 2. Domain Feature Extraction
# ---------------------------------------------------------
def extract_domain_features(df):
    df = df.copy()

    exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
    size_map = {"S": 1, "M": 2, "L": 3}
    emp_map = {"PT": 1, "FL": 2, "CT": 3, "FT": 4}

    df["exp_level_num"] = (
        df["experience_level"].map(exp_map).fillna(2).astype(int)
        if "experience_level" in df.columns
        else 2
    )
    df["company_size_num"] = (
        df["company_size"].map(size_map).fillna(2).astype(int)
        if "company_size" in df.columns
        else 2
    )
    df["emp_type_num"] = (
        df["employment_type"].map(emp_map).fillna(4).astype(int)
        if "employment_type" in df.columns
        else 4
    )

    if "work_year" in df.columns:
        df["work_year"] = (
            pd.to_numeric(df["work_year"], errors="coerce").fillna(2023).astype(int)
        )
        df["years_since_2020"] = df["work_year"] - 2020
    else:
        df["work_year"] = 2023
        df["years_since_2020"] = 3

    df["exp_year_interaction"] = df["exp_level_num"] * df["years_since_2020"]

    residence = (
        df["employee_residence"]
        if "employee_residence" in df.columns
        else pd.Series(["US"] * len(df))
    )
    location = (
        df["company_location"]
        if "company_location" in df.columns
        else pd.Series(["US"] * len(df))
    )

    df["is_same_country"] = (residence == location).astype(int)
    df["is_us_residence"] = (residence == "US").astype(int)
    df["is_us_company"] = (location == "US").astype(int)
    df["us_remote_mismatch"] = df["is_us_company"] - df["is_us_residence"]

    high_pay_countries = {"US", "CA", "CH", "GB", "DE", "AU", "NL", "DK", "SE"}
    df["is_high_pay_country"] = location.isin(high_pay_countries).astype(int)

    if "remote_ratio" in df.columns:
        df["remote_ratio"] = pd.to_numeric(df["remote_ratio"], errors="coerce").fillna(
            0
        )
        df["is_fully_remote"] = (df["remote_ratio"] == 100).astype(int)
        df["is_hybrid"] = (df["remote_ratio"] == 50).astype(int)
        df["is_on_site"] = (df["remote_ratio"] == 0).astype(int)
    else:
        df["remote_ratio"] = 0
        df["is_fully_remote"] = 0
        df["is_hybrid"] = 0
        df["is_on_site"] = 1

    if "job_title" in df.columns:
        titles = df["job_title"].astype(str).str.lower()
        df["title_is_senior"] = titles.str.contains(
            r"\bsenior\b|\bsr\b", regex=True
        ).astype(int)
        df["title_is_lead"] = titles.str.contains(
            r"\blead\b|\bprincipal\b|\bhead\b|\bstaff\b|\bchief\b", regex=True
        ).astype(int)
        df["title_is_manager"] = titles.str.contains(
            r"\bmanager\b|\bdirector\b|\bvp\b|\bpresident\b", regex=True
        ).astype(int)
        df["title_is_junior"] = titles.str.contains(
            r"\bjunior\b|\bjr\b|\bintern\b|\bassociate\b|\bentry\b", regex=True
        ).astype(int)

        def categorize_title(t):
            if any(
                k in t
                for k in [
                    "machine learning",
                    "ml",
                    "ai",
                    "deep learning",
                    "computer vision",
                    "nlp",
                    "llm",
                ]
            ):
                return "ml_ai"
            elif any(
                k in t
                for k in [
                    "data scientist",
                    "science",
                    "applied scientist",
                    "research scientist",
                ]
            ):
                return "data_scientist"
            elif any(
                k in t
                for k in [
                    "data engineer",
                    "etl",
                    "big data",
                    "architect",
                    "pipeline",
                    "infrastructure",
                ]
            ):
                return "data_engineer"
            elif any(
                k in t for k in ["analyst", "analytics", "bi", "business intelligence"]
            ):
                return "data_analyst"
            elif any(
                k in t for k in ["manager", "head", "director", "lead", "chief", "vp"]
            ):
                return "management"
            else:
                return "other_tech"

        df["title_category"] = titles.apply(categorize_title)
    else:
        df["title_is_senior"] = 0
        df["title_is_lead"] = 0
        df["title_is_manager"] = 0
        df["title_is_junior"] = 0
        df["title_category"] = "data_scientist"

    return df


tr_df = extract_domain_features(tr_raw)
va_df = extract_domain_features(va_raw)
test_df = extract_domain_features(test_raw)

# Target log transformation
tr_df["target_log"] = np.log1p(tr_df["salary_in_usd"])
va_df["target_log"] = np.log1p(va_df["salary_in_usd"])

# ---------------------------------------------------------
# 3. Frequency & Target Encoding (Strict Isolation)
# ---------------------------------------------------------
cat_cols = [
    "job_title",
    "employee_residence",
    "company_location",
    "title_category",
    "experience_level",
]

# Frequency Encoding based ONLY on train
for col in cat_cols:
    if col in tr_df.columns:
        freq_map = tr_df[col].value_counts().to_dict()
        tr_df[f"{col}_freq"] = tr_df[col].map(freq_map).fillna(0)
        va_df[f"{col}_freq"] = va_df[col].map(freq_map).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

# Out-of-fold target encoding on training split
m_smooth = 10.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    if col in tr_df.columns:
        tr_df[f"{col}_te"] = np.nan

for fold, (trn_idx, val_idx) in enumerate(kf.split(tr_df)):
    tr_fold = tr_df.iloc[trn_idx]
    global_mean_fold = tr_fold["target_log"].mean()
    for col in cat_cols:
        if col in tr_df.columns:
            stats = tr_fold.groupby(col)["target_log"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m_smooth * global_mean_fold) / (
                stats["count"] + m_smooth
            )
            te_map = smooth.to_dict()
            tr_df.iloc[val_idx, tr_df.columns.get_loc(f"{col}_te")] = (
                tr_df.iloc[val_idx][col].map(te_map).fillna(global_mean_fold)
            )

# Full train statistics for validation & test
global_mean_tr = tr_df["target_log"].mean()
for col in cat_cols:
    if col in tr_df.columns:
        stats = tr_df.groupby(col)["target_log"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + m_smooth * global_mean_tr) / (
            stats["count"] + m_smooth
        )
        te_map_full = smooth.to_dict()
        va_df[f"{col}_te"] = va_df[col].map(te_map_full).fillna(global_mean_tr)
        test_df[f"{col}_te"] = test_df[col].map(te_map_full).fillna(global_mean_tr)

# Label encoding for categorical columns
str_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "salary_currency",
    "employee_residence",
    "company_location",
    "company_size",
    "title_category",
]
for col in str_cols:
    if col in tr_df.columns:
        le = LabelEncoder()
        all_cats = pd.concat([tr_df[col], va_df[col], test_df[col]]).astype(str)
        le.fit(all_cats)
        tr_df[f"{col}_le"] = le.transform(tr_df[col].astype(str))
        va_df[f"{col}_le"] = le.transform(va_df[col].astype(str))
        test_df[f"{col}_le"] = le.transform(test_df[col].astype(str))

tr_df.to_parquet(f"{working_dir}/train_processed.parquet", index=False)
va_df.to_parquet(f"{working_dir}/val_processed.parquet", index=False)
test_df.to_parquet(f"{working_dir}/test_processed.parquet", index=False)

# ---------------------------------------------------------
# 4. Model Architecture & Training Setup
# ---------------------------------------------------------
exclude_cols = [
    "id",
    "salary",
    "salary_in_usd",
    "salary_currency",
    "target_log",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
    "experience_level",
    "employment_type",
    "title_category",
]

feature_cols = [c for c in tr_df.columns if c not in exclude_cols]

X_tr = tr_df[feature_cols].fillna(0).values
y_tr_log = tr_df["target_log"].values

X_va = va_df[feature_cols].fillna(0).values
y_va_log = va_df["target_log"].values
y_va_usd = va_df["salary_in_usd"].values

X_te = test_df[feature_cols].fillna(0).values

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)
X_te_scaled = scaler.transform(X_te)

models = {}

models["lgb"] = lgb.LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.025,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
)

models["xgb"] = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.025,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

if has_catboost:
    models["cat"] = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
else:
    models["hist"] = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=30,
    )

models["et"] = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=16,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

models["mlp"] = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    alpha=0.01,
    learning_rate_init=0.005,
    max_iter=250,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42,
)

val_preds_log = {}
test_preds_log = {}

# Train LightGBM
lgb_callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=0),
]
models["lgb"].fit(
    X_tr,
    y_tr_log,
    eval_set=[(X_va, y_va_log)],
    callbacks=lgb_callbacks,
)
val_preds_log["lgb"] = models["lgb"].predict(X_va)
test_preds_log["lgb"] = models["lgb"].predict(X_te)

# Train XGBoost
models["xgb"].fit(
    X_tr,
    y_tr_log,
    eval_set=[(X_va, y_va_log)],
    verbose=False,
)
val_preds_log["xgb"] = models["xgb"].predict(X_va)
test_preds_log["xgb"] = models["xgb"].predict(X_te)

# Train CatBoost / HistGB
if has_catboost:
    models["cat"].fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        verbose=False,
    )
    val_preds_log["cat"] = models["cat"].predict(X_va)
    test_preds_log["cat"] = models["cat"].predict(X_te)
else:
    models["hist"].fit(X_tr, y_tr_log)
    val_preds_log["hist"] = models["hist"].predict(X_va)
    test_preds_log["hist"] = models["hist"].predict(X_te)

# Train ExtraTrees
models["et"].fit(X_tr, y_tr_log)
val_preds_log["et"] = models["et"].predict(X_va)
test_preds_log["et"] = models["et"].predict(X_te)

# Train MLP
models["mlp"].fit(X_tr_scaled, y_tr_log)
val_preds_log["mlp"] = models["mlp"].predict(X_va_scaled)
test_preds_log["mlp"] = models["mlp"].predict(X_te_scaled)

# ---------------------------------------------------------
# 5. Non-Negative RMSE Ensemble Optimization
# ---------------------------------------------------------
model_names = list(val_preds_log.keys())
V_mat = np.column_stack([val_preds_log[m] for m in model_names])
T_mat = np.column_stack([test_preds_log[m] for m in model_names])


def loss_func(weights):
    weights = weights / np.sum(weights)
    pred_log = V_mat @ weights
    pred_usd = np.maximum(np.expm1(pred_log), 0)
    return np.sqrt(mean_squared_error(y_va_usd, pred_usd))


init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1) for _ in range(len(model_names))]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = res.x / np.sum(res.x)

# ---------------------------------------------------------
# 6. Evaluation, Logging & Submission Output
# ---------------------------------------------------------
joblib.dump(models, f"{working_dir}/base_models.joblib")
np.save(f"{working_dir}/blend_weights.npy", opt_weights)

print("--- Individual Model Performance (USD RMSE) ---")
for idx, name in enumerate(model_names):
    model_val_log = V_mat[:, idx]
    model_val_usd = np.maximum(np.expm1(model_val_log), 0)
    model_rmse = np.sqrt(mean_squared_error(y_va_usd, model_val_usd))
    weight = opt_weights[idx]
    print(
        f"Model: {name:<6} | Blend Weight: {weight:.4f} | Validation RMSE: ${model_rmse:,.2f}"
    )

ensemble_val_log = V_mat @ opt_weights
ensemble_val_usd = np.maximum(np.expm1(ensemble_val_log), 0)
final_val_rmse = np.sqrt(mean_squared_error(y_va_usd, ensemble_val_usd))

print("------------------------------------------------")
print(f"Ensemble USD RMSE: ${final_val_rmse:,.2f}")

ensemble_test_log = T_mat @ opt_weights
ensemble_test_usd = np.maximum(np.expm1(ensemble_test_log), 0)

submission_path = f"{submission_dir}/submission.csv"
submission_df = pd.DataFrame(
    {
        "id": test_ids,
        "salary_in_usd": ensemble_test_usd,
    }
)

submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert (
    len(submission_df) == T_mat.shape[0]
), f"Row count mismatch: expected {T_mat.shape[0]}, got {len(submission_df)}"
assert (
    not submission_df["salary_in_usd"].isna().any()
), "Submission contains NaN values!"
assert (
    submission_df["salary_in_usd"] >= 0
).all(), "Submission contains negative values!"

print(f"Final Validation Score: {final_val_rmse}")
