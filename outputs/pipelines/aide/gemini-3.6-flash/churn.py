import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"

# 2. Map Target Variable
y_raw = train_df[target_col]
if (
    isinstance(y_raw.dtype, pd.CategoricalDtype)
    or y_raw.dtype == object
    or isinstance(y_raw.iloc[0], (str, bool))
):
    mapping = {
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
    }
    y = (
        y_raw.map(
            lambda x: mapping.get(x, 1 if str(x).lower() in ["true", "1", "yes"] else 0)
        )
        .astype(int)
        .values
    )
else:
    y = y_raw.astype(int).values

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# 3. Clean and Encode Categorical Features
# Convert all categorical/object dtypes to string first to prevent pandas Categorical setitem errors
for col in df_all.columns:
    if (
        isinstance(df_all[col].dtype, pd.CategoricalDtype)
        or df_all[col].dtype == object
    ):
        df_all[col] = df_all[col].astype(str)

for col in df_all.columns:
    if df_all[col].dtype == object:
        vals = set(df_all[col].dropna().unique())
        vals_lower = {v.lower() for v in vals}
        if vals_lower.issubset({"yes", "no", "true", "false"}):
            df_all[col] = (
                df_all[col]
                .str.lower()
                .map({"yes": 1, "true": 1, "no": 0, "false": 0})
                .fillna(0)
                .astype(int)
            )

# Frequency encoding & Label encoding for remaining non-numeric columns
for col in df_all.columns:
    if df_all[col].dtype == object:
        freq = df_all[col].value_counts(normalize=True).to_dict()
        df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0)
        le = LabelEncoder()
        df_all[col] = le.fit_transform(df_all[col])


# 4. Domain Feature Engineering
def add_engineered_features(df):
    data = df.copy()

    def get_c(possible_names):
        for name in possible_names:
            if name in data.columns:
                return name
        return None

    day_min = get_c(["total_day_minutes", "total_day_min"])
    day_calls = get_c(["total_day_calls"])
    day_charge = get_c(["total_day_charge"])

    eve_min = get_c(["total_eve_minutes", "total_eve_min"])
    eve_calls = get_c(["total_eve_calls"])
    eve_charge = get_c(["total_eve_charge"])

    night_min = get_c(["total_night_minutes", "total_night_min"])
    night_calls = get_c(["total_night_calls"])
    night_charge = get_c(["total_night_charge"])

    intl_min = get_c(["total_intl_minutes", "total_intl_min"])
    intl_calls = get_c(["total_intl_calls"])
    intl_charge = get_c(["total_intl_charge"])

    cust_serv = get_c(["number_customer_service_calls", "customer_service_calls"])
    vmail_msgs = get_c(["number_vmail_messages", "vmail_messages"])
    acct_len = get_c(["account_length", "account_len"])
    intl_plan = get_c(["international_plan", "intl_plan"])
    vmail_plan = get_c(["voice_mail_plan", "vmail_plan"])

    min_cols = [c for c in [day_min, eve_min, night_min, intl_min] if c]
    call_cols = [c for c in [day_calls, eve_calls, night_calls, intl_calls] if c]
    charge_cols = [c for c in [day_charge, eve_charge, night_charge, intl_charge] if c]

    if min_cols:
        data["total_minutes"] = data[min_cols].sum(axis=1)
    if call_cols:
        data["total_calls"] = data[call_cols].sum(axis=1)
    if charge_cols:
        data["total_charge"] = data[charge_cols].sum(axis=1)

    if "total_minutes" in data and "total_calls" in data:
        data["overall_avg_call_length"] = data["total_minutes"] / (
            data["total_calls"] + 1e-5
        )
    if "total_charge" in data and "total_minutes" in data:
        data["overall_charge_per_min"] = data["total_charge"] / (
            data["total_minutes"] + 1e-5
        )

    if cust_serv:
        data["cust_service_high"] = (data[cust_serv] >= 4).astype(float)
        data["cust_service_med"] = (data[cust_serv] >= 3).astype(float)
        if acct_len:
            data["cust_service_per_month"] = data[cust_serv] / (
                data[acct_len] / 30.0 + 1.0
            )

    for period, m_c, c_c, ch_c in [
        ("day", day_min, day_calls, day_charge),
        ("eve", eve_min, eve_calls, eve_charge),
        ("night", night_min, night_calls, night_charge),
        ("intl", intl_min, intl_calls, intl_charge),
    ]:
        if m_c and c_c:
            data[f"{period}_min_per_call"] = data[m_c] / (data[c_c] + 1e-5)
        if ch_c and m_c:
            data[f"{period}_charge_per_min"] = data[ch_c] / (data[m_c] + 1e-5)
        if m_c and "total_minutes" in data:
            data[f"{period}_min_ratio"] = data[m_c] / (data["total_minutes"] + 1e-5)

    if intl_plan and intl_min:
        data["intl_plan_x_min"] = data[intl_plan] * data[intl_min]
    if intl_plan and intl_calls:
        data["intl_plan_x_calls"] = data[intl_plan] * data[intl_calls]
    if vmail_plan and vmail_msgs:
        data["vmail_plan_x_msgs"] = data[vmail_plan] * data[vmail_msgs]

    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    return data


df_feat = add_engineered_features(df_all)

X_tr = df_feat.iloc[:n_train].reset_index(drop=True)
X_te = df_feat.iloc[n_train:].reset_index(drop=True)

# 5. Stratified 5-Fold Cross Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_hgb = np.zeros(n_train)

preds_lgb = np.zeros(len(X_te))
preds_xgb = np.zeros(len(X_te))
preds_cat = np.zeros(len(X_te))
preds_hgb = np.zeros(len(X_te))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_train_f, y_train_f = X_tr.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val_f)[:, 1]
    preds_lgb += model_lgb.predict_proba(X_te)[:, 1] / skf.n_splits

    # Model 2: XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42 + fold,
        early_stopping_rounds=50,
        tree_method="hist",
    )
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val_f)[:, 1]
    preds_xgb += model_xgb.predict_proba(X_te)[:, 1] / skf.n_splits

    # Model 3: CatBoost (if available)
    if HAS_CATBOOST:
        model_cat = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            eval_metric="AUC",
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=0,
        )
        model_cat.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f))
        oof_cat[val_idx] = model_cat.predict_proba(X_val_f)[:, 1]
        preds_cat += model_cat.predict_proba(X_te)[:, 1] / skf.n_splits

    # Model 4: HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=500, max_depth=5, random_state=42 + fold
    )
    model_hgb.fit(X_train_f, y_train_f)
    oof_hgb[val_idx] = model_hgb.predict_proba(X_val_f)[:, 1]
    preds_hgb += model_hgb.predict_proba(X_te)[:, 1] / skf.n_splits

# 6. Rank-normalize OOF predictions for stable blending
rank_oof_lgb = rankdata(oof_lgb) / len(oof_lgb)
rank_oof_xgb = rankdata(oof_xgb) / len(oof_xgb)
rank_oof_hgb = rankdata(oof_hgb) / len(oof_hgb)

if HAS_CATBOOST:
    rank_oof_cat = rankdata(oof_cat) / len(oof_cat)
    oof_ensemble = (
        0.35 * rank_oof_lgb
        + 0.35 * rank_oof_cat
        + 0.20 * rank_oof_xgb
        + 0.10 * rank_oof_hgb
    )

    rank_preds_lgb = rankdata(preds_lgb) / len(preds_lgb)
    rank_preds_xgb = rankdata(preds_xgb) / len(preds_xgb)
    rank_preds_cat = rankdata(preds_cat) / len(preds_cat)
    rank_preds_hgb = rankdata(preds_hgb) / len(preds_hgb)

    test_ensemble = (
        0.35 * rank_preds_lgb
        + 0.35 * rank_preds_cat
        + 0.20 * rank_preds_xgb
        + 0.10 * rank_preds_hgb
    )
else:
    oof_ensemble = 0.45 * rank_oof_lgb + 0.35 * rank_oof_xgb + 0.20 * rank_oof_hgb

    rank_preds_lgb = rankdata(preds_lgb) / len(preds_lgb)
    rank_preds_xgb = rankdata(preds_xgb) / len(preds_xgb)
    rank_preds_hgb = rankdata(preds_hgb) / len(preds_hgb)

    test_ensemble = (
        0.45 * rank_preds_lgb + 0.35 * rank_preds_xgb + 0.20 * rank_preds_hgb
    )

# Evaluate Validation Performance
final_auc = roc_auc_score(y, oof_ensemble)
print(f"5-Fold Stratified CV Ensembled ROC AUC Score: {final_auc:.5f}")

# Save Submission File
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_ensemble)), "class": test_ensemble})
sub.to_csv("./working/submission.csv", index=False)
