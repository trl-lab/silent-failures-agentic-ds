import json
import os
import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

# Drop date_first_booking if present (severe target leakage)
if "date_first_booking" in train_df.columns:
    train_df = train_df.drop(columns=["date_first_booking"])
if "date_first_booking" in test_df.columns:
    test_df = test_df.drop(columns=["date_first_booking"])

target_col = "country_destination"
train_target = train_df[target_col].copy()

# Fit LabelEncoder on training target
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_target)
target_classes = list(label_encoder.classes_)
num_classes = len(target_classes)

train_df["is_train"] = 1
test_df["is_train"] = 0
test_df[target_col] = "UNKNOWN"

combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 2. Date and Time Feature Engineering
# ---------------------------------------------------------
combined_df["dac"] = pd.to_datetime(
    combined_df["date_account_created"], errors="coerce"
)
combined_df["dac_year"] = combined_df["dac"].dt.year.fillna(-1).astype(int)
combined_df["dac_month"] = combined_df["dac"].dt.month.fillna(-1).astype(int)
combined_df["dac_day"] = combined_df["dac"].dt.day.fillna(-1).astype(int)
combined_df["dac_dayofweek"] = combined_df["dac"].dt.dayofweek.fillna(-1).astype(int)
combined_df["dac_dayofyear"] = combined_df["dac"].dt.dayofyear.fillna(-1).astype(int)
combined_df["dac_quarter"] = combined_df["dac"].dt.quarter.fillna(-1).astype(int)
combined_df["dac_is_weekend"] = combined_df["dac_dayofweek"].isin([5, 6]).astype(int)
combined_df["dac_week"] = (
    combined_df["dac"].dt.isocalendar().week.fillna(-1).astype(int)
)

# Parse timestamp_first_active (YYYYMMDDHHMMSS)
tfa_str = combined_df["timestamp_first_active"].astype(str).str.zfill(14)
combined_df["tfa"] = pd.to_datetime(tfa_str, format="%Y%m%d%H%M%S", errors="coerce")

combined_df["tfa_year"] = combined_df["tfa"].dt.year.fillna(-1).astype(int)
combined_df["tfa_month"] = combined_df["tfa"].dt.month.fillna(-1).astype(int)
combined_df["tfa_day"] = combined_df["tfa"].dt.day.fillna(-1).astype(int)
combined_df["tfa_hour"] = combined_df["tfa"].dt.hour.fillna(-1).astype(int)
combined_df["tfa_minute"] = combined_df["tfa"].dt.minute.fillna(-1).astype(int)
combined_df["tfa_dayofweek"] = combined_df["tfa"].dt.dayofweek.fillna(-1).astype(int)
combined_df["tfa_dayofyear"] = combined_df["tfa"].dt.dayofyear.fillna(-1).astype(int)
combined_df["tfa_quarter"] = combined_df["tfa"].dt.quarter.fillna(-1).astype(int)
combined_df["tfa_is_weekend"] = combined_df["tfa_dayofweek"].isin([5, 6]).astype(int)

# Cyclical transformations
combined_df["sin_tfa_hour"] = np.sin(2 * np.pi * combined_df["tfa_hour"] / 24.0)
combined_df["cos_tfa_hour"] = np.cos(2 * np.pi * combined_df["tfa_hour"] / 24.0)
combined_df["sin_tfa_month"] = np.sin(2 * np.pi * combined_df["tfa_month"] / 12.0)
combined_df["cos_tfa_month"] = np.cos(2 * np.pi * combined_df["tfa_month"] / 12.0)
combined_df["sin_tfa_dayofweek"] = np.sin(
    2 * np.pi * combined_df["tfa_dayofweek"] / 7.0
)
combined_df["cos_tfa_dayofweek"] = np.cos(
    2 * np.pi * combined_df["tfa_dayofweek"] / 7.0
)

# Lag between first active and account creation
time_diff = (combined_df["dac"] - combined_df["tfa"]).dt.total_seconds()
combined_df["time_lag_seconds"] = time_diff.fillna(-999)
combined_df["time_lag_days"] = (time_diff / 86400.0).fillna(-999)
combined_df["is_same_day_active_account"] = (
    (combined_df["dac_year"] == combined_df["tfa_year"])
    & (combined_df["dac_month"] == combined_df["tfa_month"])
    & (combined_df["dac_day"] == combined_df["tfa_day"])
).astype(int)

min_date = combined_df["dac"].min()
combined_df["days_since_start"] = (
    (combined_df["dac"] - min_date).dt.days.fillna(-1).astype(int)
)

# Platform growth and vintage metrics
launch_date = pd.Timestamp("2010-01-01")
combined_df["months_since_launch"] = (
    (combined_df["dac"].dt.year - 2010) * 12 + combined_df["dac"].dt.month
).fillna(-1).astype(int)
combined_df["days_since_launch"] = (
    (combined_df["dac"] - launch_date).dt.days.fillna(-1).astype(int)
)

# Enhanced temporal seasonality indicators
combined_df["dac_is_month_end"] = (
    combined_df["dac"].dt.is_month_end.fillna(False).astype(int)
)
combined_df["dac_is_month_start"] = (
    combined_df["dac"].dt.is_month_start.fillna(False).astype(int)
)
combined_df["dac_is_quarter_end"] = (
    combined_df["dac"].dt.is_quarter_end.fillna(False).astype(int)
)
combined_df["dac_is_quarter_start"] = (
    combined_df["dac"].dt.is_quarter_start.fillna(False).astype(int)
)
combined_df["dac_is_summer"] = combined_df["dac_month"].isin([6, 7, 8]).astype(int)
combined_df["dac_is_vacation_season"] = (
    combined_df["dac_month"].isin([6, 7, 8, 12]).astype(int)
)
combined_df["tfa_is_month_end"] = (
    combined_df["tfa"].dt.is_month_end.fillna(False).astype(int)
)
combined_df["tfa_is_summer"] = combined_df["tfa_month"].isin([6, 7, 8]).astype(int)
combined_df["tfa_is_vacation_season"] = (
    combined_df["tfa_month"].isin([6, 7, 8, 12]).astype(int)
)

# ---------------------------------------------------------
# 3. Demographic & Age Feature Engineering
# ---------------------------------------------------------
raw_age = combined_df["age"].copy()
corrected_age = raw_age.copy()
is_birth_year = (corrected_age >= 1900) & (corrected_age <= 2005)
corrected_age.loc[is_birth_year] = (
    combined_df.loc[is_birth_year, "tfa_year"] - corrected_age.loc[is_birth_year]
)

invalid_age = (corrected_age < 14) | (corrected_age > 100) | corrected_age.isna()
corrected_age.loc[invalid_age] = np.nan

combined_df["clean_age"] = corrected_age
combined_df["age_is_missing"] = invalid_age.astype(int)

age_bins = [-1, 18, 25, 35, 45, 55, 65, 105]
age_labels = [
    "age_u18",
    "age_18_25",
    "age_25_35",
    "age_35_45",
    "age_45_55",
    "age_55_65",
    "age_65p",
]
combined_df["age_bracket"] = (
    pd.cut(combined_df["clean_age"], bins=age_bins, labels=age_labels)
    .astype(str)
    .fillna("missing")
)

combined_df["gender_clean"] = (
    combined_df["gender"]
    .fillna("-unknown-")
    .replace({"-unknown-": "UNKNOWN", "OTHER": "OTHER"})
)

# Profile completeness indicators
combined_df["profile_has_age"] = (1 - combined_df["age_is_missing"]).astype(int)
combined_df["profile_has_gender"] = (
    ~combined_df["gender_clean"].isin(["UNKNOWN", "-unknown-", "OTHER"])
).astype(int)
combined_df["profile_has_browser"] = (
    ~combined_df["first_browser"].isin(["-unknown-", "Unknown", np.nan])
).astype(int)
combined_df["profile_has_affiliate"] = (
    ~combined_df["first_affiliate_tracked"].isin(["untracked", np.nan])
).astype(int)
combined_df["profile_completeness_score"] = (
    combined_df["profile_has_age"]
    + combined_df["profile_has_gender"]
    + combined_df["profile_has_browser"]
    + combined_df["profile_has_affiliate"]
)


# ---------------------------------------------------------
# 4. Device, Browser, and Channel Family Categorizations
# ---------------------------------------------------------
def categorize_os(device):
    d = str(device).lower()
    if "mac" in d or "iphone" in d or "ipad" in d:
        return "Apple"
    elif "windows" in d:
        return "Windows"
    elif "android" in d:
        return "Android"
    elif "linux" in d:
        return "Linux"
    else:
        return "Other"


def categorize_device_class(device):
    d = str(device).lower()
    if "desktop" in d:
        return "Desktop"
    elif "phone" in d or "iphone" in d:
        return "Phone"
    elif "pad" in d or "tablet" in d:
        return "Tablet"
    else:
        return "Other"


def categorize_browser(browser):
    b = str(browser).lower()
    if "chrome" in b:
        return "Chrome"
    elif "safari" in b:
        return "Safari"
    elif "firefox" in b:
        return "Firefox"
    elif "ie" in b or "internet explorer" in b or "edge" in b:
        return "IE_Edge"
    elif "android" in b:
        return "Android_Browser"
    elif "opera" in b:
        return "Opera"
    elif "-unknown-" in b:
        return "Unknown"
    else:
        return "Other"


combined_df["os_family"] = combined_df["first_device_type"].apply(categorize_os)
combined_df["device_family"] = combined_df["first_device_type"].apply(
    categorize_device_class
)
combined_df["browser_family"] = combined_df["first_browser"].apply(categorize_browser)
combined_df["is_english_lang"] = (combined_df["language"] == "en").astype(int)

# Domain-specific language-destination affinity indicators
combined_df["lang_aff_FR"] = (combined_df["language"] == "fr").astype(int)
combined_df["lang_aff_ES"] = (combined_df["language"] == "es").astype(int)
combined_df["lang_aff_DE"] = (combined_df["language"] == "de").astype(int)
combined_df["lang_aff_IT"] = (combined_df["language"] == "it").astype(int)
combined_df["lang_aff_PT"] = (combined_df["language"] == "pt").astype(int)
combined_df["lang_aff_NL"] = (combined_df["language"] == "nl").astype(int)
combined_df["lang_aff_EN"] = (combined_df["language"] == "en").astype(int)
combined_df["lang_aff_US_GB_CA_AU"] = (combined_df["language"] == "en").astype(
    int
)
combined_df["lang_is_european"] = (
    combined_df["language"]
    .isin(
        [
            "fr",
            "es",
            "de",
            "it",
            "pt",
            "nl",
            "el",
            "sv",
            "da",
            "no",
            "fi",
            "pl",
            "cs",
            "hu",
        ]
    )
    .astype(int)
)
combined_df["lang_is_asian"] = (
    combined_df["language"].isin(["zh", "ja", "ko", "th", "id", "vi"]).astype(int)
)

# ---------------------------------------------------------
# 5. Multi-way Categorical Interaction Features
# ---------------------------------------------------------
combined_df["first_affiliate_tracked"] = combined_df["first_affiliate_tracked"].fillna(
    "untracked"
)
combined_df["affiliate_channel"] = combined_df["affiliate_channel"].fillna("UNKNOWN")
combined_df["affiliate_provider"] = combined_df["affiliate_provider"].fillna("UNKNOWN")
combined_df["signup_app"] = combined_df["signup_app"].fillna("UNKNOWN")
combined_df["signup_method"] = combined_df["signup_method"].fillna("UNKNOWN")
combined_df["signup_flow"] = combined_df["signup_flow"].fillna(-1).astype(str)

combined_df["channel_provider"] = (
    combined_df["affiliate_channel"] + "_" + combined_df["affiliate_provider"]
)
combined_df["device_browser"] = (
    combined_df["first_device_type"] + "_" + combined_df["first_browser"]
)
combined_df["app_device"] = (
    combined_df["signup_app"] + "_" + combined_df["first_device_type"]
)
combined_df["app_method"] = (
    combined_df["signup_app"] + "_" + combined_df["signup_method"]
)
combined_df["flow_app"] = combined_df["signup_flow"] + "_" + combined_df["signup_app"]
combined_df["lang_affiliate"] = (
    combined_df["language"] + "_" + combined_df["affiliate_channel"]
)
combined_df["os_browser"] = (
    combined_df["os_family"] + "_" + combined_df["browser_family"]
)
combined_df["gender_age_bracket"] = (
    combined_df["gender_clean"] + "_" + combined_df["age_bracket"]
)
combined_df["language_signup_method"] = (
    combined_df["language"] + "_" + combined_df["signup_method"]
)
combined_df["channel_app"] = (
    combined_df["affiliate_channel"] + "_" + combined_df["signup_app"]
)

# ---------------------------------------------------------
# 6. Separate Train & Test to Prevent Information Leakage
# ---------------------------------------------------------
train_mask = combined_df["is_train"] == 1
test_mask = combined_df["is_train"] == 0

train_processed = combined_df[train_mask].copy().reset_index(drop=True)
test_processed = combined_df[test_mask].copy().reset_index(drop=True)

# Train-only statistics for imputation and aggregations
median_age = train_processed["clean_age"].median()
median_lag = train_processed["time_lag_seconds"].median()
train_processed["clean_age_imputed"] = train_processed["clean_age"].fillna(median_age)
test_processed["clean_age_imputed"] = test_processed["clean_age"].fillna(median_age)

agg_groups = [
    "language",
    "first_device_type",
    "signup_app",
    "affiliate_channel",
    "affiliate_provider",
]
for grp_col in agg_groups:
    grp_stats = (
        train_processed.groupby(grp_col)
        .agg(
            age_mean=("clean_age_imputed", "mean"),
            age_std=("clean_age_imputed", "std"),
            lag_mean=("time_lag_seconds", "mean"),
            lag_std=("time_lag_seconds", "std"),
        )
        .reset_index()
    )
    grp_stats["age_std"] = grp_stats["age_std"].fillna(0)
    grp_stats["lag_std"] = grp_stats["lag_std"].fillna(0)

    col_age_mean = f"grp_{grp_col}_age_mean"
    col_age_std = f"grp_{grp_col}_age_std"
    col_lag_mean = f"grp_{grp_col}_lag_mean"
    col_lag_std = f"grp_{grp_col}_lag_std"

    grp_stats = grp_stats.rename(
        columns={
            "age_mean": col_age_mean,
            "age_std": col_age_std,
            "lag_mean": col_lag_mean,
            "lag_std": col_lag_std,
        }
    )

    train_processed = train_processed.merge(grp_stats, on=grp_col, how="left")
    test_processed = test_processed.merge(grp_stats, on=grp_col, how="left")

    train_processed[col_age_mean] = train_processed[col_age_mean].fillna(median_age)
    train_processed[col_age_std] = train_processed[col_age_std].fillna(0)
    test_processed[col_age_mean] = test_processed[col_age_mean].fillna(median_age)
    test_processed[col_age_std] = test_processed[col_age_std].fillna(0)

    train_processed[col_lag_mean] = train_processed[col_lag_mean].fillna(median_lag)
    train_processed[col_lag_std] = train_processed[col_lag_std].fillna(0)
    test_processed[col_lag_mean] = test_processed[col_lag_mean].fillna(median_lag)
    test_processed[col_lag_std] = test_processed[col_lag_std].fillna(0)

# ---------------------------------------------------------
# 7. Frequency Encoding (Fit strictly on Train)
# ---------------------------------------------------------
freq_cols = [
    "first_browser",
    "first_device_type",
    "language",
    "affiliate_provider",
    "affiliate_channel",
    "first_affiliate_tracked",
    "signup_app",
    "signup_method",
    "channel_provider",
    "device_browser",
    "app_device",
    "os_browser",
    "gender_age_bracket",
    "language_signup_method",
    "channel_app",
]

train_len = len(train_processed)
for col in freq_cols:
    freq_map = train_processed[col].value_counts() / train_len
    train_processed[f"{col}_freq"] = (
        train_processed[col].map(freq_map).fillna(0).astype(np.float32)
    )
    test_processed[f"{col}_freq"] = (
        test_processed[col].map(freq_map).fillna(0).astype(np.float32)
    )

# ---------------------------------------------------------
# 8. Categorical Label Encoding
# ---------------------------------------------------------
categorical_features = [
    "gender_clean",
    "signup_method",
    "signup_flow",
    "language",
    "affiliate_channel",
    "affiliate_provider",
    "first_affiliate_tracked",
    "signup_app",
    "first_device_type",
    "first_browser",
    "os_family",
    "device_family",
    "browser_family",
    "age_bracket",
    "channel_provider",
    "device_browser",
    "app_device",
    "app_method",
    "flow_app",
    "lang_affiliate",
    "os_browser",
    "gender_age_bracket",
    "language_signup_method",
    "channel_app",
]

for col in categorical_features:
    le = LabelEncoder()
    train_unique = train_processed[col].astype(str).unique().tolist()
    if "UNKNOWN_VAL" not in train_unique:
        train_unique.append("UNKNOWN_VAL")
    le.fit(train_unique)

    train_vals = train_processed[col].astype(str)
    train_processed[col] = le.transform(train_vals)

    test_vals = test_processed[col].astype(str)
    test_vals_clean = test_vals.map(lambda s: s if s in le.classes_ else "UNKNOWN_VAL")
    test_processed[col] = le.transform(test_vals_clean)

# ---------------------------------------------------------
# 9. Cross-Validation Splits and Out-of-Fold Multi-Class Target Encoding
# ---------------------------------------------------------
train_processed["target_encoded"] = y_train_encoded
train_processed["fold"] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, y_train_encoded)
):
    train_processed.loc[val_idx, "fold"] = fold

# High-cardinality columns for multi-class target encoding
te_cols = [
    "language",
    "first_browser",
    "first_device_type",
    "affiliate_provider",
    "channel_provider",
    "device_browser",
    "gender_age_bracket",
    "language_signup_method",
    "channel_app",
]

smooth_m = 10.0
global_priors = np.bincount(y_train_encoded, minlength=num_classes) / len(
    y_train_encoded
)

for col in te_cols:
    for c in range(num_classes):
        train_processed[f"te_{col}_class_{c}"] = np.float32(global_priors[c])
        test_processed[f"te_{col}_class_{c}"] = np.float32(global_priors[c])

for fold in range(5):
    trn_idx = train_processed["fold"] != fold
    val_idx = train_processed["fold"] == fold

    df_trn = train_processed.loc[trn_idx]
    for col in te_cols:
        grp = (
            df_trn.groupby([col, "target_encoded"]).size().unstack(fill_value=0)
        )
        for c in range(num_classes):
            if c not in grp.columns:
                grp[c] = 0
        cat_counts = df_trn[col].value_counts()
        for c in range(num_classes):
            smooth_te = (grp[c] + smooth_m * global_priors[c]) / (
                cat_counts + smooth_m
            )
            val_mapped = (
                train_processed.loc[val_idx, col]
                .map(smooth_te)
                .fillna(global_priors[c])
                .astype(np.float32)
            )
            train_processed.loc[val_idx, f"te_{col}_class_{c}"] = val_mapped

for col in te_cols:
    grp_full = (
        train_processed.groupby([col, "target_encoded"])
        .size()
        .unstack(fill_value=0)
    )
    for c in range(num_classes):
        if c not in grp_full.columns:
            grp_full[c] = 0
    full_counts = train_processed[col].value_counts()
    for c in range(num_classes):
        smooth_te_full = (grp_full[c] + smooth_m * global_priors[c]) / (
            full_counts + smooth_m
        )
        test_mapped = (
            test_processed[col]
            .map(smooth_te_full)
            .fillna(global_priors[c])
            .astype(np.float32)
        )
        test_processed[f"te_{col}_class_{c}"] = test_mapped

drop_columns = [
    "id",
    "gender",
    "date_account_created",
    "timestamp_first_active",
    "dac",
    "tfa",
    "is_train",
    "country_destination",
    "target_encoded",
    "fold",
]
feature_cols = [
    c
    for c in train_processed.columns
    if c not in drop_columns and train_processed[c].dtype != "object"
]


# ---------------------------------------------------------
# 10. Metric Definition: NDCG@5
# ---------------------------------------------------------
def compute_ndcg5(y_true, y_pred_probs):
    """Computes exact NDCG@5 for multi-class ranking predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)

    top5_preds = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :5]
    discounts = 1.0 / np.log2(np.arange(2, 7))

    matches = top5_preds == y_true[:, None]
    user_ndcg = (matches * discounts).sum(axis=1)
    return float(np.mean(user_ndcg))


# ---------------------------------------------------------
# 11. Model Training & Out-of-Fold Evaluation
# ---------------------------------------------------------
n_samples = len(train_processed)
n_test = len(test_processed)
n_folds = 5

lgb_params_a = {
    "objective": "multiclass",
    "num_class": num_classes,
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 25,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

lgb_params_b = {
    "objective": "multiclass",
    "num_class": num_classes,
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 1.5,
    "reg_lambda": 5.0,
    "n_jobs": -1,
    "random_state": SEED + 1,
    "verbose": -1,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "learning_rate": 0.04,
    "max_depth": 6,
    "min_child_weight": 5,
    "colsample_bytree": 0.75,
    "subsample": 0.85,
    "reg_alpha": 1.5,
    "reg_lambda": 4.0,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}

oof_lgb_a_probs = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_lgb_b_probs = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_xgb_probs = np.zeros((n_samples, num_classes), dtype=np.float32)

test_lgb_a_probs = np.zeros((n_test, num_classes), dtype=np.float32)
test_lgb_b_probs = np.zeros((n_test, num_classes), dtype=np.float32)
test_xgb_probs = np.zeros((n_test, num_classes), dtype=np.float32)

X_test = test_processed[feature_cols].copy()
dtest_xgb = xgb.DMatrix(X_test)

for fold in range(n_folds):
    trn_idx = train_processed["fold"] != fold
    val_idx = train_processed["fold"] == fold

    X_tr = train_processed.loc[trn_idx, feature_cols]
    y_tr = train_processed.loc[trn_idx, "target_encoded"].values
    X_va = train_processed.loc[val_idx, feature_cols]
    y_va = train_processed.loc[val_idx, "target_encoded"].values

    # Train LightGBM Model A (Deeper leaves)
    lgb_trn_a = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_a = lgb.Dataset(X_va, label=y_va, reference=lgb_trn_a)

    lgb_model_a = lgb.train(
        lgb_params_a,
        lgb_trn_a,
        num_boost_round=1000,
        valid_sets=[lgb_val_a],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb_a_probs[val_idx] = lgb_model_a.predict(
        X_va, num_iteration=lgb_model_a.best_iteration
    )
    test_lgb_a_probs += (
        lgb_model_a.predict(X_test, num_iteration=lgb_model_a.best_iteration) / n_folds
    )

    # Train LightGBM Model B (Shallow regularized)
    lgb_trn_b = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_b = lgb.Dataset(X_va, label=y_va, reference=lgb_trn_b)

    lgb_model_b = lgb.train(
        lgb_params_b,
        lgb_trn_b,
        num_boost_round=1000,
        valid_sets=[lgb_val_b],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb_b_probs[val_idx] = lgb_model_b.predict(
        X_va, num_iteration=lgb_model_b.best_iteration
    )
    test_lgb_b_probs += (
        lgb_model_b.predict(X_test, num_iteration=lgb_model_b.best_iteration) / n_folds
    )

    # Train XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
    dval_xgb = xgb.DMatrix(X_va, label=y_va)

    xgb_model = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=1000,
        evals=[(dval_xgb, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_xgb = xgb_model.predict(
        dval_xgb, iteration_range=(0, xgb_model.best_iteration + 1)
    )
    oof_xgb_probs[val_idx] = val_pred_xgb
    test_xgb_probs += (
        xgb_model.predict(dtest_xgb, iteration_range=(0, xgb_model.best_iteration + 1))
        / n_folds
    )

# ---------------------------------------------------------
# 12. Evaluate Ensemble and Generate Submission
# ---------------------------------------------------------
y_all_true = train_processed["target_encoded"].values

score_lgb_a = compute_ndcg5(y_all_true, oof_lgb_a_probs)
score_lgb_b = compute_ndcg5(y_all_true, oof_lgb_b_probs)
score_xgb = compute_ndcg5(y_all_true, oof_xgb_probs)
print(f"LGBM-A OOF NDCG@5: {score_lgb_a:.6f}")
print(f"LGBM-B OOF NDCG@5: {score_lgb_b:.6f}")
print(f"XGBoost OOF NDCG@5: {score_xgb:.6f}")

# Grid Search for Optimal Blending Weights Maximizing NDCG@5
best_score = -1.0
best_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

for w1_i in range(0, 101, 5):
    w1 = w1_i / 100.0
    for w2_i in range(0, 101 - w1_i, 5):
        w2 = w2_i / 100.0
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue
        blend_probs = w1 * oof_lgb_a_probs + w2 * oof_lgb_b_probs + w3 * oof_xgb_probs
        score = compute_ndcg5(y_all_true, blend_probs)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

# Fine-grained search around the best weights
w1_best, w2_best, w3_best = best_weights
for dw1 in np.linspace(-0.04, 0.04, 9):
    for dw2 in np.linspace(-0.04, 0.04, 9):
        w1 = np.clip(w1_best + dw1, 0.0, 1.0)
        w2 = np.clip(w2_best + dw2, 0.0, 1.0)
        if w1 + w2 > 1.0:
            continue
        w3 = round(1.0 - w1 - w2, 4)
        blend_probs = w1 * oof_lgb_a_probs + w2 * oof_lgb_b_probs + w3 * oof_xgb_probs
        score = compute_ndcg5(y_all_true, blend_probs)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"Optimized Blending Weights (LGB-A, LGB-B, XGB): {best_weights}")
w_a, w_b, w_xgb = best_weights
oof_blend_probs = (
    w_a * oof_lgb_a_probs + w_b * oof_lgb_b_probs + w_xgb * oof_xgb_probs
)
test_ensemble_probs = (
    w_a * test_lgb_a_probs + w_b * test_lgb_b_probs + w_xgb * test_xgb_probs
)

# ---------------------------------------------------------
# Metric-Aligned Class-Wise Power Ranking Optimization
# ---------------------------------------------------------
def eval_power_calibration(gammas):
    g = np.clip(gammas, 0.1, 5.0)
    calib = np.power(np.maximum(oof_blend_probs, 1e-12), g)
    return -compute_ndcg5(y_all_true, calib)

best_gamma = np.ones(num_classes, dtype=np.float64)
best_calib_score = compute_ndcg5(y_all_true, oof_blend_probs)
print(f"Pre-calibration Blend NDCG@5: {best_calib_score:.6f}")

candidate_powers = [
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
    1.3,
    1.4,
    1.5,
    1.7,
]
for _ in range(2):
    for c in range(num_classes):
        best_gc = best_gamma[c]
        for p in candidate_powers:
            test_g = best_gamma.copy()
            test_g[c] = p
            score = -eval_power_calibration(test_g)
            if score > best_calib_score:
                best_calib_score = score
                best_gc = p
        best_gamma[c] = best_gc

opt_res = minimize(
    eval_power_calibration,
    best_gamma,
    method="Nelder-Mead",
    options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-5},
)
if -opt_res.fun > best_calib_score:
    best_gamma = np.clip(opt_res.x, 0.1, 5.0)
    best_calib_score = -opt_res.fun

print(f"Post-calibration Calibrated NDCG@5: {best_calib_score:.6f}")
print(f"Optimized Class Exponents: {np.round(best_gamma, 4)}")

final_score = best_calib_score
print(f"Final Validation Score: {final_score}")

calibrated_test_probs = np.power(
    np.maximum(test_ensemble_probs, 1e-12), best_gamma
)
test_pred_indices = np.argmax(calibrated_test_probs, axis=1)
test_pred_countries = label_encoder.inverse_transform(test_pred_indices)

sub_id = np.arange(n_test, dtype=int)
submission_df = pd.DataFrame(
    {"id": sub_id, "country_destination": test_pred_countries}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)