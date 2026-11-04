import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading train and test datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Drop date_first_booking if present (causes target leakage as it's missing for non-booking/test users)
if "date_first_booking" in train_df.columns:
    train_df = train_df.drop(columns=["date_first_booking"])
if "date_first_booking" in test_df.columns:
    test_df = test_df.drop(columns=["date_first_booking"])

# Target column and user IDs
target_col = "country_destination"
y_raw = train_df[target_col].copy()
train_df = train_df.drop(columns=[target_col])

# Encode target classes
le_target = LabelEncoder()
y_all = le_target.fit_transform(y_raw)
class_names = le_target.classes_.tolist()
num_classes = len(class_names)

# ---------------------------------------------------------
# 2. Chronological Validation Split (Prevent Temporal Leakage)
# ---------------------------------------------------------
train_df["date_account_created_dt"] = pd.to_datetime(train_df["date_account_created"])
sort_indices = train_df["date_account_created_dt"].argsort()

train_df = train_df.iloc[sort_indices].reset_index(drop=True)
y_all = y_all[sort_indices]

# Time-based split: Use latest 15% of train set as holdout validation set
n_samples = len(train_df)
val_size = int(n_samples * 0.15)
train_split_idx = n_samples - val_size

train_subset = train_df.iloc[:train_split_idx].copy().reset_index(drop=True)
val_subset = train_df.iloc[train_split_idx:].copy().reset_index(drop=True)

y_train = y_all[:train_split_idx]
y_val = y_all[train_split_idx:]


# ---------------------------------------------------------
# 3. Domain Feature Engineering Functions
# ---------------------------------------------------------
def clean_age(df_age):
    """Clean age anomalies and extract birth year corrections."""
    age = df_age.copy()
    mask_birth_year = (age > 1850) & (age <= 2014)
    age[mask_birth_year] = 2014 - age[mask_birth_year]
    mask_invalid = (age < 14) | (age > 100)
    age[mask_invalid] = np.nan
    return age


def extract_datetime_features(df):
    """Extract temporal signals, activity latency lag, log lag, and cyclical temporal encodings."""
    dac = pd.to_datetime(df["date_account_created"])
    tfa_str = df["timestamp_first_active"].astype(str)
    tfa = pd.to_datetime(tfa_str, format="%Y%m%d%H%M%S", errors="coerce")

    res = pd.DataFrame(index=df.index)

    dac_month = dac.dt.month.fillna(-1)
    dac_dow = dac.dt.dayofweek.fillna(-1)

    res["dac_year"] = dac.dt.year.fillna(-1).astype(int)
    res["dac_month"] = dac_month.astype(int)
    res["dac_day"] = dac.dt.day.fillna(-1).astype(int)
    res["dac_dayofweek"] = dac_dow.astype(int)
    res["dac_dayofyear"] = dac.dt.dayofyear.fillna(-1).astype(int)
    res["dac_quarter"] = dac.dt.quarter.fillna(-1).astype(int)
    res["dac_is_weekend"] = dac.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encodings for DAC
    res["dac_month_sin"] = np.sin(2 * np.pi * np.maximum(0, dac_month) / 12.0)
    res["dac_month_cos"] = np.cos(2 * np.pi * np.maximum(0, dac_month) / 12.0)
    res["dac_dow_sin"] = np.sin(2 * np.pi * np.maximum(0, dac_dow) / 7.0)
    res["dac_dow_cos"] = np.cos(2 * np.pi * np.maximum(0, dac_dow) / 7.0)

    tfa_month = tfa.dt.month.fillna(-1)
    tfa_dow = tfa.dt.dayofweek.fillna(-1)
    tfa_hour = tfa.dt.hour.fillna(-1)

    res["tfa_year"] = tfa.dt.year.fillna(-1).astype(int)
    res["tfa_month"] = tfa_month.astype(int)
    res["tfa_day"] = tfa.dt.day.fillna(-1).astype(int)
    res["tfa_hour"] = tfa_hour.astype(int)
    res["tfa_dayofweek"] = tfa_dow.astype(int)
    res["tfa_dayofyear"] = tfa.dt.dayofyear.fillna(-1).astype(int)

    # Cyclical encodings for TFA
    res["tfa_month_sin"] = np.sin(2 * np.pi * np.maximum(0, tfa_month) / 12.0)
    res["tfa_month_cos"] = np.cos(2 * np.pi * np.maximum(0, tfa_month) / 12.0)
    res["tfa_dow_sin"] = np.sin(2 * np.pi * np.maximum(0, tfa_dow) / 7.0)
    res["tfa_dow_cos"] = np.cos(2 * np.pi * np.maximum(0, tfa_dow) / 7.0)
    res["tfa_hour_sin"] = np.sin(2 * np.pi * np.maximum(0, tfa_hour) / 24.0)
    res["tfa_hour_cos"] = np.cos(2 * np.pi * np.maximum(0, tfa_hour) / 24.0)

    time_lag_sec = (dac - tfa).dt.total_seconds()
    res["time_lag_seconds"] = time_lag_sec.fillna(0)
    res["time_lag_days"] = (time_lag_sec / 86400.0).fillna(0)
    res["time_lag_log"] = np.log1p(np.maximum(0, time_lag_sec.fillna(0)))
    res["has_activity_lag"] = (time_lag_sec.abs() > 0).astype(int)

    return res


def build_categorical_interactions(df):
    """Construct high-value domain categorical interactions and parsed family features."""
    res = pd.DataFrame(index=df.index)

    cat_cols = [
        "gender",
        "signup_method",
        "signup_flow",
        "language",
        "affiliate_channel",
        "affiliate_provider",
        "first_affiliate_tracked",
        "signup_app",
        "first_device_type",
        "first_browser",
    ]

    for c in cat_cols:
        res[c] = df[c].fillna("-unknown-").astype(str)

    res["browser_family"] = res["first_browser"].apply(
        lambda x: x.split()[0] if x != "-unknown-" else "-unknown-"
    )
    res["device_family"] = res["first_device_type"].apply(
        lambda x: "Desktop" if "Desktop" in x else ("Tablet" if "Tablet" in x or "iPad" in x else ("Phone" if "Phone" in x or "iPhone" in x else x))
    )

    res["channel_provider"] = res["affiliate_channel"] + "_" + res["affiliate_provider"]
    res["device_browser"] = res["first_device_type"] + "_" + res["first_browser"]
    res["app_device"] = res["signup_app"] + "_" + res["first_device_type"]
    res["gender_signup"] = res["gender"] + "_" + res["signup_method"]
    res["lang_app"] = res["language"] + "_" + res["signup_app"]

    return res


# ---------------------------------------------------------
# 4. Feature Engineering Assembly
# ---------------------------------------------------------
train_age_cleaned = clean_age(train_subset["age"])
val_age_cleaned = clean_age(val_subset["age"])
test_age_cleaned = clean_age(test_df["age"])

median_age = train_age_cleaned.median()


def assemble_features_df(df, age_cleaned):
    features = pd.DataFrame(index=df.index)
    features["age"] = age_cleaned.fillna(median_age)
    features["age_is_missing"] = age_cleaned.isna().astype(int)
    features["age_group"] = pd.cut(
        features["age"], bins=[0, 18, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    dt_feats = extract_datetime_features(df)
    features = pd.concat([features, dt_feats], axis=1)

    cat_feats = build_categorical_interactions(df)
    features = pd.concat([features, cat_feats], axis=1)

    return features


def compute_oof_multiclass_te(df_tr, y_tr, df_va, df_te, te_cols, n_classes, n_splits=5, smoothing=10.0):
    global_priors = np.bincount(y_tr, minlength=n_classes) / len(y_tr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    tr_te_arr = np.zeros((len(df_tr), len(te_cols) * n_classes), dtype=np.float32)
    va_te_arr = np.zeros((len(df_va), len(te_cols) * n_classes), dtype=np.float32)
    te_te_arr = np.zeros((len(df_te), len(te_cols) * n_classes), dtype=np.float32)

    for col_idx, col in enumerate(te_cols):
        for tr_idx, val_idx in skf.split(df_tr, y_tr):
            fold_cat = df_tr[col].iloc[tr_idx].values
            fold_y = y_tr[tr_idx]

            for k in range(n_classes):
                y_k = (fold_y == k).astype(float)
                counts = pd.Series(y_k).groupby(fold_cat).agg(['count', 'sum'])
                smoothed = (counts['sum'] + smoothing * global_priors[k]) / (counts['count'] + smoothing)
                map_dict = smoothed.to_dict()

                val_cat = df_tr[col].iloc[val_idx]
                feat_idx = col_idx * n_classes + k
                tr_te_arr[val_idx, feat_idx] = val_cat.map(map_dict).fillna(global_priors[k]).values

        full_cat = df_tr[col].values
        for k in range(n_classes):
            y_k = (y_tr == k).astype(float)
            counts = pd.Series(y_k).groupby(full_cat).agg(['count', 'sum'])
            smoothed = (counts['sum'] + smoothing * global_priors[k]) / (counts['count'] + smoothing)
            map_dict = smoothed.to_dict()

            feat_idx = col_idx * n_classes + k
            va_te_arr[:, feat_idx] = df_va[col].map(map_dict).fillna(global_priors[k]).values
            te_te_arr[:, feat_idx] = df_te[col].map(map_dict).fillna(global_priors[k]).values

    te_col_names = [f"{c}_te_class_{k}" for c in te_cols for k in range(n_classes)]
    return (pd.DataFrame(tr_te_arr, columns=te_col_names, index=df_tr.index),
            pd.DataFrame(va_te_arr, columns=te_col_names, index=df_va.index),
            pd.DataFrame(te_te_arr, columns=te_col_names, index=df_te.index))


df_train_feats = assemble_features_df(train_subset, train_age_cleaned)
df_val_feats = assemble_features_df(val_subset, val_age_cleaned)
df_test_feats = assemble_features_df(test_df, test_age_cleaned)

high_card_cols = ["first_browser", "channel_provider", "device_browser", "app_device", "affiliate_provider"]
tr_te, va_te, te_te = compute_oof_multiclass_te(
    df_train_feats, y_train, df_val_feats, df_test_feats, high_card_cols, num_classes
)

df_train_feats = pd.concat([df_train_feats, tr_te], axis=1)
df_val_feats = pd.concat([df_val_feats, va_te], axis=1)
df_test_feats = pd.concat([df_test_feats, te_te], axis=1)

categorical_columns = df_train_feats.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in categorical_columns:
    freq_map = df_train_feats[col].value_counts().to_dict()
    df_train_feats[f"{col}_freq"] = df_train_feats[col].map(freq_map).fillna(0)
    df_val_feats[f"{col}_freq"] = df_val_feats[col].map(freq_map).fillna(0)
    df_test_feats[f"{col}_freq"] = df_test_feats[col].map(freq_map).fillna(0)

for col in categorical_columns:
    train_vals = df_train_feats[col].astype(str)
    val_vals = df_val_feats[col].astype(str)
    test_vals = df_test_feats[col].astype(str)

    unique_categories = sorted(train_vals.unique())
    cat_to_code = {cat: i + 1 for i, cat in enumerate(unique_categories)}

    df_train_feats[col] = train_vals.map(cat_to_code).fillna(0).astype(int)
    df_val_feats[col] = val_vals.map(cat_to_code).fillna(0).astype(int)
    df_test_feats[col] = test_vals.map(cat_to_code).fillna(0).astype(int)

X_train = df_train_feats.values.astype(np.float32)
X_val = df_val_feats.values.astype(np.float32)
X_test = df_test_feats.values.astype(np.float32)

feature_names = df_train_feats.columns.tolist()
categorical_indices = [feature_names.index(c) for c in categorical_columns]


# ---------------------------------------------------------
# 5. Metric Definition: NDCG@5
# ---------------------------------------------------------
def compute_ndcg5(y_true, y_proba):
    """Computes exact NDCG@5 score for multi-class destination predictions."""
    top5_preds = np.argsort(y_proba, axis=1)[:, ::-1][:, :5]
    hits = top5_preds == y_true[:, None]
    discounts = 1.0 / np.log2(np.arange(2, 7))
    ndcg_per_sample = (hits * discounts).sum(axis=1)
    return float(np.mean(ndcg_per_sample))


# ---------------------------------------------------------
# 6. Model 1: Deep LightGBM Classifier
# ---------------------------------------------------------
lgb_train_ds = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_indices
)
lgb_val_ds = lgb.Dataset(
    X_val,
    label=y_val,
    reference=lgb_train_ds,
    categorical_feature=categorical_indices,
)

lgb_params1 = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

gbm_model1 = lgb.train(
    lgb_params1,
    lgb_train_ds,
    num_boost_round=700,
    valid_sets=[lgb_val_ds],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_probs1 = gbm_model1.predict(X_val)
test_probs1 = gbm_model1.predict(X_test)
score1 = compute_ndcg5(y_val, val_probs1)
print(f"LGBM Model 1 Val NDCG@5: {score1:.5f}")

# ---------------------------------------------------------
# 7. Model 2: Regularized Shallow LightGBM Classifier
# ---------------------------------------------------------
lgb_params2 = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 5,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 2024,
    "n_jobs": -1,
    "verbose": -1,
}

gbm_model2 = lgb.train(
    lgb_params2,
    lgb_train_ds,
    num_boost_round=700,
    valid_sets=[lgb_val_ds],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_probs2 = gbm_model2.predict(X_val)
test_probs2 = gbm_model2.predict(X_test)
score2 = compute_ndcg5(y_val, val_probs2)
print(f"LGBM Model 2 Val NDCG@5: {score2:.5f}")

# ---------------------------------------------------------
# 8. Model 3: XGBoost Multi-class Classifier
# ---------------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "nthread": -1,
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=600,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

val_probs3 = xgb_model.predict(dval)
test_probs3 = xgb_model.predict(dtest)
score3 = compute_ndcg5(y_val, val_probs3)
print(f"XGBoost Model 3 Val NDCG@5: {score3:.5f}")

# ---------------------------------------------------------
# 8.1 Model 4: CatBoost Multi-class Classifier
# ---------------------------------------------------------
cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=6,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=0,
    thread_count=-1
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

val_probs4 = cat_model.predict_proba(X_val)
test_probs4 = cat_model.predict_proba(X_test)
score4 = compute_ndcg5(y_val, val_probs4)
print(f"CatBoost Model 4 Val NDCG@5: {score4:.5f}")

# ---------------------------------------------------------
# 9. Ensemble Blend Optimization
# ---------------------------------------------------------
best_blend_score = -1.0
best_weights = (0.25, 0.25, 0.25, 0.25)

for w1 in np.linspace(0, 1, 11):
    for w2 in np.linspace(0, 1 - w1, 11):
        for w3 in np.linspace(0, 1 - w1 - w2, 11):
            w4 = max(0.0, 1.0 - w1 - w2 - w3)
            blended_val = w1 * val_probs1 + w2 * val_probs2 + w3 * val_probs3 + w4 * val_probs4
            score = compute_ndcg5(y_val, blended_val)
            if score > best_blend_score:
                best_blend_score = score
                best_weights = (w1, w2, w3, w4)

w1, w2, w3, w4 = best_weights
blended_val = w1 * val_probs1 + w2 * val_probs2 + w3 * val_probs3 + w4 * val_probs4
blended_test = w1 * test_probs1 + w2 * test_probs2 + w3 * test_probs3 + w4 * test_probs4
print(
    f"Ensemble Weights: ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f}) -> Blended NDCG@5: {best_blend_score:.5f}"
)


# ---------------------------------------------------------
# 10. Class Probability Calibration for NDCG@5
# ---------------------------------------------------------
def loss_func(weights):
    adj_probs = blended_val * weights
    return -compute_ndcg5(y_val, adj_probs)


init_c = np.ones(num_classes)
res = minimize(
    loss_func,
    init_c,
    method="Nelder-Mead",
    options={"maxiter": 300, "disp": False},
)

opt_class_weights = res.x
final_val_probs = blended_val * opt_class_weights
final_val_ndcg = compute_ndcg5(y_val, final_val_probs)

# ---------------------------------------------------------
# 11. Test Prediction & Submission Generation
# ---------------------------------------------------------
final_test_probs = blended_test * opt_class_weights
pred_class_indices = np.argmax(final_test_probs, axis=1)
pred_countries = [class_names[idx] for idx in pred_class_indices]

submission_df = pd.DataFrame(
    {"id": np.arange(len(X_test)), "country_destination": pred_countries}
)

submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_ndcg}")
