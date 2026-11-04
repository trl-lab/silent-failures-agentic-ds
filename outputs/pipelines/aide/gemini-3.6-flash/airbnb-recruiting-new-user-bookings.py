import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Store target and label encode
y_raw = train_df["country_destination"].values
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

n_train = len(train_df)
n_test = len(test_df)

# Preserve IDs for session merging if applicable
user_ids_all = pd.concat([train_df.get("id"), test_df.get("id")], ignore_index=True)

df_all = pd.concat(
    [
        train_df.drop(columns=["country_destination"], errors="ignore"),
        test_df.drop(columns=["country_destination"], errors="ignore"),
    ],
    axis=0,
    ignore_index=True,
)

# Remove leakage / non-predictive columns
cols_to_drop = ["id", "date_first_booking"]
df_all = df_all.drop(columns=[c for c in cols_to_drop if c in df_all.columns])

# 2. Check and Merge Session Aggregates if Available
session_feats = None
sessions_path_parquet = "./input/sessions.parquet"
sessions_path_csv = "./input/sessions.csv"

if os.path.exists(sessions_path_parquet):
    sessions_df = pd.read_parquet(sessions_path_parquet)
elif os.path.exists(sessions_path_csv):
    sessions_df = pd.read_csv(sessions_path_csv)
else:
    sessions_df = None

if sessions_df is not None and "user_id" in sessions_df.columns:
    sessions_df["user_id"] = sessions_df["user_id"].astype(str)

    # Aggregating secs_elapsed
    secs_agg = sessions_df.groupby("user_id")["secs_elapsed"].agg(
        ["sum", "mean", "std", "median", "max", "min", "count"]
    )
    secs_agg.columns = [f"session_secs_{c}" for c in secs_agg.columns]

    # Action counts for top actions
    if "action" in sessions_df.columns:
        action_counts = (
            sessions_df.groupby(["user_id", "action"]).size().unstack(fill_value=0)
        )
        top_actions = action_counts.sum().nlargest(25).index
        action_counts = action_counts[top_actions]
        action_counts.columns = [f"act_{c}" for c in action_counts.columns]
        session_feats = pd.concat([secs_agg, action_counts], axis=1)
    else:
        session_feats = secs_agg

    session_feats.index.name = "user_id_key"

# 3. Categorical Interactions & Feature Engineering

# Categorical Interactions
for col in [
    "first_device_type",
    "first_browser",
    "affiliate_channel",
    "affiliate_provider",
    "signup_method",
    "signup_app",
]:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna("-unknown-").astype(str)

if "first_device_type" in df_all.columns and "first_browser" in df_all.columns:
    df_all["device_browser"] = (
        df_all["first_device_type"] + "_" + df_all["first_browser"]
    )

if "affiliate_channel" in df_all.columns and "affiliate_provider" in df_all.columns:
    df_all["affiliate_combo"] = (
        df_all["affiliate_channel"] + "_" + df_all["affiliate_provider"]
    )

if "signup_method" in df_all.columns and "signup_app" in df_all.columns:
    df_all["signup_combo"] = df_all["signup_method"] + "_" + df_all["signup_app"]

# Age cleaning & indicators
df_all["age_cleaned"] = df_all["age"].apply(
    lambda x: np.nan if (pd.isna(x) or x < 14 or x > 100) else float(x)
)
df_all["age_isna"] = df_all["age_cleaned"].isna().astype(int)

# Date account created features
dac = pd.to_datetime(df_all["date_account_created"], errors="coerce")
df_all["dac_year"] = dac.dt.year.fillna(-1).astype(int)
df_all["dac_month"] = dac.dt.month.fillna(-1).astype(int)
df_all["dac_day"] = dac.dt.day.fillna(-1).astype(int)
df_all["dac_dayofweek"] = dac.dt.dayofweek.fillna(-1).astype(int)
df_all["dac_dayofyear"] = dac.dt.dayofyear.fillna(-1).astype(int)
df_all["dac_quarter"] = dac.dt.quarter.fillna(-1).astype(int)
df_all = df_all.drop(columns=["date_account_created"], errors="ignore")

# Timestamp first active features
tfa_str = df_all["timestamp_first_active"].astype(str)
tfa_dt = pd.to_datetime(tfa_str, format="%Y%m%d%H%M%S", errors="coerce")
df_all["tfa_year"] = tfa_dt.dt.year.fillna(-1).astype(int)
df_all["tfa_month"] = tfa_dt.dt.month.fillna(-1).astype(int)
df_all["tfa_day"] = tfa_dt.dt.day.fillna(-1).astype(int)
df_all["tfa_hour"] = tfa_dt.dt.hour.fillna(-1).astype(int)
df_all["tfa_dayofweek"] = tfa_dt.dt.dayofweek.fillna(-1).astype(int)
df_all["tfa_dayofyear"] = tfa_dt.dt.dayofyear.fillna(-1).astype(int)

# Lag metrics
time_lag_sec = (dac - tfa_dt).dt.total_seconds().fillna(-1)
df_all["time_lag_seconds"] = time_lag_sec
df_all["time_lag_days"] = (time_lag_sec / 86400.0).astype(np.float32)
df_all = df_all.drop(columns=["timestamp_first_active"], errors="ignore")

# Merge sessions data if available
if session_feats is not None and user_ids_all is not None:
    df_all["user_id_key"] = user_ids_all.astype(str)
    df_all = df_all.merge(session_feats, on="user_id_key", how="left")
    df_all = df_all.drop(columns=["user_id_key"])

# Encoding categoricals
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    df_all[col] = df_all[col].fillna("-unknown-").astype(str)
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

# Split back into train and test
X_train = df_encoded.iloc[:n_train].copy()
X_test = df_encoded.iloc[n_train:].copy()


# 4. Validation Metric: NDCG@5 Function
def compute_ndcg5(y_true, y_prob):
    top5 = np.argsort(-y_prob, axis=1)[:, :5]
    ranks = np.where(top5 == y_true[:, None])
    scores = np.zeros(len(y_true))
    scores[ranks[0]] = 1.0 / np.log2(ranks[1] + 2)
    return np.mean(scores)


# 5. Model Training & Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_probs = np.zeros((n_train, len(le_target.classes_)))
test_probs = np.zeros((n_test, len(le_target.classes_)))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": len(le_target.classes_),
    "eval_metric": "mlogloss",
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "tree_method": "hist",
    "n_estimators": 180,
    "random_state": SEED,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_preds = model.predict_proba(X_va)
    oof_probs[val_idx] = val_preds
    test_probs += model.predict_proba(X_test) / n_splits

# 6. Calculate Overall Cross-Validation NDCG@5 Score
cv_score = compute_ndcg5(y, oof_probs)
print(f"5-Fold Cross-Validation NDCG@5: {cv_score:.5f}")

# 7. Generate Predictions and Save Submission File
test_top5_indices = np.argsort(-test_probs, axis=1)[:, :5]

sub_id = []
sub_country = []

for i, top5 in enumerate(test_top5_indices):
    top5_countries = le_target.inverse_transform(top5)
    for c in top5_countries:
        sub_id.append(i)
        sub_country.append(c)

sub_df = pd.DataFrame(
    {"id": np.arange(len(sub_country)), "country_destination": sub_country}
)

sub_df.to_csv("./working/submission.csv", index=False)
