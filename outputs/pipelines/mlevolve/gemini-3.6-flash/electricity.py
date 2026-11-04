import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. Setup directories and load raw data
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Inspect original target format semantics
sample_target = train_df["class"].iloc[0]
is_up_down = isinstance(sample_target, str) and sample_target.upper() in [
    "UP",
    "DOWN",
]
is_str_num = isinstance(sample_target, str) and sample_target in ["0", "1"]

# Process target variable
target_col = "class"
target_mapping = {"UP": 1, "DOWN": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
train_df[target_col] = (
    train_df[target_col]
    .astype(str)
    .str.strip()
    .str.upper()
    .map(target_mapping)
    .astype(int)
)

# Extract test IDs
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

n_train = len(train_df)
n_test = len(test_df)

# Combine datasets sequentially to compute continuous temporal and market features
full_df = pd.concat(
    [train_df, test_df.drop(columns=["id"], errors="ignore")],
    axis=0,
    ignore_index=True,
)

# Standardize column names to consistent casing
standard_names = {
    "date": "Date",
    "day": "Day",
    "period": "Period",
    "nswprice": "NSWprice",
    "nswdemand": "NSWdemand",
    "vicprice": "VICprice",
    "vicdemand": "VICdemand",
    "transfer": "transfer",
    "class": "class",
}
renames = {
    c: standard_names[c.lower()]
    for c in full_df.columns
    if c.lower() in standard_names
}
full_df = full_df.rename(columns=renames)

# Convert all feature columns to numeric dtypes to ensure no Categorical columns exist
for col in full_df.columns:
    if col != target_col:
        if (
            isinstance(full_df[col].dtype, pd.CategoricalDtype)
            or full_df[col].dtype == object
        ):
            full_df[col] = pd.to_numeric(full_df[col].astype(str), errors="coerce")
        else:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")


# 2. Feature Engineering Pipeline
def extract_market_features(df):
    df_feat = df.copy()

    # Cyclical encodings for time variables
    if "Period" in df_feat.columns:
        p_val = df_feat["Period"].astype(float)
        p_norm = (
            p_val * 2 * np.pi
            if p_val.max() <= 1.0
            else (p_val - 1.0) / 48.0 * 2 * np.pi
        )
        df_feat["period_sin"] = np.sin(p_norm)
        df_feat["period_cos"] = np.cos(p_norm)

    if "Day" in df_feat.columns:
        d_val = df_feat["Day"].astype(float)
        d_norm = d_val / 7.0 * 2 * np.pi
        df_feat["day_sin"] = np.sin(d_norm)
        df_feat["day_cos"] = np.cos(d_norm)

    # Cross-state market spreads & interactions
    if "NSWprice" in df_feat.columns and "VICprice" in df_feat.columns:
        df_feat["price_diff"] = df_feat["NSWprice"] - df_feat["VICprice"]
        df_feat["price_ratio"] = (df_feat["NSWprice"] + 1e-4) / (
            df_feat["VICprice"] + 1e-4
        )

    if "NSWdemand" in df_feat.columns and "VICdemand" in df_feat.columns:
        df_feat["demand_diff"] = df_feat["NSWdemand"] - df_feat["VICdemand"]
        df_feat["demand_sum"] = df_feat["NSWdemand"] + df_feat["VICdemand"]
        df_feat["demand_ratio"] = (df_feat["NSWdemand"] + 1e-4) / (
            df_feat["VICdemand"] + 1e-4
        )

    if (
        "transfer" in df_feat.columns
        and "NSWdemand" in df_feat.columns
        and "VICdemand" in df_feat.columns
    ):
        df_feat["nsw_net_demand"] = df_feat["NSWdemand"] - df_feat["transfer"]
        df_feat["vic_net_demand"] = df_feat["VICdemand"] + df_feat["transfer"]

    if "NSWprice" in df_feat.columns and "NSWdemand" in df_feat.columns:
        df_feat["nsw_market_val"] = df_feat["NSWprice"] * df_feat["NSWdemand"]

    # Temporal Lags (30m, 1h, 2h, 24h, 48h)
    lag_cols = [
        c
        for c in [
            "NSWprice",
            "NSWdemand",
            "VICprice",
            "VICdemand",
            "transfer",
            "price_diff",
            "demand_diff",
        ]
        if c in df_feat.columns
    ]
    lags = [1, 2, 4, 48, 96]

    for c in lag_cols:
        for l in lags:
            df_feat[f"{c}_lag_{l}"] = df_feat[c].shift(l)
            if l in [1, 48]:
                df_feat[f"{c}_diff_{l}"] = df_feat[c] - df_feat[c].shift(l)

    # (Target lag features excluded to ensure feature availability consistency between train and test)

    # Rolling window aggregations (on past shifted values)
    roll_cols = [
        c
        for c in ["NSWprice", "NSWdemand", "VICprice", "VICdemand", "price_diff"]
        if c in df_feat.columns
    ]
    windows = [12, 48]
    for c in roll_cols:
        shifted = df_feat[c].shift(1)
        for w in windows:
            r_mean = shifted.rolling(w, min_periods=1).mean()
            r_std = shifted.rolling(w, min_periods=1).std().fillna(0)
            df_feat[f"{c}_roll_{w}_mean"] = r_mean
            df_feat[f"{c}_roll_{w}_std"] = r_std
            df_feat[f"{c}_roll_{w}_z"] = (df_feat[c] - r_mean) / (r_std + 1e-4)

    return df_feat


full_processed = extract_market_features(full_df)

# Separate into processed train and test sets
train_processed = full_processed.iloc[:n_train].copy()
test_processed = full_processed.iloc[n_train:].copy()

feature_cols = [c for c in train_processed.columns if c != target_col]

# Time-based Out-Of-Time Validation Split (80% train, 20% validation)
val_size = int(n_train * 0.2)
train_split = train_processed.iloc[:-val_size].copy()
val_split = train_processed.iloc[-val_size:].copy()

X_train = train_split[feature_cols].copy()
y_train = train_split[target_col].copy().values

X_val = val_split[feature_cols].copy()
y_val = val_split[target_col].copy().values

X_test = test_processed[feature_cols].copy()

# Leakage-free Imputation using Training set medians
train_medians = X_train.median().fillna(0)
X_train = X_train.fillna(train_medians).fillna(0)
X_val = X_val.fillna(train_medians).fillna(0)
X_test = X_test.fillna(train_medians).fillna(0)

# Leakage-free Standardization fitted ONLY on train split
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
X_test = pd.DataFrame(
    scaler.transform(X_test), columns=feature_cols, index=X_test.index
)

# 3. Model Training
# Model 1: LightGBM Classifier
lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

# Model 2: XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric="logloss",
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Model 3: ExtraTrees Classifier
et_model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
et_model.fit(X_train, y_train)

# Model 4: Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

# 4. Ensemble Inference & Threshold Optimization on Validation Set
lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
et_val_prob = et_model.predict_proba(X_val)[:, 1]
rf_val_prob = rf_model.predict_proba(X_val)[:, 1]

w_lgb, w_xgb, w_et, w_rf = 0.40, 0.40, 0.10, 0.10
val_probs = (
    w_lgb * lgb_val_prob
    + w_xgb * xgb_val_prob
    + w_et * et_val_prob
    + w_rf * rf_val_prob
)

best_thresh = 0.5
best_acc = 0.0

for thresh in np.linspace(0.35, 0.65, 31):
    acc = accuracy_score(y_val, (val_probs >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

val_preds = (val_probs >= best_thresh).astype(int)
val_acc = accuracy_score(y_val, val_preds)

# 5. Test Inference & Formatting Submission
lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
et_test_prob = et_model.predict_proba(X_test)[:, 1]
rf_test_prob = rf_model.predict_proba(X_test)[:, 1]

test_probs = (
    w_lgb * lgb_test_prob
    + w_xgb * xgb_test_prob
    + w_et * et_test_prob
    + w_rf * rf_test_prob
)
test_preds_binary = (test_probs >= best_thresh).astype(int)

if is_up_down:
    test_preds_final = np.where(test_preds_binary == 1, "UP", "DOWN")
elif is_str_num:
    test_preds_final = test_preds_binary.astype(str)
else:
    test_preds_final = test_preds_binary

submission_df = pd.DataFrame({"id": test_ids, "class": test_preds_final})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_acc}")
