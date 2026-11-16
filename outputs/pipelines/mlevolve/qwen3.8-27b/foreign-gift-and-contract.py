import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import os
import pickle

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nTarget stats:\n{train['Amount'].describe()}")
print(f"\nMissing values (train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values (test):\n{test.isnull().sum()[test.isnull().sum() > 0]}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "Amount"
id_col = "id" if "id" in train.columns else None

y_train_full = train[target_col].values
train_features = train.drop(columns=[target_col] + ([id_col] if id_col else []))
test_features = (
    test.drop(columns=[target_col] + ([id_col] if id_col else []))
    if target_col in test.columns
    else test.drop(columns=[id_col] if id_col else [])
)

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = [c for c in train_features.columns if train_features[c].dtype == bool]
text_cols = [
    c
    for c in categorical_cols
    if train_features[c].apply(lambda x: isinstance(x, str) and len(str(x)) > 50).mean()
    > 0.3
]
categorical_cols_clean = [c for c in categorical_cols if c not in text_cols]

print(f"\nNumeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols_clean)}): {categorical_cols_clean}")
print(f"Text cols ({len(text_cols)}): {text_cols}")
print(f"Bool cols ({len(bool_cols)}): {bool_cols}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
train_feat, val_feat, y_tr, y_val = train_test_split(
    train_features, y_train_full, test_size=0.15, random_state=42
)
print(f"\nTrain split: {train_feat.shape}, Val split: {val_feat.shape}")
print(f"Target mean (train): {y_tr.mean():.2f}, (val): {y_val.mean():.2f}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# Fit stateful transforms ONLY on training split to prevent leakage
freq_maps = {}
for col in categorical_cols_clean:
    if col in train_feat.columns:
        freq_maps[col] = train_feat[col].value_counts(normalize=True)

# Target encoding with Bayesian smoothing: (sum + prior_mean * m) / (count + m), m=10
target_enc_maps = {}
m_smooth = 10
for col in categorical_cols_clean:
    if col in train_feat.columns:
        temp_df = train_feat.copy()
        temp_df[target_col] = y_tr
        agg = temp_df.groupby(col)[target_col].agg(['sum', 'count'])
        prior_mean = y_tr.mean()
        agg['smoothed'] = (agg['sum'] + prior_mean * m_smooth) / (agg['count'] + m_smooth)
        target_enc_maps[col] = agg['smoothed']

# Date-based feature engineering
date_cols = ['Receipt Date', 'Contract Start Date', 'Contract End Date']
date_feature_names = []

def add_date_features(df):
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            date_feature_names.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek', f'{col}_is_weekend'])
    # Contract duration
    if 'Contract Start Date' in df.columns and 'Contract End Date' in df.columns:
        df['contract_duration'] = (df['Contract End Date'] - df['Contract Start Date']).dt.days
        date_feature_names.append('contract_duration')
    # Days since receipt
    if 'Receipt Date' in df.columns:
        ref_date = pd.Timestamp('2024-01-01')
        df['days_since_receipt'] = (ref_date - df['Receipt Date']).dt.days
        date_feature_names.append('days_since_receipt')
    return df

train_feat = add_date_features(train_feat)
val_feat = add_date_features(val_feat)
test_features = add_date_features(test_features)


# Apply features function
def apply_features(
    df,
    freq_maps,
    target_enc_maps,
    numeric_cols,
    categorical_cols_clean,
    text_cols,
    bool_cols,
    train_mean,
):
    df = df.copy()

    # Text features
    for col in text_cols:
        if col in df.columns:
            df[f"{col}_len"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
            if df[col].nunique() > len(df) * 0.5:
                df = df.drop(columns=[col])

    # Categorical: frequency + target encoding + missing indicator + label encode
    for col in categorical_cols_clean:
        if col in df.columns:
            if col in freq_maps:
                df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)
            else:
                df[f"{col}_freq"] = 0

            if col in target_enc_maps:
                df[f"{col}_target_enc"] = (
                    df[col].map(target_enc_maps[col]).fillna(train_mean)
                )
            else:
                df[f"{col}_target_enc"] = 0

            df[f"{col}_isna"] = df[col].isna().astype(int)
            df[col] = df[col].astype("category")

    # Numeric: transforms
    numeric_in_df = [
        c for c in numeric_cols if c in df.columns and df[c].dtype != "category"
    ]
    for col in numeric_in_df:
        if col in df.columns:
            positive_mask = df[col] > 0
            if positive_mask.sum() > len(df) * 0.5:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            df[f"{col}_sq"] = df[col] ** 2

    # Interactions (limited)
    if len(numeric_in_df) <= 15:
        top_numeric = numeric_in_df[:8]
        for i in range(len(top_numeric)):
            for j in range(i + 1, len(top_numeric)):
                c1, c2 = top_numeric[i], top_numeric[j]
                if c1 in df.columns and c2 in df.columns:
                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

    # Boolean
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


train_feat_proc = apply_features(
    train_feat,
    freq_maps,
    target_enc_maps,
    numeric_cols,
    categorical_cols_clean,
    text_cols,
    bool_cols,
    y_tr.mean(),
)
val_feat_proc = apply_features(
    val_feat,
    freq_maps,
    target_enc_maps,
    numeric_cols,
    categorical_cols_clean,
    text_cols,
    bool_cols,
    y_tr.mean(),
)
test_feat_proc = apply_features(
    test_features,
    freq_maps,
    target_enc_maps,
    numeric_cols,
    categorical_cols_clean,
    text_cols,
    bool_cols,
    y_tr.mean(),
)

print(f"\nProcessed train shape: {train_feat_proc.shape}")
print(f"Processed val shape: {val_feat_proc.shape}")
print(f"Processed test shape: {test_feat_proc.shape}")


# ============================================================
# 5. PREPARE FINAL ARRAYS
# ============================================================
def prepare_arrays(df):
    arrays = []
    for col in df.columns:
        if df[col].dtype == "category":
            arrays.append(df[col].cat.codes.values)
        elif df[col].dtype == bool:
            arrays.append(df[col].astype(int).values)
        else:
            arr = pd.to_numeric(df[col], errors="coerce").values
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrays.append(arr)
    return np.column_stack(arrays) if arrays else np.zeros((len(df), 1))


X_train = prepare_arrays(train_feat_proc)
X_val = prepare_arrays(val_feat_proc)
X_test = prepare_arrays(test_feat_proc)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Save feature names
feature_names = train_feat_proc.columns.tolist()
os.makedirs("./working", exist_ok=True)
with open("./working/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Test ids
if id_col and id_col in test.columns:
    test_ids = test[id_col].values
else:
    test_ids = np.arange(len(test))

# ============================================================
# 6. MODEL: LightGBM + XGBoost Ensemble on original target scale
# ============================================================
print(f"\ny_train range: [{y_tr.min():.4f}, {y_tr.max():.4f}]")
print(f"y_train std: {y_tr.std():.4f}")

# --- LightGBM ---
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 5.0,
    "max_depth": -1,
    "min_split_gain": 0.01,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

train_data_lgb = lgb.Dataset(X_train, label=y_tr)
val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

callbacks_lgb = [
    lgb.early_stopping(stopping_rounds=200, verbose=False),
    lgb.log_evaluation(period=50),
]

lgb_model = lgb.train(
    lgb_params,
    train_data_lgb,
    num_boost_round=3000,
    valid_sets=[val_data_lgb],
    valid_names=["val"],
    callbacks=callbacks_lgb,
)

print(f"\nLightGBM Best iteration: {lgb_model.best_iteration}")
print(f"LightGBM Best validation RMSE: {lgb_model.best_score['val']['rmse']:.4f}")

# --- XGBoost ---
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_jobs": -1,
    "seed": 42,
}

dtrain = xgb.DMatrix(X_train, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=3000,
    evals=[(dval, "val")],
    early_stopping_rounds=200,
    verbose_eval=50,
)

print(f"\nXGBoost Best iteration: {xgb_model.best_iteration}")
print(f"XGBoost Best validation RMSE: {xgb_model.best_score:.4f}")

# ============================================================
# 7. PREDICTIONS & EVALUATION (Ensemble)
# ============================================================
lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
xgb_val_pred = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))

val_pred = 0.5 * lgb_val_pred + 0.5 * xgb_val_pred
val_pred = np.clip(val_pred, 0, None)

lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
dtest = xgb.DMatrix(X_test)
xgb_test_pred = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))

test_pred = 0.5 * lgb_test_pred + 0.5 * xgb_test_pred
test_pred = np.clip(test_pred, 0, None)

val_rmse = root_mean_squared_error(y_val, val_pred)
baseline_rmse = root_mean_squared_error(y_val, np.full(len(y_val), y_tr.mean()))

print(f"\nEnsemble Validation RMSE (original scale): {val_rmse:.4f}")
print(f"LightGBM alone RMSE: {root_mean_squared_error(y_val, np.clip(lgb_val_pred, 0, None)):.4f}")
print(f"XGBoost alone RMSE: {root_mean_squared_error(y_val, np.clip(xgb_val_pred, 0, None)):.4f}")
print(f"Baseline (mean) RMSE: {baseline_rmse:.4f}")
print(
    f"Improvement over baseline: {(baseline_rmse - val_rmse) / baseline_rmse * 100:.1f}%"
)

# Feature importance
importance = lgb_model.feature_importance(importance_type="gain")
top_idx = np.argsort(importance)[::-1][:15]
print("\nTop 15 features by gain (LightGBM):")
for i, idx in enumerate(top_idx):
    if idx < len(feature_names):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.1f}")

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids.astype(int), "Amount": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Test predictions range: [{test_pred.min():.2f}, {test_pred.max():.2f}], mean: {test_pred.mean():.2f}"
)

# ============================================================
# FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score:.4f}")
