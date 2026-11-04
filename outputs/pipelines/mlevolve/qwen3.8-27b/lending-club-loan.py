import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:\n{train['loan_status'].value_counts()}")
print(f"\nTarget dtype: {train['loan_status'].dtype}")

# ============================================================
# 2. DETERMINE BINARY TARGET ENCODING
# ============================================================
target_values = train["loan_status"].unique()
print(f"\nUnique target values: {target_values}")

if train["loan_status"].nunique() == 2:
    train["target"] = (train["loan_status"] == "Charged Off").astype(int)
    print(
        f"Binary target mapping: {train['loan_status'].unique()} -> {train['target'].unique()}"
    )
else:
    bad_statuses = {"Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"}
    train["target"] = train["loan_status"].isin(bad_statuses).astype(int)
    print(f"Multi-class -> binary mapping. Bad rate: {train['target'].mean():.4f}")

# ============================================================
# 3. IDENTIFY FEATURE TYPES & PREPARE SPLIT
# ============================================================
id_col = "id" if "id" in train.columns else None
if id_col:
    train_id = train[id_col].copy()
    test_id = test[id_col].copy()
    train.drop(columns=[id_col], inplace=True)
    test.drop(columns=[id_col], inplace=True)

train.drop(columns=["loan_status"], inplace=True)
if "loan_status" in test.columns:
    test.drop(columns=["loan_status"], inplace=True)

# Remove high-cardinality ID-like noise features
drop_cols = ['title', 'emp_title', 'zip_code', 'member_id', 'url', 'application_type', 'pymnt_plan']
drop_cols = [c for c in drop_cols if c in train.columns]
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)
print(f"Dropped noise features: {drop_cols}")

numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
bool_cols = [c for c in train.columns if train[c].dtype == "bool"]

print(f"\nNumeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Boolean features: {len(bool_cols)}")

# ============================================================
# 4. STRATIFIED TRAIN/VAL SPLIT (BEFORE feature engineering)
# ============================================================
X_train_full = train.drop(columns=["target"]).copy()
y_train_full = train["target"].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nTrain split: {X_train.shape}, Val split: {X_val.shape}")
print(f"Train target mean: {y_train.mean():.4f}, Val target mean: {y_val.mean():.4f}")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, numeric_cols, categorical_cols, bool_cols, cat_encoders=None):
    """Apply feature engineering. Fit parameters only on training data."""
    df = df.copy()

    # Boolean columns -> numeric
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Categorical encoding (label encoding for tree models)
    for col in categorical_cols:
        if col in df.columns:
            if cat_encoders is not None and col in cat_encoders:
                known_cats = cat_encoders[col]
                df[col] = df[col].map(known_cats)
                df[col] = df[col].fillna(-1)
            else:
                df[col] = df[col].astype("category")
                categories = df[col].cat.categories
                df[col] = df[col].cat.codes
                if cat_encoders is not None:
                    cat_encoders[col] = {
                        cat: code
                        for cat, code in zip(categories, range(len(categories)))
                    }

    # Log transform for skewed positive numeric features
    for col in numeric_cols:
        if col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.min() > 0:
                skew = non_null.skew()
                if skew > 1.0:
                    df[col] = np.log1p(df[col])

    # Interaction features
    if "annual_inc" in df.columns and "dti" in df.columns:
        df["debt_amount"] = df["annual_inc"] * df["dti"] / 100
    if "annual_inc" in df.columns and "loan_amnt" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    if "loan_amnt" in df.columns and "int_rate" in df.columns:
        df["monthly_payment_est"] = df["loan_amnt"] * (df["int_rate"] / 100 / 12)
    if "revol_bal" in df.columns and "revol_util" in df.columns:
        df["credit_limit_used"] = df["revol_bal"] / (df["revol_util"] + 1) * 100
    if "total_acc" in df.columns and "revol_acc" in df.columns:
        df["installment_acc_ratio"] = (df["total_acc"] - df["revol_acc"]) / (
            df["total_acc"] + 1
        )
    if "pub_rec" in df.columns and "total_acc" in df.columns:
        df["pub_rec_ratio"] = df["pub_rec"] / (df["total_acc"] + 1)
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)
    if "inq_last_6mths" in df.columns and "open_acc" in df.columns:
        df["recent_inquiry_ratio"] = df["inq_last_6mths"] / (df["open_acc"] + 1)
    if "dti" in df.columns and "int_rate" in df.columns:
        df["dti_rate_interaction"] = df["dti"] * df["int_rate"]
    if "annual_inc" in df.columns and "int_rate" in df.columns:
        df["income_rate_interaction"] = df["annual_inc"] * df["int_rate"]
    if "utilization_ratio" in df.columns and "int_rate" in df.columns:
        df["util_rate_interaction"] = df["utilization_ratio"] * df["int_rate"]
    if "acc_now_delinq" in df.columns and "total_acc" in df.columns:
        df["delinq_ratio"] = df["acc_now_delinq"] / (df["total_acc"] + 1)
    if "chargeoff_within_12_mths" in df.columns and "total_acc" in df.columns:
        df["chargeoff_ratio"] = df["chargeoff_within_12_mths"] / (df["total_acc"] + 1)
    if "delinq_amnt" in df.columns and "revol_bal" in df.columns:
        df["delinq_bal_ratio"] = df["delinq_amnt"] / (df["revol_bal"] + 1)

    # Polynomial features
    if "int_rate" in df.columns:
        df["int_rate_sq"] = df["int_rate"] ** 2
    if "dti" in df.columns:
        df["dti_sq"] = df["dti"] ** 2

    # Missing value indicators
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df


# Fit categorical encoders on training data
cat_encoders = {}
X_train_eng = engineer_features(
    X_train, numeric_cols, categorical_cols, bool_cols, cat_encoders=cat_encoders
)
X_val_eng = engineer_features(
    X_val, numeric_cols, categorical_cols, bool_cols, cat_encoders=cat_encoders
)
X_test_eng = engineer_features(
    test, numeric_cols, categorical_cols, bool_cols, cat_encoders=cat_encoders
)

# Align columns
all_cols = X_train_eng.columns.tolist()
for col in all_cols:
    if col not in X_val_eng.columns:
        X_val_eng[col] = 0
    if col not in X_test_eng.columns:
        X_test_eng[col] = 0

X_val_eng = X_val_eng[all_cols]
X_test_eng = X_test_eng[all_cols]

# Replace inf values
X_train_eng = X_train_eng.replace([np.inf, -np.inf], np.nan)
X_val_eng = X_val_eng.replace([np.inf, -np.inf], np.nan)
X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan)

print(f"\nFinal feature matrix:")
print(f"  Train: {X_train_eng.shape}")
print(f"  Val:   {X_val_eng.shape}")
print(f"  Test:  {X_test_eng.shape}")

# ============================================================
# 6. MODEL DESIGN & TRAINING (LightGBM)
# ============================================================
pos_count = int(y_train.sum())
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count
print(
    f"\n  Class balance: pos={pos_count}, neg={neg_count}, scale_pos_weight={scale_pos_weight:.3f}"
)

seeds = [42, 123, 456, 789, 1024]
models = []
val_preds_list = []
test_preds_list = []

for seed in seeds:
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
        "seed": seed,
        "num_threads": -1,
    }

    dtrain = lgb.Dataset(X_train_eng, label=y_train, feature_name=all_cols)
    dval = lgb.Dataset(X_val_eng, label=y_val, feature_name=all_cols, reference=dtrain)

    print(f"\nTraining LightGBM seed={seed} with early stopping (patience=50)...")

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    best_iter = model.best_iteration
    best_val_auc = model.best_score["valid"]["auc"]
    print(f"  Seed {seed} - Best iteration: {best_iter}, Val AUC: {best_val_auc:.6f}")

    models.append(model)
    val_preds_list.append(model.predict(X_val_eng, num_iteration=best_iter))
    test_preds_list.append(model.predict(X_test_eng, num_iteration=best_iter))

# ============================================================
# 7. PREDICTIONS & EVALUATION (Ensemble Average)
# ============================================================
val_preds = np.mean(val_preds_list, axis=0)
test_preds = np.mean(test_preds_list, axis=0)

val_auc = roc_auc_score(y_val, val_preds)
print(f"\nValidation ROC AUC (ensemble average): {val_auc:.6f}")

print(
    f"Test predictions: shape={test_preds.shape}, mean={test_preds.mean():.4f}, "
    f"min={test_preds.min():.4f}, max={test_preds.max():.4f}"
)

# Feature importance (from first model)
importance = models[0].feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=all_cols).sort_values(ascending=False)
print(f"\nTop 15 features by gain importance:")
for i, (feat, imp) in enumerate(feat_imp.head(15).items()):
    print(f"  {i+1:2d}. {feat}: {imp:.1f}")

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

# Use 0-based integer row index as 'id' per task specification
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "loan_status": test_preds})

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"  Columns: {submission.columns.tolist()}")
print(f"  First 5 rows:\n{submission.head()}")

# ============================================================
# 9. FINAL VALIDATION SCORE
# ============================================================
score = val_auc
print(f"\nFinal Validation Score: {score}")
