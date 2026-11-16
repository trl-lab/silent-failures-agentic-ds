import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
TARGET = "Average or Greater than 30"
ID_COL = "id"

if ID_COL in train_df.columns:
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values
else:
    train_ids = np.arange(len(train_df))
    test_ids = np.arange(len(test_df))

y = train_df[TARGET].values
train_df = train_df.drop(columns=[TARGET])

if ID_COL in train_df.columns:
    train_df = train_df.drop(columns=[ID_COL])
if ID_COL in test_df.columns:
    test_df = test_df.drop(columns=[ID_COL])

feature_cols = list(train_df.columns)
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT (BEFORE any processing)
# ============================================================
train_features, val_features, y_train, y_val = train_test_split(
    train_df, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train split: {train_features.shape}, Val split: {val_features.shape}")
print(f"Train target rate: {y_train.mean():.4f}, Val target rate: {y_val.mean():.4f}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nPerforming feature engineering...")

numeric_cols = [
    c
    for c in train_features.columns
    if train_features[c].dtype in ["int64", "float64", "int32", "float32"]
]
categorical_cols = [
    c
    for c in train_features.columns
    if train_features[c].dtype == "object" or train_features[c].dtype.name == "category"
]
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# --- Frequency encoding (fit on train only) ---
freq_encodings = {}
for col in categorical_cols:
    freq_encodings[col] = train_features[col].value_counts(normalize=True)


def apply_freq_encoding(df, freq_encodings, categorical_cols):
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(freq_encodings[col]).fillna(0)
    return df


train_features = apply_freq_encoding(train_features, freq_encodings, categorical_cols)
val_features = apply_freq_encoding(val_features, freq_encodings, categorical_cols)
test_df = apply_freq_encoding(test_df, freq_encodings, categorical_cols)

# --- One-hot encode low-cardinality categoricals ---
low_card_cats = [c for c in categorical_cols if train_features[c].nunique() <= 15]
high_card_cats = [c for c in categorical_cols if train_features[c].nunique() > 15]
print(f"Low-cardinality cats (one-hot): {low_card_cats}")
print(f"High-cardinality cats (freq only): {high_card_cats}")

if low_card_cats:
    ohe_train = pd.get_dummies(
        train_features[low_card_cats], prefix=low_card_cats, drop_first=False
    )
    ohe_val = pd.get_dummies(
        val_features[low_card_cats], prefix=low_card_cats, drop_first=False
    )
    ohe_test = pd.get_dummies(
        test_df[low_card_cats], prefix=low_card_cats, drop_first=False
    )

    all_ohe_cols = ohe_train.columns.tolist()
    for col in all_ohe_cols:
        if col not in ohe_val.columns:
            ohe_val[col] = 0
        if col not in ohe_test.columns:
            ohe_test[col] = 0
    ohe_val = ohe_val[all_ohe_cols]
    ohe_test = ohe_test[all_ohe_cols]

    train_features = train_features.drop(columns=low_card_cats).join(ohe_train)
    val_features = val_features.drop(columns=low_card_cats).join(ohe_val)
    test_df = test_df.drop(columns=low_card_cats).join(ohe_test)

# Drop high-cardinality categoricals (keep their freq encoding)
if high_card_cats:
    train_features = train_features.drop(columns=high_card_cats)
    val_features = val_features.drop(columns=high_card_cats)
    test_df = test_df.drop(columns=high_card_cats)

# --- Missing value handling with indicators ---
missing_cols = [c for c in train_features.columns if train_features[c].isnull().any()]
print(f"Columns with missing values: {len(missing_cols)}")

if missing_cols:
    for col in missing_cols:
        train_features[f"{col}_missing"] = train_features[col].isnull().astype(int)
        val_features[f"{col}_missing"] = val_features[col].isnull().astype(int)
        test_df[f"{col}_missing"] = test_df[col].isnull().astype(int)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_features[missing_cols])
    train_features[missing_cols] = imputer.transform(train_features[missing_cols])
    val_features[missing_cols] = imputer.transform(val_features[missing_cols])
    test_df[missing_cols] = imputer.transform(test_df[missing_cols])

# --- Ratio features ---
print("\nCreating ratio and interaction features...")
num_cols = [
    c
    for c in train_features.columns
    if train_features[c].dtype in ["int64", "float64", "int32", "float32"]
]

ratio_features = []
for i, c1 in enumerate(num_cols):
    for c2 in num_cols[i + 1 :]:
        if train_features[c1].std() > 0 and train_features[c2].std() > 0:
            new_col = f"{c1}_div_{c2}"
            ratio_features.append(new_col)
            train_features[new_col] = train_features[c1] / (
                train_features[c2].abs() + 1e-8
            )
            val_features[new_col] = val_features[c1] / (val_features[c2].abs() + 1e-8)
            test_df[new_col] = test_df[c1] / (test_df[c2].abs() + 1e-8)

# Limit ratio features if too many
if len(ratio_features) > 50:
    variances = train_features[ratio_features].var().sort_values(ascending=False)
    top_ratios = variances.head(50).index.tolist()
    dropped = [c for c in ratio_features if c not in top_ratios]
    train_features = train_features.drop(columns=dropped)
    val_features = val_features.drop(columns=dropped)
    test_df = test_df.drop(columns=dropped)
    ratio_features = top_ratios

print(f"Created {len(ratio_features)} ratio features")

# --- Interaction features (products of top correlated features) ---
if len(num_cols) > 1:
    corr_with_target = (
        train_features[num_cols]
        .corrwith(pd.Series(y_train, index=train_features.index))
        .abs()
        .sort_values(ascending=False)
    )
    top_feats = corr_with_target.head(min(10, len(num_cols))).index.tolist()

    for i, c1 in enumerate(top_feats):
        for c2 in top_feats[i + 1 :]:
            new_col = f"{c1}_x_{c2}"
            train_features[new_col] = train_features[c1] * train_features[c2]
            val_features[new_col] = val_features[c1] * val_features[c2]
            test_df[new_col] = test_df[c1] * test_df[c2]

# --- Log transform for skewed features ---
skewed_cols = [
    c
    for c in train_features.columns
    if train_features[c].dtype in ["int64", "float64", "int32", "float32"]
    and train_features[c].skew() > 2
    and (train_features[c] > 0).all()
]
for col in skewed_cols:
    train_features[f"{col}_log"] = np.log1p(train_features[col])
    val_features[f"{col}_log"] = np.log1p(val_features[col])
    test_df[f"{col}_log"] = np.log1p(test_df[col])

if skewed_cols:
    print(f"Applied log transform to {len(skewed_cols)} skewed features")

# --- Final cleanup ---
train_features = train_features.astype(float)
val_features = val_features.astype(float)
test_df = test_df.astype(float)

train_features = train_features.replace([np.inf, -np.inf], 0)
val_features = val_features.replace([np.inf, -np.inf], 0)
test_df = test_df.replace([np.inf, -np.inf], 0)

# Fill any remaining NaNs
train_features = train_features.fillna(0)
val_features = val_features.fillna(0)
test_df = test_df.fillna(0)

assert not train_features.isnull().any().any(), "Train still has NaNs!"
assert not val_features.isnull().any().any(), "Val still has NaNs!"
assert not test_df.isnull().any().any(), "Test still has NaNs!"

print(f"\nFinal shapes:")
print(f"  Train: {train_features.shape}")
print(f"  Val:   {val_features.shape}")
print(f"  Test:  {test_df.shape}")

# ============================================================
# 5. FEATURE SELECTION via Mutual Information
# ============================================================
print("\nPerforming feature selection...")
mi_scores = mutual_info_classif(
    train_features.fillna(0), y_train, random_state=42, n_neighbors=5
)
mi_df = pd.DataFrame({"feature": train_features.columns, "mi_score": mi_scores})
mi_df = mi_df.sort_values("mi_score", ascending=False)

threshold = max(mi_df["mi_score"].quantile(0.4), 0.001)
selected_features = mi_df[mi_df["mi_score"] >= threshold]["feature"].tolist()

if len(selected_features) < 10:
    selected_features = mi_df.head(10)["feature"].tolist()

print(f"Selected {len(selected_features)} / {train_features.shape[1]} features")
print(f"Top 10 by MI: {mi_df.head(10)['feature'].tolist()}")

X_train = train_features[selected_features]
X_val = val_features[selected_features]
X_test = test_df[selected_features]

# ============================================================
# 6. MULTI-SEED XGBoost TRAINING WITH EARLY STOPPING
# ============================================================
print("\nTraining multi-seed XGBoost ensemble...")

# Base parameters
base_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "gamma": 0.1,
    "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}

seeds = [42, 123, 456, 789, 1024]
val_preds_all = []
test_preds_all = []
seed_aucs = []

for seed in seeds:
    params = base_params.copy()
    params["random_state"] = seed
    params["n_estimators"] = 2000
    params["early_stopping_rounds"] = 100

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    seed_auc = roc_auc_score(y_val, val_pred)
    seed_aucs.append(seed_auc)
    val_preds_all.append(val_pred)
    test_preds_all.append(test_pred)

    print(f"  Seed {seed}: best_iter={model.best_iteration}, val_auc={seed_auc:.4f}")

# ============================================================
# 7. ENSEMBLE AND EVALUATION
# ============================================================
val_preds_ensemble = np.mean(val_preds_all, axis=0)
test_preds_ensemble = np.mean(test_preds_all, axis=0)

ensemble_auc = roc_auc_score(y_val, val_preds_ensemble)
print(f"\nSingle-seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}")
print(f"Mean single-seed AUC: {np.mean(seed_aucs):.4f}")
print(f"Ensemble (5-seed avg) AUC: {ensemble_auc:.4f}")

# ============================================================
# 8. GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Average or Greater than 30": test_preds_ensemble}
)

submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 9. FINAL SCORE
# ============================================================
score = ensemble_auc
print(f"\nValidation Setup: Stratified 80/20 split, metric=ROC AUC")
print(
    f"Model: XGBoost (5-seed ensemble, interaction constraints, MI feature selection)"
)
print(f"Final Validation Score: {score}")
