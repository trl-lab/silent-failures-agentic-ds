import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget distribution:\n{train['DEATHS_TOTAL_DESCRIPTION'].value_counts()}")
print(f"\nDtypes:\n{train.dtypes}")
print(f"\nMissing values (train):\n{train.isnull().sum()[train.isnull().sum() > 0]}")

# ============================================================
# 2. IDENTIFY AND PREPARE FEATURES
# ============================================================
TARGET = "DEATHS_TOTAL_DESCRIPTION"
ID_COL = "id"

y = train[TARGET].copy()
train_features = train.drop(columns=[TARGET])

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, is_train=True, le_dict=None, numeric_cols=None):
    """Engineer features from raw data with domain-specific volcanic features."""
    df = df.copy()

    # --- Temporal features ---
    if 'YEAR' in df.columns:
        df['century'] = (df['YEAR'] // 100).astype(int)
        df['era'] = pd.cut(
            df['YEAR'],
            bins=[0, 1500, 1800, 1900, 2000, 2100],
            labels=["pre-1500", "1500-1799", "1800-1899", "1900-1999", "2000+"],
        ).astype(str)
        df['decade'] = (df['YEAR'] // 10 * 10).astype(int)
        # Modern era indicator (better reporting of deaths)
        df['modern_era'] = (df['YEAR'] >= 1900).astype(int)
        df['recent_era'] = (df['YEAR'] >= 1950).astype(int)

    # --- Geographic features ---
    lat_col = 'LATITUDE' if 'LATITUDE' in df.columns else None
    lon_col = 'LONGITUDE' if 'LONGITUDE' in df.columns else None

    if lat_col is not None and lon_col is not None:
        df['abs_lat'] = df[lat_col].abs()
        df['abs_lon'] = df[lon_col].abs()
        df['tropical_zone'] = (df[lat_col].abs() < 23.5).astype(int)
        df['temperate_zone'] = (
            (df[lat_col].abs() >= 23.5) & (df[lat_col].abs() <= 66.5)
        ).astype(int)
        df['polar_zone'] = (df[lat_col].abs() > 66.5).astype(int)
        df['dist_equator_norm'] = df[lat_col].abs() / 90.0

    # --- ELEVATION features ---
    if 'ELEVATION' in df.columns:
        df['elev_log'] = np.log1p(df['ELEVATION'].clip(lower=0))
        df['elev_high'] = (df['ELEVATION'] > 3000).astype(int)
        df['elev_mid'] = ((df['ELEVATION'] >= 1000) & (df['ELEVATION'] <= 3000)).astype(int)

    # --- VEI (Volcanic Explosivity Index) - KEY FEATURE ---
    if 'VEI' in df.columns:
        df['VEI_log'] = np.log1p(df['VEI'].clip(lower=0))
        df['VEI_high'] = (df['VEI'] >= 4).astype(int)

    # --- Damage/Injury/Death related features (log transforms only) ---
    for prefix in ['DAMAGE', 'INJURIES', 'MISSING', 'HOUSES_DESTROYED']:
        value_col = f'{prefix}'
        total_value_col = f'{prefix}_TOTAL'

        if value_col in df.columns:
            df[f'{prefix}_log'] = np.log1p(df[value_col].clip(lower=0))
        if total_value_col in df.columns:
            df[f'{prefix}_total_log'] = np.log1p(df[total_value_col].clip(lower=0))

    # --- Key interaction: VEI with other factors (only high-value ones) ---
    if 'VEI' in df.columns:
        if 'YEAR' in df.columns:
            df['VEI_x_modern'] = df['VEI'] * df['modern_era']
        if 'ELEVATION' in df.columns:
            df['VEI_x_elev'] = df['VEI'] * df['ELEVATION'].clip(lower=0)

    # --- Total damage features ---
    if 'DAMAGE_MILLIONS_DOLLARS_TOTAL' in df.columns:
        df['total_damage_log'] = np.log1p(df['DAMAGE_MILLIONS_DOLLARS_TOTAL'].clip(lower=0))

    # --- NUM_SLIDES (number of slides/flows) ---
    if 'NUM_SLIDES' in df.columns:
        df['num_slides_log'] = np.log1p(df['NUM_SLIDES'].clip(lower=0))

    # --- VALIDITY feature ---
    if 'VALIDITY' in df.columns:
        df['validity'] = df['VALIDITY']

    # --- Categorical encoding (only key categorical features) ---
    key_cat_cols = ['MORPHOLOGY', 'STATUS', 'COUNTRY', 'TIME_ERUPT']
    cat_cols_present = [c for c in key_cat_cols if c in df.columns]
    for col in cat_cols_present:
        if is_train:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            if le_dict is None:
                le_dict = {}
            le_dict[col] = le
        else:
            if le_dict is not None and col in le_dict:
                le = le_dict[col]
                known_classes = set(le.classes_)
                df[col + "_enc"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known_classes else -1
                )
            else:
                df[col + "_enc"] = -1

    # Drop original categorical columns
    for col in cat_cols_present:
        if col + "_enc" in df.columns:
            df = df.drop(columns=[col])

    # Drop all remaining non-numeric, non-ID columns (descriptions, comments, etc.)
    drop_cols = [c for c in df.columns if df[c].dtype == "object" and c != ID_COL]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Drop ID-like columns
    id_like_cols = ['VOL_ID', 'SIG_ID', 'TSU_ID', 'HAZ_EVENT_ID', 'OBJECTID', 'ID', 'x', 'y', 'NUM']
    drop_id_cols = [c for c in id_like_cols if c in df.columns]
    if drop_id_cols:
        df = df.drop(columns=drop_id_cols)

    # Drop ID column if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Fill remaining NaN
    df = df.fillna(-999)

    return df, le_dict


# Apply feature engineering
train_feat, le_dict = engineer_features(
    train_features, is_train=True, numeric_cols=numeric_cols
)
test_feat, _ = engineer_features(
    test, is_train=False, le_dict=le_dict, numeric_cols=numeric_cols
)

# Ensure feature alignment
common_features = [c for c in train_feat.columns if c in test_feat.columns]
train_feat = train_feat[common_features]
test_feat = test_feat[common_features]

print(f"\nEngineered features shape: {train_feat.shape}")
print(f"Number of features: {len(common_features)}")

# ============================================================
# 4. SPLIT TRAIN INTO TRAIN/VALIDATION (stratified)
# ============================================================
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y.astype(str))

train_idx, val_idx = train_test_split(
    range(len(train_feat)), test_size=0.2, random_state=42, stratify=y_encoded
)

X_train = train_feat.iloc[train_idx].reset_index(drop=True)
X_val = train_feat.iloc[val_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}")
print(f"Val class distribution:\n{y_val.value_counts()}")

# ============================================================
# 5. SCALE NUMERIC FEATURES (fit on train only)
# ============================================================
# Note: Tree-based models don't need scaling, but we keep it for consistency
# and it doesn't hurt. Actually, let's skip scaling for tree models as it's unnecessary.
X_train_scaled = X_train
X_val_scaled = X_val
X_test_scaled = test_feat

# ============================================================
# 6. ENCODE TARGETS FOR MODEL TRAINING
# ============================================================
le_target_classes = le_target.classes_
y_train_enc = le_target.transform(y_train.values)
y_val_enc = le_target.transform(y_val.values)
n_classes = len(le_target_classes)

print(f"\nClasses ({n_classes}): {le_target_classes}")

# ============================================================
# 7. TRAIN MULTIPLE SEEDS FOR EACH MODEL TYPE
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]
N_ESTIMATORS = 2000
EARLY_STOPPING_PATIENCE = 300

lgb_models = []
xgb_models = []

for seed in SEEDS:
    # --- LightGBM ---
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        min_child_samples=5,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        verbose=-1,
        class_weight="balanced",
        reg_alpha=0.1,
        reg_lambda=0.1,
    )
    lgb_model.fit(
        X_train_scaled.values,
        y_train_enc,
        eval_set=[(X_val_scaled.values, y_val_enc)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False)],
    )
    lgb_models.append(lgb_model)
    lgb_val_acc = accuracy_score(y_val_enc, lgb_model.predict(X_val_scaled.values))
    print(
        f"  LGBM seed={seed}: best_iter={lgb_model.best_iteration_}, val_acc={lgb_val_acc:.4f}"
    )

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        eval_metric="mlogloss",
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
        tree_method="hist",
        scale_pos_weight=1.0,
    )
    xgb_model.fit(
        X_train_scaled.values,
        y_train_enc,
        eval_set=[(X_val_scaled.values, y_val_enc)],
        verbose=False,
    )
    xgb_models.append(xgb_model)
    xgb_val_acc = accuracy_score(y_val_enc, xgb_model.predict(X_val_scaled.values))
    print(
        f"  XGB  seed={seed}: best_iter={xgb_model.best_iteration}, val_acc={xgb_val_acc:.4f}"
    )

# ============================================================
# 8. EQUAL-WEIGHT ENSEMBLE
# ============================================================
lgb_val_probas = np.array([m.predict_proba(X_val_scaled.values) for m in lgb_models])
xgb_val_probas = np.array([m.predict_proba(X_val_scaled.values) for m in xgb_models])

lgb_val_avg = lgb_val_probas.mean(axis=0)
xgb_val_avg = xgb_val_probas.mean(axis=0)

# Equal-weight ensemble (no tuning on small validation set)
val_proba = 0.5 * lgb_val_avg + 0.5 * xgb_val_avg
val_pred_idx = val_proba.argmax(axis=1)
val_acc = accuracy_score(y_val_enc, val_pred_idx)

lgb_only_acc = accuracy_score(y_val_enc, lgb_val_avg.argmax(axis=1))
xgb_only_acc = accuracy_score(y_val_enc, xgb_val_avg.argmax(axis=1))

print(f"\nLGBM-only accuracy: {lgb_only_acc:.4f}")
print(f"XGB-only accuracy:  {xgb_only_acc:.4f}")
print(f"Equal-weight ensemble validation accuracy: {val_acc:.4f}")

# ============================================================
# 9. TEST INFERENCE
# ============================================================
lgb_test_probas = np.array([m.predict_proba(X_test_scaled.values) for m in lgb_models])
xgb_test_probas = np.array([m.predict_proba(X_test_scaled.values) for m in xgb_models])

lgb_test_avg = lgb_test_probas.mean(axis=0)
xgb_test_avg = xgb_test_probas.mean(axis=0)

# Equal-weight ensemble (no post-processing adjustment)
test_proba = 0.5 * lgb_test_avg + 0.5 * xgb_test_avg
test_pred_idx = test_proba.argmax(axis=1)
test_pred = le_target.inverse_transform(test_pred_idx)

print(f"\nTest prediction distribution:")
for cls in le_target_classes:
    count = (test_pred == cls).sum()
    print(f"  {cls}: {count} ({count/len(test_pred)*100:.1f}%)")

# ============================================================
# 10. GENERATE SUBMISSION
# ============================================================
if ID_COL in test.columns:
    submission_ids = test[ID_COL].values
else:
    submission_ids = range(len(test))

submission = pd.DataFrame({"id": submission_ids, "DEATHS_TOTAL_DESCRIPTION": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 11. FINAL VALIDATION SCORE
# ============================================================
val_pred = le_target.inverse_transform(val_pred_idx)

score = accuracy_score(y_val.values, val_pred)
print(
    f"\nValidation: stratified 80/20 split, metric=accuracy, equal-weight ensemble"
)
print(f"Val prediction distribution:")
for cls in le_target_classes:
    count = (val_pred == cls).sum()
    print(f"  {cls}: {count} ({count/len(val_pred)*100:.1f}%)")
print(f"Final Validation Score: {score}")
