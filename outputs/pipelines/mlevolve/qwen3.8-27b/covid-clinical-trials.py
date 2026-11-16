import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import xgboost as xgb

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "days_to_complete"
id_col = "id" if "id" in train.columns else None

# Separate target
y_full = train[target_col].copy()
train = train.drop(columns=[target_col])

# Identify feature columns
feature_cols = [c for c in train.columns if c != id_col]

# Classify features
cat_cols = (
    train[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
)
num_cols = (
    train[feature_cols]
    .select_dtypes(include=["int64", "float64", "int32", "float32"])
    .columns.tolist()
)
bool_cols = train[feature_cols].select_dtypes(include=["bool"]).columns.tolist()
cat_cols.extend(bool_cols)

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any transformation)
# ============================================================
X_train_full = train.copy()

if id_col and id_col in X_train_full.columns:
    ids_full = X_train_full[id_col].copy()
else:
    ids_full = pd.Series(range(len(X_train_full)))

X_train_part, X_val_part, y_train_part, y_val_part, ids_train, ids_val = (
    train_test_split(X_train_full, y_full, ids_full, test_size=0.25, random_state=42)
)

print(f"Train part: {X_train_part.shape}, Val part: {X_val_part.shape}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")

X_train_proc = X_train_part.copy()
X_val_proc = X_val_part.copy()
X_test_proc = test.copy()

# --- 4a: Handle missing values ---
for col in cat_cols:
    if col in X_train_proc.columns:
        X_train_proc[col] = X_train_proc[col].fillna("MISSING").astype(str)
        X_val_proc[col] = X_val_proc[col].fillna("MISSING").astype(str)
        X_test_proc[col] = X_test_proc[col].fillna("MISSING").astype(str)

num_imputer = SimpleImputer(strategy="median")
num_cols_present = [c for c in num_cols if c in X_train_proc.columns]
if num_cols_present:
    X_train_proc[num_cols_present] = num_imputer.fit_transform(
        X_train_proc[num_cols_present]
    )
    X_val_proc[num_cols_present] = num_imputer.transform(X_val_proc[num_cols_present])
    X_test_proc[num_cols_present] = num_imputer.transform(X_test_proc[num_cols_present])

# --- 4b: Text Feature Extraction for text-heavy columns ---
text_cols = ['Title', 'Conditions', 'Interventions', 'Outcome Measures', 'Locations', 'Sponsor/Collaborators', 'Study Documents']
text_cols_present = [c for c in text_cols if c in X_train_proc.columns]

# Medical keywords to check
medical_keywords = ['covid', 'vaccine', 'treatment', 'phase', 'trial', 'randomized',
                    'placebo', 'control', 'safety', 'efficacy', 'dose', 'injection',
                    'oral', 'intravenous', 'subcutaneous', 'mRNA', 'antibody',
                    'infection', 'pneumonia', 'respiratory', 'hospital', 'recovery',
                    'death', 'mortality', 'morbidity', 'immune', 'serum', 'blood',
                    'nasal', 'spray', 'booster', 'bivalent', 'monovalent', 'variant']

# Build TF-IDF vectorizers on train only
tfidf_vectorizers = {}
for col in text_cols_present:
    tfidf_vectorizers[col] = TfidfVectorizer(max_features=20, stop_words='english',
                                              lowercase=True, sublinear_tf=True)
    tfidf_vectorizers[col].fit(X_train_proc[col])

# Extract text features
for col in text_cols_present:
    # Word count, unique word count, average word length
    for df, name in [(X_train_proc, 'train'), (X_val_proc, 'val'), (X_test_proc, 'test')]:
        texts = df[col].str.lower()
        df[f'{col}_word_count'] = texts.str.split().str.len().fillna(0)
        df[f'{col}_unique_word_count'] = texts.apply(lambda x: len(set(x.split())) if isinstance(x, str) and x != 'MISSING' else 0)
        df[f'{col}_avg_word_len'] = texts.apply(lambda x: np.mean([len(w) for w in x.split()]) if isinstance(x, str) and x != 'MISSING' else 0)
        # Keyword features
        for kw in medical_keywords:
            df[f'{col}_has_{kw}'] = texts.str.contains(kw, na=False).astype(int)

    # TF-IDF features
    X_train_tfidf = tfidf_vectorizers[col].transform(X_train_proc[col]).toarray()
    X_val_tfidf = tfidf_vectorizers[col].transform(X_val_proc[col]).toarray()
    X_test_tfidf = tfidf_vectorizers[col].transform(X_test_proc[col]).toarray()

    tfidf_feature_names = [f'{col}_tfidf_{i}' for i in range(X_train_tfidf.shape[1])]
    X_train_proc[tfidf_feature_names] = X_train_tfidf
    X_val_proc[tfidf_feature_names] = X_val_tfidf
    X_test_proc[tfidf_feature_names] = X_test_tfidf

# --- 4c: Date-based features ---
date_cols = ['First Posted', 'Results First Posted', 'Last Update Posted']
for col in date_cols:
    if col in X_train_proc.columns:
        for df in [X_train_proc, X_val_proc, X_test_proc]:
            parsed = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = parsed.dt.year.fillna(0).astype(int)
            df[f'{col}_month'] = parsed.dt.month.fillna(0).astype(int)
            df[f'{col}_dayofweek'] = parsed.dt.dayofweek.fillna(0).astype(int)

# --- 4d: Enrollment bucket features ---
if 'Enrollment' in X_train_proc.columns:
    X_train_proc['Enrollment_log'] = np.log1p(X_train_proc['Enrollment'].clip(lower=0))
    X_val_proc['Enrollment_log'] = np.log1p(X_val_proc['Enrollment'].clip(lower=0))
    X_test_proc['Enrollment_log'] = np.log1p(X_test_proc['Enrollment'].clip(lower=0))

    # Enrollment range buckets
    enrollment_bins = [0, 10, 50, 100, 500, 1000, 10000, np.inf]
    enrollment_labels = ['very_small', 'small', 'medium', 'large', 'very_large', 'huge', 'massive']
    X_train_proc['Enrollment_range'] = pd.cut(X_train_proc['Enrollment'], bins=enrollment_bins, labels=enrollment_labels, right=False).astype(str)
    X_val_proc['Enrollment_range'] = pd.cut(X_val_proc['Enrollment'], bins=enrollment_bins, labels=enrollment_labels, right=False).astype(str)
    X_test_proc['Enrollment_range'] = pd.cut(X_test_proc['Enrollment'], bins=enrollment_bins, labels=enrollment_labels, right=False).astype(str)

# --- 4e: Frequency Encoding for Status and Study Type (and other short categoricals) ---
short_cat_cols = [c for c in cat_cols if c in X_train_proc.columns and c not in text_cols_present]
for col in short_cat_cols:
    freq_map = X_train_proc[col].value_counts(normalize=True).to_dict()
    X_train_proc[f"{col}_freq"] = X_train_proc[col].map(freq_map).fillna(0)
    X_val_proc[f"{col}_freq"] = X_val_proc[col].map(freq_map).fillna(0)
    X_test_proc[f"{col}_freq"] = X_test_proc[col].map(freq_map).fillna(0)

# --- 4f: Target Encoding with smoothing (fit ONLY on X_train_part) ---
global_mean = y_train_part.mean()
for col in short_cat_cols:
    if col not in X_train_proc.columns:
        continue
    agg = y_train_part.groupby(X_train_proc[col]).agg(["mean", "count"])
    min_samples = 10
    smoothed = (agg["mean"] * agg["count"] + global_mean * min_samples) / (
        agg["count"] + min_samples
    )
    te_map = smoothed.to_dict()
    X_train_proc[f"{col}_target_enc"] = X_train_proc[col].map(te_map).fillna(global_mean)
    X_val_proc[f"{col}_target_enc"] = X_val_proc[col].map(te_map).fillna(global_mean)
    X_test_proc[f"{col}_target_enc"] = X_test_proc[col].map(te_map).fillna(global_mean)

    std_agg = y_train_part.groupby(X_train_proc[col]).std()
    std_map = std_agg.to_dict()
    X_train_proc[f"{col}_target_std"] = X_train_proc[col].map(std_map).fillna(y_train_part.std())
    X_val_proc[f"{col}_target_std"] = X_val_proc[col].map(std_map).fillna(y_train_part.std())
    X_test_proc[f"{col}_target_std"] = X_test_proc[col].map(std_map).fillna(y_train_part.std())

# --- 4g: Numerical Feature Engineering ---
if len(num_cols_present) > 0:
    for col in num_cols_present:
        if X_train_proc[col].min() >= 0:
            X_train_proc[f"{col}_log"] = np.log1p(X_train_proc[col])
            X_val_proc[f"{col}_log"] = np.log1p(X_val_proc[col])
            X_test_proc[f"{col}_log"] = np.log1p(X_test_proc[col])

# --- 4h: Label Encode remaining categorical columns (handle unseen categories) ---
for col in X_train_proc.select_dtypes(include=["object", "category"]).columns.tolist():
    train_vals = X_train_proc[col].astype(str)
    unique_vals = sorted(train_vals.unique())
    val_to_code = {v: i for i, v in enumerate(unique_vals)}
    unknown_code = len(unique_vals)

    X_train_proc[col] = train_vals.map(val_to_code).fillna(unknown_code).astype(int)
    X_val_proc[col] = X_val_proc[col].astype(str).map(val_to_code).fillna(unknown_code).astype(int)
    X_test_proc[col] = X_test_proc[col].astype(str).map(val_to_code).fillna(unknown_code).astype(int)

print(f"Processed train shape: {X_train_proc.shape}")
print(f"Processed val shape: {X_val_proc.shape}")
print(f"Processed test shape: {X_test_proc.shape}")

# ============================================================
# 5. TARGET TRANSFORMATION
# ============================================================
skewness = y_full.skew()
print(f"Target skewness: {skewness:.3f}")

if skewness > 0.5:
    use_log_target = True
    y_train_log = np.log1p(y_train_part)
    y_val_log = np.log1p(y_val_part)
    print("Using log1p target transformation")
else:
    use_log_target = False
    y_train_log = y_train_part.copy()
    y_val_log = y_val_part.copy()
    print("Using raw target (low skewness)")

# ============================================================
# 6. TRAIN 3-MODEL ENSEMBLE
# ============================================================
print("\nTraining 3-model ensemble...")

# Model 1: LightGBM tuned for small data
lgb_params_1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 5000,
    "random_state": 42,
}

# Model 2: LightGBM with stronger regularization
lgb_params_2 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "verbose": -1,
    "n_estimators": 5000,
    "random_state": 123,
}

# Model 3: XGBoost with early stopping
xgb_params_3 = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": 456,
    "tree_method": "hist",
    "verbosity": 0,
}

# Train Model 1
print("Training Model 1 (LGBM small-data tuned)...")
model_1 = lgb.LGBMRegressor(**lgb_params_1)
model_1.fit(
    X_train_proc, y_train_log,
    eval_set=[(X_val_proc, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=500)],
)
best_iter_1 = model_1.best_iteration_ if hasattr(model_1, 'best_iteration_') and model_1.best_iteration_ > 0 else model_1.n_estimators
print(f"  Best iteration: {best_iter_1}, Best val RMSE: {model_1.best_score_['valid_0']['rmse']:.4f}")

# Train Model 2
print("Training Model 2 (LGBM strong regularization)...")
model_2 = lgb.LGBMRegressor(**lgb_params_2)
model_2.fit(
    X_train_proc, y_train_log,
    eval_set=[(X_val_proc, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=500)],
)
best_iter_2 = model_2.best_iteration_ if hasattr(model_2, 'best_iteration_') and model_2.best_iteration_ > 0 else model_2.n_estimators
print(f"  Best iteration: {best_iter_2}, Best val RMSE: {model_2.best_score_['valid_0']['rmse']:.4f}")

# Train Model 3
print("Training Model 3 (XGBoost)...")
dtrain = xgb.DMatrix(X_train_proc, label=y_train_log)
dval = xgb.DMatrix(X_val_proc, label=y_val_log)
dtest = xgb.DMatrix(X_test_proc)

model_3 = xgb.train(
    xgb_params_3, dtrain,
    num_boost_round=5000,
    evals=[(dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=500,
)
best_score_3 = model_3.best_score if isinstance(model_3.best_score, (int, float)) else list(model_3.best_score.values())[0]
print(f"  Best iteration: {model_3.best_iteration}, Best val RMSE: {best_score_3:.4f}")

# ============================================================
# 7. VALIDATION INFERENCE (Ensemble)
# ============================================================
# Get individual model predictions
val_pred_1 = model_1.predict(X_val_proc, num_iteration=best_iter_1)
val_pred_2 = model_2.predict(X_val_proc, num_iteration=best_iter_2)
val_pred_3 = model_3.predict(dval, iteration_range=(0, model_3.best_iteration + 1))

# Compute individual validation RMSEs (on log scale for weighting)
rmse_1 = np.sqrt(mean_squared_error(y_val_log, val_pred_1))
rmse_2 = np.sqrt(mean_squared_error(y_val_log, val_pred_2))
rmse_3 = np.sqrt(mean_squared_error(y_val_log, val_pred_3))

print(f"\nIndividual model validation RMSEs (log scale):")
print(f"  Model 1 (LGBM tuned): {rmse_1:.4f}")
print(f"  Model 2 (LGBM reg):   {rmse_2:.4f}")
print(f"  Model 3 (XGBoost):    {rmse_3:.4f}")

# Weights proportional to inverse validation RMSE
inv_rmse = np.array([1.0/rmse_1, 1.0/rmse_2, 1.0/rmse_3])
weights = inv_rmse / inv_rmse.sum()
print(f"  Ensemble weights: [{weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f}]")

# Weighted ensemble prediction
val_pred_ensemble = weights[0] * val_pred_1 + weights[1] * val_pred_2 + weights[2] * val_pred_3

if use_log_target:
    val_pred_raw = np.expm1(val_pred_ensemble)
else:
    val_pred_raw = val_pred_ensemble

# Clip to non-negative
val_pred_raw = np.maximum(val_pred_raw, 0.0)

# Official metric: RMSE on raw scale
val_rmse = np.sqrt(mean_squared_error(y_val_part, val_pred_raw))
print(f"\nEnsemble Validation RMSE (raw scale): {val_rmse:.4f}")

# ============================================================
# 8. TEST INFERENCE (identical processing to validation)
# ============================================================
test_pred_1 = model_1.predict(X_test_proc, num_iteration=best_iter_1)
test_pred_2 = model_2.predict(X_test_proc, num_iteration=best_iter_2)
test_pred_3 = model_3.predict(dtest, iteration_range=(0, model_3.best_iteration + 1))

test_pred_ensemble = weights[0] * test_pred_1 + weights[1] * test_pred_2 + weights[2] * test_pred_3

if use_log_target:
    test_pred_raw = np.expm1(test_pred_ensemble)
else:
    test_pred_raw = test_pred_ensemble

# Same clipping as validation
test_pred_raw = np.maximum(test_pred_raw, 0.0)

print(f"Test predictions: mean={test_pred_raw.mean():.2f}, min={test_pred_raw.min():.2f}, max={test_pred_raw.max():.2f}")

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
if id_col and id_col in test.columns:
    ids_test = test[id_col].values
else:
    ids_test = np.arange(len(test))

submission = pd.DataFrame(
    {"id": ids_test.astype(int), "days_to_complete": test_pred_raw}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 10. FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score}")