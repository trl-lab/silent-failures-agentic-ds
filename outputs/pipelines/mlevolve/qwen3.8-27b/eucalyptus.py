import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Target distribution:\n{train['Utility'].value_counts()}")
print(f"Missing values in train:\n{train.isnull().sum()}")
print(f"Missing values in test:\n{test.isnull().sum()}")

# Identify feature columns (exclude target)
target_col = "Utility"
feature_cols = [c for c in train.columns if c != target_col]

# Separate target
y = train[target_col].copy()
X = train.drop(columns=[target_col]).copy()
X_test = test.copy()

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")


# === FEATURE ENGINEERING (Simplified - no groupby aggregates to avoid leakage) ===
def engineer_features(df, is_train=True, train_stats=None):
    """Apply simplified feature engineering - only safe features."""
    df = df.copy()

    # --- Missing value indicators (domain: absence has meaning) ---
    for col in ["Surv", "Vig", "Ins_res", "Stem_Fm", "Crown_Fm", "Brnch_Fm"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # --- Key interactions between raw measurements ---
    if "Vig" in df.columns and "DBH" in df.columns:
        df["Vig_x_DBH"] = df["Vig"] * df["DBH"]
    if "Vig" in df.columns and "Ht" in df.columns:
        df["Vig_x_Ht"] = df["Vig"] * df["Ht"]
    if "DBH" in df.columns and "Ht" in df.columns:
        df["DBH_to_Ht_ratio"] = df["DBH"] / (df["Ht"] + 1e-6)
    if "Stem_Fm" in df.columns and "Crown_Fm" in df.columns:
        df["Stem_x_Crown"] = df["Stem_Fm"] * df["Crown_Fm"]

    # --- New domain-informed interaction features ---
    if "Vig" in df.columns:
        df["Vig_sq"] = df["Vig"] ** 2
    if "Vig" in df.columns and "Stem_Fm" in df.columns and "Crown_Fm" in df.columns:
        df["composite_quality"] = (df["Vig"] + df["Stem_Fm"] + df["Crown_Fm"]) / 3.0
    if "Rainfall" in df.columns and "Vig" in df.columns:
        df["Rainfall_x_Vig"] = df["Rainfall"] * df["Vig"]

    # --- Temporal feature: is_late_trial binary ---
    if "Year" in df.columns:
        if is_train:
            year_median = df["Year"].median()
            train_stats["year_median"] = year_median
        else:
            year_median = train_stats.get("year_median", 0)
        df["is_late_trial"] = (df["Year"] >= year_median).astype(int)

    return df, train_stats


# --- Encode categorical features (label encoding for tree models) ---
# Fit encoders on full training data (all 588 samples) since we'll use CV
cat_cols_final = X.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}

for col in cat_cols_final:
    le = pd.Series(X[col].astype(str).unique(), name="categories").reset_index()
    le.columns = ["category", "encoded"]
    le["encoded"] = range(len(le))
    label_encoders[col] = le.set_index("category")["encoded"].to_dict()

    X[col] = X[col].astype(str).map(label_encoders[col]).fillna(-1).astype(int)
    if col in X_test.columns:
        X_test[col] = (
            X_test[col].astype(str).map(label_encoders[col]).fillna(-1).astype(int)
        )

# Apply feature engineering: fit stats on full training data, apply to all
train_stats = {"year_median": X["Year"].median() if "Year" in X.columns else 0}
X, train_stats = engineer_features(X, is_train=True, train_stats=train_stats)
X_test, _ = engineer_features(X_test, is_train=False, train_stats=train_stats)

print(f"\nAfter feature engineering:")
print(f"Train features: {X.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# === MODEL DESIGN ===
# LightGBM with strong regularization for small dataset
n_classes = len(y.unique())
print(f"\nNumber of classes: {n_classes}")
print(f"Classes: {sorted(y.unique())}")

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 4,
    "min_data_in_leaf": 25,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 15.0,
    "min_gain_to_split": 0.5,
    "verbose": -1,
    "seed": 42,
}

# === ENCODE TARGET LABELS TO INTEGERS ===
target_classes = sorted(y.unique())
target_label_to_int = {cls: idx for idx, cls in enumerate(target_classes)}
target_int_to_label = {idx: cls for cls, idx in target_label_to_int.items()}

y_enc = y.map(target_label_to_int).astype(int)

print(f"Target label mapping: {target_label_to_int}")

# === XGBOOST PARAMETERS ===
xgb_params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "eval_metric": "mlogloss",
    "seed": 42,
    "verbosity": 0,
}

# === 5-FOLD STRATIFIED CROSS-VALIDATION WITH ENSEMBLE ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []
cv_lgb_best_iterations = []
cv_xgb_best_iterations = []
cv_oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    X_fold_train = X.iloc[train_idx].copy()
    X_fold_val = X.iloc[val_idx].copy()
    y_fold_train = y_enc.iloc[train_idx]
    y_fold_val = y_enc.iloc[val_idx]

    # --- Species-level aggregate features (computed ONLY on training fold to avoid leakage) ---
    sp = "Sp"
    if sp in X_fold_train.columns:
        for agg_col in ["Vig", "Surv", "Ht"]:
            if agg_col in X_fold_train.columns:
                sp_mean = X_fold_train.groupby(sp)[agg_col].transform("mean")
                X_fold_train[f"Sp_mean_{agg_col}"] = X_fold_train[sp].map(sp_mean)
                X_fold_val[f"Sp_mean_{agg_col}"] = X_fold_val[sp].map(sp_mean)

    # --- LightGBM ---
    dtrain = lgb.Dataset(X_fold_train, label=y_fold_train)
    dval = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain)

    fold_lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )

    lgb_best_iter = fold_lgb_model.best_iteration
    cv_lgb_best_iterations.append(lgb_best_iter)
    lgb_val_proba = fold_lgb_model.predict(X_fold_val, num_iteration=lgb_best_iter)

    # --- XGBoost ---
    dtrain_xgb = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval_xgb = xgb.DMatrix(X_fold_val, label=y_fold_val)

    fold_xgb_model = xgb.train(
        xgb_params,
        dtrain_xgb,
        num_boost_round=2000,
        evals=[(dval_xgb, "valid")],
        early_stopping_rounds=100,
    )

    xgb_best_iter = fold_xgb_model.best_iteration
    cv_xgb_best_iterations.append(xgb_best_iter)
    xgb_val_proba = fold_xgb_model.predict(dval_xgb, iteration_range=(0, xgb_best_iter))

    # --- Ensemble: average probabilities ---
    ensemble_val_proba = (lgb_val_proba + xgb_val_proba) / 2.0
    fold_val_pred_int = np.argmax(ensemble_val_proba, axis=1)
    fold_val_pred = np.array([target_int_to_label[p] for p in fold_val_pred_int])
    fold_acc = accuracy_score(y.iloc[val_idx], fold_val_pred)
    cv_accuracies.append(fold_acc)
    cv_oof_preds[val_idx] = fold_val_pred_int

    print(f"  Fold {fold+1}: lgb_iter={lgb_best_iter}, xgb_iter={xgb_best_iter}, accuracy={fold_acc:.4f}")

cv_mean_acc = np.mean(cv_accuracies)
cv_std_acc = np.std(cv_accuracies)
median_lgb_iter = int(np.median(cv_lgb_best_iterations))
median_xgb_iter = int(np.median(cv_xgb_best_iterations))

print(f"\nCV Accuracy: {cv_mean_acc:.4f} (+/- {cv_std_acc:.4f})")
print(f"LGB Best Iterations: {cv_lgb_best_iterations}")
print(f"XGB Best Iterations: {cv_xgb_best_iterations}")
print(f"Median LGB Iteration: {median_lgb_iter}")
print(f"Median XGB Iteration: {median_xgb_iter}")

# === TRAIN FINAL MODELS ON ALL DATA ===
# Add species-level features to full training data for final models
sp = "Sp"
if sp in X.columns:
    for agg_col in ["Vig", "Surv", "Ht"]:
        if agg_col in X.columns:
            sp_mean = X.groupby(sp)[agg_col].transform("mean")
            X[f"Sp_mean_{agg_col}"] = sp_mean
            X_test[f"Sp_mean_{agg_col}"] = X_test[sp].map(sp_mean)

# Final LightGBM
dtrain_full = lgb.Dataset(X, label=y_enc)
final_lgb_model = lgb.train(
    lgb_params,
    dtrain_full,
    num_boost_round=median_lgb_iter,
    callbacks=[lgb.log_evaluation(500)],
)

# Final XGBoost
dtrain_full_xgb = xgb.DMatrix(X, label=y_enc)
final_xgb_model = xgb.train(
    xgb_params,
    dtrain_full_xgb,
    num_boost_round=median_xgb_iter,
)

print(f"\nFinal LGB model trained on all {len(X)} samples with {median_lgb_iter} rounds")
print(f"Final XGB model trained on all {len(X)} samples with {median_xgb_iter} rounds")

# === TEST PREDICTIONS (ENSEMBLE) ===
lgb_test_proba = final_lgb_model.predict(X_test, num_iteration=median_lgb_iter)
dtest_xgb = xgb.DMatrix(X_test)
xgb_test_proba = final_xgb_model.predict(dtest_xgb, iteration_range=(0, median_xgb_iter - 1))

test_pred_proba = (lgb_test_proba + xgb_test_proba) / 2.0
test_pred_int = np.argmax(test_pred_proba, axis=1)
test_pred = np.array([target_int_to_label[p] for p in test_pred_int])

# === GENERATE SUBMISSION ===
submission = pd.DataFrame({"id": range(len(test_pred)), "Utility": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Predicted class distribution:\n{pd.Series(test_pred).value_counts().sort_index()}"
)
print(f"Submission head:\n{submission.head()}")

# === FEATURE IMPORTANCE ===
importance = final_lgb_model.feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
print(f"\nTop 10 features by gain:")
print(feat_imp.head(10))

# === SAVE MODELS ===
final_lgb_model.save_model("./working/lgbm_best_model.txt")
final_xgb_model.save_model("./working/xgb_best_model.json")
print("\nModels saved to ./working/")

# === FINAL SCORE ===
score = cv_mean_acc
print(f"Final Validation Score: {score}")