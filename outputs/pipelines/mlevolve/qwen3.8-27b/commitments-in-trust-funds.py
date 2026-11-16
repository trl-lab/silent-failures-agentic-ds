import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import json
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

target_col = "New Commitments (US$)"
print(
    f"Target stats: mean={train[target_col].mean():.2f}, median={train[target_col].median():.2f}, "
    f"std={train[target_col].std():.2f}, zeros={(train[target_col] == 0).mean()*100:.1f}%"
)

# ============================================================
# 2. CREATE TRAIN/VAL SPLIT FIRST (before any processing)
# ============================================================
print("\nCreating train/validation split...")
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)
print(f"Train split: {train_data.shape}, Val split: {val_data.shape}")

# ============================================================
# 3. IDENTIFY COLUMN TYPES
# ============================================================
y_train = train_data[target_col].values
y_val = val_data[target_col].values

X_train = train_data.drop(columns=[target_col])
X_val = val_data.drop(columns=[target_col])
X_test = test.copy()

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

id_cols = [
    c
    for c in X_train.columns
    if "id" in c.lower() or "index" in c.lower() or "name" in c.lower()
]
print(f"Potential ID columns: {id_cols}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")

# --- 4a: Frequency Encoding for Categoricals (fit on train only) ---
for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True)
    X_train[col + "_freq"] = X_train[col].map(freq).fillna(0)
    X_val[col + "_freq"] = X_val[col].map(freq).fillna(0)
    X_test[col + "_freq"] = X_test[col].map(freq).fillna(0)

    count = X_train[col].value_counts()
    X_train[col + "_count"] = X_train[col].map(count).fillna(0)
    X_val[col + "_count"] = X_val[col].map(count).fillna(0)
    X_test[col + "_count"] = X_test[col].map(count).fillna(0)

# --- 4b: Log transform for skewed numerical features ---
skewed_cols = []
for col in num_cols:
    if col in id_cols:
        continue
    non_null = X_train[col].dropna()
    if len(non_null) > 0 and non_null.std() > 0:
        skew = non_null.skew()
        if abs(skew) > 1.0:
            skewed_cols.append(col)
            X_train[col + "_log"] = np.sign(X_train[col]) * np.log1p(
                np.abs(X_train[col])
            )
            X_val[col + "_log"] = np.sign(X_val[col]) * np.log1p(np.abs(X_val[col]))
            X_test[col + "_log"] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))

print(f"Skewed columns log-transformed: {skewed_cols}")

# --- 4c: Interaction features between key categoricals ---
if len(cat_cols) >= 2:
    cat_cardinalities = {col: X_train[col].nunique() for col in cat_cols}
    top_cats = sorted(cat_cols, key=lambda c: cat_cardinalities[c])[:5]

    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            col_name = f"{top_cats[i]}_x_{top_cats[j]}"
            X_train[col_name] = (
                X_train[top_cats[i]].astype(str)
                + "_"
                + X_train[top_cats[j]].astype(str)
            )
            X_val[col_name] = (
                X_val[top_cats[i]].astype(str) + "_" + X_val[top_cats[j]].astype(str)
            )
            X_test[col_name] = (
                X_test[top_cats[i]].astype(str) + "_" + X_test[top_cats[j]].astype(str)
            )

    print(f"Created {(len(top_cats)*(len(top_cats)-1))//2} interaction features")

# --- 4d: Target aggregation features (fit on train only) ---
if len(cat_cols) > 0:
    for col in cat_cols[:5]:
        group_mean = train_data.groupby(col)[target_col].mean()
        group_std = train_data.groupby(col)[target_col].std()
        group_count = train_data.groupby(col)[target_col].count()

        X_train[col + "_target_mean"] = (
            X_train[col].map(group_mean).fillna(train_data[target_col].mean())
        )
        X_val[col + "_target_mean"] = (
            X_val[col].map(group_mean).fillna(train_data[target_col].mean())
        )
        X_test[col + "_target_mean"] = (
            X_test[col].map(group_mean).fillna(train_data[target_col].mean())
        )

        X_train[col + "_target_std"] = X_train[col].map(group_std).fillna(0)
        X_val[col + "_target_std"] = X_val[col].map(group_std).fillna(0)
        X_test[col + "_target_std"] = X_test[col].map(group_std).fillna(0)

        X_train[col + "_group_count"] = X_train[col].map(group_count).fillna(0)
        X_val[col + "_group_count"] = X_val[col].map(group_count).fillna(0)
        X_test[col + "_group_count"] = X_test[col].map(group_count).fillna(0)

    print("Created target aggregation features (fit on train only)")

# --- 4e: Handle missing values ---
for col in num_cols:
    if X_train[col].isnull().sum() > 0:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_val[col] = X_val[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

for col in cat_cols:
    if X_train[col].isnull().sum() > 0:
        X_train[col] = X_train[col].fillna("MISSING")
        X_val[col] = X_val[col].fillna("MISSING")
        X_test[col] = X_test[col].fillna("MISSING")

# Fill missing in newly created features
for col in X_train.columns:
    if X_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        if X_train[col].isnull().sum() > 0:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
    elif X_train[col].dtype == object:
        if X_train[col].isnull().sum() > 0:
            X_train[col] = X_train[col].fillna("MISSING")
            X_val[col] = X_val[col].fillna("MISSING")
            X_test[col] = X_test[col].fillna("MISSING")

print("Handled missing values.")

# ============================================================
# 5. FINAL FEATURE MATRIX PREPARATION
# ============================================================
for col in X_train.columns:
    if X_train[col].dtype == object or X_train[col].dtype.name == "category":
        all_cats = pd.unique(pd.concat([X_train[col], X_val[col], X_test[col]]))
        cat_map = {c: i for i, c in enumerate(sorted(all_cats))}
        X_train[col] = X_train[col].map(cat_map).fillna(-1).astype(int)
        X_val[col] = X_val[col].map(cat_map).fillna(-1).astype(int)
        X_test[col] = X_test[col].map(cat_map).fillna(-1).astype(int)

# Ensure no non-numeric columns remain
non_numeric = [c for c in X_train.columns if X_train[c].dtype == object]
if non_numeric:
    print(f"Warning: Non-numeric columns remaining: {non_numeric}")
    for col in non_numeric:
        X_train[col] = X_train[col].astype(str).map(lambda x: hash(x) % 10000)
        X_val[col] = X_val[col].astype(str).map(lambda x: hash(x) % 10000)
        X_test[col] = X_test[col].astype(str).map(lambda x: hash(x) % 10000)

print(
    f"\nFinal feature shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
)

# ============================================================
# 6. DEFINE MODEL CONFIGURATIONS
# ============================================================
base_params = {
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "n_estimators": 5000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

params_huber = {**base_params, "objective": "huber", "alpha": 1.0}
params_mse = {**base_params, "objective": "regression"}
params_l1 = {**base_params, "objective": "regression_l1"}


# ============================================================
# 7. TRAINING FUNCTION WITH EARLY STOPPING
# ============================================================
def train_model(params, X_tr, y_tr, X_v, y_v, name="model"):
    """Train a LightGBM model with early stopping on validation RMSE."""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_v, y_v)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=1000),
        ],
    )
    best_iter = model.best_iteration_
    val_pred = model.predict(X_v, num_iteration=best_iter)
    val_rmse = root_mean_squared_error(y_v, val_pred)
    print(f"  {name}: best_iter={best_iter}, val_RMSE={val_rmse:.4f}")
    return model, best_iter, val_rmse


# ============================================================
# 8. TRAIN ALL MODELS AND SELECT BEST
# ============================================================
print("\n" + "=" * 60)
print("Training models...")
print("=" * 60)

results = {}

print("\n[1/3] Training Huber loss model...")
model_huber, iter_huber, rmse_huber = train_model(
    params_huber, X_train, y_train, X_val, y_val, name="Huber"
)
results["huber"] = (model_huber, iter_huber, rmse_huber)

print("\n[2/3] Training MSE regression model...")
model_mse, iter_mse, rmse_mse = train_model(
    params_mse, X_train, y_train, X_val, y_val, name="MSE"
)
results["mse"] = (model_mse, iter_mse, rmse_mse)

print("\n[3/3] Training L1 regression model...")
model_l1, iter_l1, rmse_l1 = train_model(
    params_l1, X_train, y_train, X_val, y_val, name="L1"
)
results["l1"] = (model_l1, iter_l1, rmse_l1)

# ============================================================
# 9. SELECT BEST MODEL BY VALIDATION RMSE
# ============================================================
print("\n" + "=" * 60)
print("Model Selection (by Validation RMSE):")
print("=" * 60)

best_name = min(results, key=lambda k: results[k][2])
best_model, best_iter, best_rmse = results[best_name]

for name, (_, _, rmse) in results.items():
    marker = " ★ BEST" if name == best_name else ""
    print(f"  {name}: RMSE={rmse:.4f}{marker}")

print(f"\nSelected model: {best_name} (best_iter={best_iter})")

# ============================================================
# 10. FINAL VALIDATION & TEST PREDICTIONS (consistent post-processing)
# ============================================================
val_pred_final = best_model.predict(X_val, num_iteration=best_iter)

val_rmse_final = root_mean_squared_error(y_val, val_pred_final)
print(f"\nValidation RMSE: {val_rmse_final:.4f}")

test_pred = best_model.predict(X_test, num_iteration=best_iter)

print(
    f"Test predictions: mean={test_pred.mean():.2f}, min={test_pred.min():.2f}, max={test_pred.max():.2f}"
)
print(
    f"Val predictions: mean={val_pred_final.mean():.2f}, min={val_pred_final.min():.2f}, max={val_pred_final.max():.2f}"
)

# ============================================================
# 11. GENERATE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)

if "id" in test.columns:
    test_ids = test["id"].values
else:
    test_ids = np.arange(len(test))

submission = pd.DataFrame({"id": test_ids, "New Commitments (US$)": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Submission head:\n{submission.head()}")

# ============================================================
# 12. FEATURE IMPORTANCE (top 15)
# ============================================================
importance = pd.Series(
    best_model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nTop 15 Feature Importances:")
for feat, imp in importance.head(15).items():
    print(f"  {feat}: {imp}")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================
score = val_rmse_final
baseline_rmse = root_mean_squared_error(y_val, np.full(len(y_val), y_train.mean()))
print(f"\n--- SUMMARY ---")
print(f"Best model: {best_name}")
print(f"Best iteration: {best_iter}")
print(f"Validation RMSE (official metric): {score:.4f}")
print(f"Baseline (mean) RMSE: {baseline_rmse:.4f}")
print(f"Improvement over baseline: {(1 - score / baseline_rmse) * 100:.1f}%")
print(f"Final Validation Score: {score:.4f}")
