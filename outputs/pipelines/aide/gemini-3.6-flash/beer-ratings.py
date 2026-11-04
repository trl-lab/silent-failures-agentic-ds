import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "review_overall"
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Feature Engineering
# Handle datetime columns
date_cols = [
    c for c in combined_df.columns if "time" in c.lower() or "date" in c.lower()
]
for col in date_cols:
    if combined_df[col].dtype == "object" or np.issubdtype(
        combined_df[col].dtype, np.number
    ):
        try:
            combined_df[col] = pd.to_datetime(
                combined_df[col],
                unit="s" if np.issubdtype(combined_df[col].dtype, np.number) else None,
            )
        except Exception:
            pass
    if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        combined_df[f"{col}_year"] = combined_df[col].dt.year
        combined_df[f"{col}_month"] = combined_df[col].dt.month
        combined_df[f"{col}_day"] = combined_df[col].dt.day
        combined_df[f"{col}_dayofweek"] = combined_df[col].dt.dayofweek
        combined_df[f"{col}_hour"] = combined_df[col].dt.hour
        combined_df = combined_df.drop(columns=[col])

# Sub-score columns
review_score_cols = [
    c
    for c in combined_df.columns
    if "review" in c.lower()
    and c != target_col
    and combined_df[c].dtype in [float, int, "float64", "int64"]
]

if len(review_score_cols) > 1:
    combined_df["review_scores_mean"] = combined_df[review_score_cols].mean(axis=1)
    combined_df["review_scores_std"] = combined_df[review_score_cols].std(axis=1)
    combined_df["review_scores_min"] = combined_df[review_score_cols].min(axis=1)
    combined_df["review_scores_max"] = combined_df[review_score_cols].max(axis=1)

    # Pairwise sub-score differences
    for i in range(len(review_score_cols)):
        for j in range(i + 1, len(review_score_cols)):
            col1, col2 = review_score_cols[i], review_score_cols[j]
            combined_df[f"{col1}_minus_{col2}"] = combined_df[col1] - combined_df[col2]

# Categorical Frequency Encoding & Conversion
cat_cols = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    freq = combined_df[col].value_counts().to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq)
    combined_df[col] = combined_df[col].astype("category")

# Separate sub-score features for Ridge regression
subscore_data = combined_df[review_score_cols].fillna(0).values
poly = PolynomialFeatures(degree=2, include_bias=False)
X_subscores_poly = poly.fit_transform(subscore_data)
scaler = StandardScaler()
X_subscores_poly = scaler.fit_transform(X_subscores_poly)

X_sub_train = X_subscores_poly[:n_train]
X_sub_test = X_subscores_poly[n_train:]

# Separate main feature sets
X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# 3. Model Training (LightGBM + Ridge Blend)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))

oof_ridge = np.zeros(n_train)
test_ridge = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # Fit LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits

    # Fit Ridge Regression on Polynomial Sub-score features
    X_sub_tr, y_sub_tr = X_sub_train[train_idx], y_train[train_idx]
    X_sub_va = X_sub_train[val_idx]

    ridge_model = Ridge(alpha=100.0)
    ridge_model.fit(X_sub_tr, y_sub_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_sub_va)
    test_ridge += ridge_model.predict(X_sub_test) / kf.n_splits


# 4. Optimize Ensemble Blending Weight
def blend_loss(w):
    pred = w * oof_lgb + (1 - w) * oof_ridge
    return np.sqrt(mean_squared_error(y_train, pred))


res = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
best_w = res.x

oof_blended = best_w * oof_lgb + (1 - best_w) * oof_ridge
final_test_preds = best_w * test_lgb + (1 - best_w) * test_ridge

oof_rmse_lgb = np.sqrt(mean_squared_error(y_train, oof_lgb))
oof_rmse_ridge = np.sqrt(mean_squared_error(y_train, oof_ridge))
oof_rmse_blend = np.sqrt(mean_squared_error(y_train, oof_blended))

print(f"LGB OOF RMSE: {oof_rmse_lgb:.5f}")
print(f"Ridge OOF RMSE: {oof_rmse_ridge:.5f}")
print(f"Optimal LGB Weight: {best_w:.4f}")
print(f"5-Fold CV Blended OOF RMSE: {oof_rmse_blend:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "review_overall": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
