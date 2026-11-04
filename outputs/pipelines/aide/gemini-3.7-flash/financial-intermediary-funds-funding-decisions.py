import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize_scalar
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import warnings

warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Feature engineering & preprocessing
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Parse and extract datetime features
date_cols = []
for col in list(combined.columns):
    if combined[col].dtype == "object":
        try:
            non_null_samples = combined[col].dropna()
            if (
                len(non_null_samples) > 0
                and pd.to_datetime(non_null_samples.iloc[:50], errors="coerce")
                .notnull()
                .mean()
                > 0.8
            ):
                dt_series = pd.to_datetime(combined[col], errors="coerce")
                combined[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
                combined[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
                combined[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
                combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
                    int
                )
                combined[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
                combined[f"{col}_days_epoch"] = (
                    (dt_series - pd.Timestamp("2000-01-01"))
                    .dt.days.fillna(-9999)
                    .astype(float)
                )
                date_cols.append(f"{col}_days_epoch")
                combined.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Pairwise date difference features
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        diff_name = f"diff_{date_cols[i]}_{date_cols[j]}"
        combined[diff_name] = combined[date_cols[i]] - combined[date_cols[j]]

# Categorical handling and Frequency Encoding
cat_cols = []
for col in list(combined.columns):
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        combined[col] = combined[col].fillna("Missing").astype(str)
        # Add frequency encoding
        freq = combined[col].value_counts(normalize=True)
        combined[f"{col}_freq"] = combined[col].map(freq).fillna(0).astype(float)
        cat_cols.append(col)
    else:
        combined[col] = combined[col].fillna(combined[col].median())

# Separate back into train and test
n_train = len(train_df)
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test = combined.iloc[n_train:].reset_index(drop=True)

# For LightGBM: convert categorical columns to pandas category dtype
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

# 5-fold cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_cat = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
test_cat = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

print(f"Features used ({X_train.shape[1]} total, {len(cat_cols)} categoricals)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    # CatBoost data
    X_tr_cat, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va_cat, y_va = X_train.iloc[val_idx], y[val_idx]

    # LightGBM data
    X_tr_lgb = X_train_lgb.iloc[train_idx]
    X_va_lgb = X_train_lgb.iloc[val_idx]

    # 1. Train CatBoost
    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_va_cat, y_va, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED + fold,
        verbose=0,
    )
    cat_model.fit(
        train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = cat_model.predict(val_pool)
    test_cat += cat_model.predict(test_pool) / n_splits

    # 2. Train LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_lgb)
    test_lgb += lgb_model.predict(X_test_lgb) / n_splits

    fold_cat_rmse = np.sqrt(mean_squared_error(y_va, oof_cat[val_idx]))
    fold_lgb_rmse = np.sqrt(mean_squared_error(y_va, oof_lgb[val_idx]))
    print(
        f"Fold {fold + 1} - CatBoost RMSE: {fold_cat_rmse:.4f}, LightGBM RMSE: {fold_lgb_rmse:.4f}"
    )

cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))
lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"Overall CatBoost OOF RMSE: {cat_rmse:.4f}")
print(f"Overall LightGBM OOF RMSE: {lgb_rmse:.4f}")


# Find optimal blend weight
def blend_obj(w):
    pred = w * oof_cat + (1.0 - w) * oof_lgb
    return np.sqrt(mean_squared_error(y, pred))


res = minimize_scalar(blend_obj, bounds=(0.0, 1.0), method="bounded")
best_w = res.x
oof_blend = best_w * oof_cat + (1.0 - best_w) * oof_lgb
blend_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"Optimal CatBoost Weight: {best_w:.4f}")
print(f"Validation Metric (Ensemble OOF RMSE): {blend_rmse:.4f}")

# Final test predictions
final_test_preds = best_w * test_cat + (1.0 - best_w) * test_lgb

# Prepare and save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Amount in USD": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
