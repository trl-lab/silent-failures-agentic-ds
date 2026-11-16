import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SupStruc"
y = train_df[target_col].astype(float).values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Concatenate for consistent feature engineering
combined = pd.concat([X, X_test], axis=0, ignore_index=True)

# Process datetime columns
for col in list(combined.columns):
    if (
        combined[col].dtype == "object"
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            dt = pd.to_datetime(combined[col], errors="coerce")
            if dt.notna().sum() > 0.5 * len(dt):
                combined[f"{col}_year"] = dt.dt.year.fillna(-1)
                combined[f"{col}_month"] = dt.dt.month.fillna(-1)
                combined[f"{col}_day"] = dt.dt.day.fillna(-1)
                combined[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1)
                combined.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Separate categorical and numerical columns
cat_cols = []
num_cols = []

for col in combined.columns:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        cat_cols.append(col)
        combined[col] = combined[col].fillna("missing").astype(str)
    else:
        num_cols.append(col)
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

# Spatial coordinate feature engineering
lat_cols = [c for c in num_cols if "lat" in c.lower()]
lon_cols = [c for c in num_cols if "lon" in c.lower() or "lng" in c.lower()]

if lat_cols and lon_cols:
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    lat_mean = combined[lat_col].mean()
    lon_mean = combined[lon_col].mean()
    combined["dist_center"] = np.sqrt(
        (combined[lat_col] - lat_mean) ** 2 + (combined[lon_col] - lon_mean) ** 2
    )
    combined["coord_rot_45"] = combined[lat_col] * 0.7071 + combined[lon_col] * 0.7071
    combined["coord_rot_135"] = -combined[lat_col] * 0.7071 + combined[lon_col] * 0.7071
    num_cols.extend(["dist_center", "coord_rot_45", "coord_rot_135"])

# Missing value handling for numeric columns
for col in num_cols:
    med = combined[col].median()
    if pd.isna(med):
        med = 0
    combined[col] = combined[col].fillna(med)

# Split train and test back
X_proc = combined.iloc[: len(train_df)].copy().reset_index(drop=True)
X_test_proc = combined.iloc[len(train_df) :].copy().reset_index(drop=True)

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc, y)):
    X_tr, y_tr = X_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_proc.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        cat_features=cat_cols if cat_cols else None,
        verbose=0,
    )

    model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test_proc) / kf.n_splits

oof_rmse = root_mean_squared_error(y, oof_preds)
print(f"CatBoost 5-Fold OOF RMSE: {oof_rmse:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "SupStruc": test_preds})
submission.to_csv("./working/submission.csv", index=False)
