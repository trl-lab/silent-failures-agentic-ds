import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_path = "./working/submission.csv"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "capacity_mw"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col], errors="ignore")
X_test_raw = test_df.copy()

# 2. Feature Engineering
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Drop high cardinality identifier columns
drop_id_cols = [
    col for col in ["name", "gppd_idnr", "url", "wepp_id"] if col in combined.columns
]
combined.drop(columns=drop_id_cols, inplace=True)

# Identify numeric generation columns
gen_cols = []
for c in combined.columns:
    if "generation" in c.lower() and "source" not in c.lower():
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
        gen_cols.append(c)

# Sort generation columns chronologically if possible
gen_year_cols = sorted([c for c in gen_cols if any(char.isdigit() for char in c)])

if gen_cols:
    combined["gen_mean"] = combined[gen_cols].mean(axis=1)
    combined["gen_median"] = combined[gen_cols].median(axis=1)
    combined["gen_std"] = combined[gen_cols].std(axis=1).fillna(0)
    combined["gen_max"] = combined[gen_cols].max(axis=1)
    combined["gen_min"] = combined[gen_cols].min(axis=1)
    combined["gen_sum"] = combined[gen_cols].sum(axis=1)
    combined["gen_non_null_count"] = combined[gen_cols].notnull().sum(axis=1)
    combined["gen_range"] = combined["gen_max"] - combined["gen_min"]
    combined["gen_cv"] = combined["gen_std"] / (combined["gen_mean"] + 1e-4)

# Year-over-year generation differences and trends
if len(gen_year_cols) >= 2:
    for i in range(len(gen_year_cols) - 1):
        col1 = gen_year_cols[i]
        col2 = gen_year_cols[i + 1]
        combined[f"diff_{col2}_{col1}"] = combined[col2] - combined[col1]
    combined["gen_trend_latest"] = (
        combined[gen_year_cols[-1]] - combined[gen_year_cols[0]]
    )

# Geospatial features
if "latitude" in combined.columns and "longitude" in combined.columns:
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
    lat_rad = np.radians(combined["latitude"].fillna(0))
    lon_rad = np.radians(combined["longitude"].fillna(0))
    combined["x_cart"] = np.cos(lat_rad) * np.cos(lon_rad)
    combined["y_cart"] = np.cos(lat_rad) * np.sin(lon_rad)
    combined["z_cart"] = np.sin(lat_rad)

# Commissioning year features
comm_cols = [c for c in combined.columns if "commissioning_year" in c.lower()]
for c in comm_cols:
    combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined[f"{c}_age"] = 2025 - combined[c]

# Fuel and Country group aggregations
fuel_col = None
for c in ["primary_fuel", "fuel1"]:
    if c in combined.columns:
        fuel_col = c
        break

country_col = None
for c in ["country", "country_long"]:
    if c in combined.columns:
        country_col = c
        break

if fuel_col and "gen_mean" in combined.columns:
    fuel_stats = (
        combined.groupby(fuel_col)["gen_mean"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    fuel_stats.columns = [fuel_col, "fuel_gen_mean", "fuel_gen_std", "fuel_gen_median"]
    combined = combined.merge(fuel_stats, on=fuel_col, how="left")
    combined["ratio_gen_to_fuel_mean"] = combined["gen_mean"] / (
        combined["fuel_gen_mean"] + 1e-4
    )

if fuel_col and country_col and "gen_mean" in combined.columns:
    fc_stats = (
        combined.groupby([fuel_col, country_col])["gen_mean"]
        .agg(["mean", "median"])
        .reset_index()
    )
    fc_stats.columns = [
        fuel_col,
        country_col,
        "fuel_country_gen_mean",
        "fuel_country_gen_median",
    ]
    combined = combined.merge(fc_stats, on=[fuel_col, country_col], how="left")
    combined["ratio_gen_to_fc_mean"] = combined["gen_mean"] / (
        combined["fuel_country_gen_mean"] + 1e-4
    )

# Missing values count per row
combined["missing_count"] = combined.isnull().sum(axis=1)

# Categorical encodings
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    freq = combined[c].value_counts(dropna=False)
    combined[f"{c}_freq"] = combined[c].map(freq)
    combined[c] = combined[c].astype("category")

# Split back to train and test
n_train = len(train_df)
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 3. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(n_train)
test_predictions = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = model.predict(X_va, num_iteration=model.best_iteration)
    oof_predictions[val_idx] = np.clip(val_preds, 0, None)

    fold_test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    test_predictions += np.clip(fold_test_preds, 0, None) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_predictions)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        "capacity_mw": test_predictions,
    }
)
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path} with shape {submission.shape}")
