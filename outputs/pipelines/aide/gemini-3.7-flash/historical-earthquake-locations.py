import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Damage Description (total)"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Encode Target
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
num_classes = len(le.classes_)

# 2. Feature Engineering & Preprocessing
n_train = len(train_df)
combined = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0
).reset_index(drop=True)

# Row-level missingness indicator
combined["missing_feature_count"] = combined.isnull().sum(axis=1)

# Datetime / Temporal Feature Extraction
for col in list(combined.columns):
    col_lower = str(col).lower()
    if (
        "date" in col_lower
        or "time" in col_lower
        or combined[col].dtype == "datetime64[ns]"
    ):
        try:
            dt_series = pd.to_datetime(combined[col], errors="coerce")
            combined[f"{col}_year"] = dt_series.dt.year
            combined[f"{col}_month"] = dt_series.dt.month
            combined[f"{col}_day"] = dt_series.dt.day
            combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek
            combined = combined.drop(columns=[col])
        except Exception:
            pass

# Domain-specific Earthquake Features
# Identify lat / lon
lat_col = next((c for c in combined.columns if c.lower() in ["latitude", "lat"]), None)
lon_col = next(
    (c for c in combined.columns if c.lower() in ["longitude", "lon", "long"]), None
)

if lat_col and lon_col:
    lat_rad = np.radians(pd.to_numeric(combined[lat_col], errors="coerce"))
    lon_rad = np.radians(pd.to_numeric(combined[lon_col], errors="coerce"))
    combined["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    combined["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    combined["coord_z"] = np.sin(lat_rad)

# Identify magnitude and depth columns
mag_col = next(
    (
        c
        for c in combined.columns
        if any(k in c.lower() for k in ["magnitude", "primary_mag", "eq_mag", "mag"])
    ),
    None,
)
depth_col = next(
    (
        c
        for c in combined.columns
        if any(k in c.lower() for k in ["focal_depth", "depth", "focal depth"])
    ),
    None,
)
intensity_col = next(
    (
        c
        for c in combined.columns
        if any(k in c.lower() for k in ["intensity", "mmi", "mercalli"])
    ),
    None,
)

if mag_col:
    mag_vals = pd.to_numeric(combined[mag_col], errors="coerce")
    combined["mag_energy_proxy"] = 1.5 * mag_vals

    if depth_col:
        depth_vals = pd.to_numeric(combined[depth_col], errors="coerce").clip(lower=0)
        combined["shallow_mag_ratio"] = mag_vals / np.log1p(depth_vals + 1.0)
        combined["mag_depth_prod"] = mag_vals * depth_vals

    if intensity_col:
        intensity_vals = pd.to_numeric(combined[intensity_col], errors="coerce")
        combined["intensity_mag_diff"] = intensity_vals - mag_vals

# Frequency Encoding and Categorical Type Conversion
cat_cols = []
for col in combined.columns:
    if (
        combined[col].dtype == "object"
        or combined[col].dtype.name == "category"
        or not np.issubdtype(combined[col].dtype, np.number)
    ):
        # Frequency encoding
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_freq"] = combined[col].map(freq)
        combined[col] = combined[col].astype(str).astype("category")
        cat_cols.append(col)

X_train = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)

# 3. Multi-Seed Stratified K-Fold Training
n_splits = 5
seeds = [42, 2024]
oof_preds = np.zeros((len(X_train), num_classes))
test_preds = np.zeros((len(X_test), num_classes))

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "feature_fraction": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_state": seed,
        "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr_f, y_tr_f = X_train.iloc[train_idx], y[train_idx]
        X_va_f, y_va_f = X_train.iloc[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_tr_f, label=y_tr_f, categorical_feature=cat_cols)
        val_data = lgb.Dataset(
            X_va_f,
            label=y_va_f,
            reference=trn_data,
            categorical_feature=cat_cols,
        )

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        val_pred = model.predict(X_va_f, num_iteration=model.best_iteration)
        oof_preds[val_idx] += val_pred / len(seeds)

        test_pred_fold = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += test_pred_fold / (n_splits * len(seeds))

# 4. Evaluation and Submission
oof_labels = np.argmax(oof_preds, axis=1)
overall_acc = accuracy_score(y, oof_labels)
print(f"5-Fold CV Accuracy: {overall_acc:.5f}")

final_test_labels = le.inverse_transform(np.argmax(test_preds, axis=1))

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), target_col: final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
