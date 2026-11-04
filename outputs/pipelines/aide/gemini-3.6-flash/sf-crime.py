import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Save target and fit label encoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["Category"])
num_classes = len(label_encoder.classes_)

# Combine train and test for uniform feature engineering
n_train = len(train_df)
combined = pd.concat(
    [
        train_df.drop(columns=["Category", "Descript", "Resolution"], errors="ignore"),
        test_df.drop(columns=["Category", "Descript", "Resolution"], errors="ignore"),
    ],
    ignore_index=True,
)

# Clean spatial outliers
outlier_mask = (combined["X"] > -121) | (combined["Y"] > 50)
combined.loc[outlier_mask, "X"] = np.nan
combined.loc[outlier_mask, "Y"] = np.nan

# Fill spatial NaNs using PdDistrict medians
combined["X"] = combined.groupby("PdDistrict")["X"].transform(
    lambda g: g.fillna(g.median())
)
combined["Y"] = combined.groupby("PdDistrict")["Y"].transform(
    lambda g: g.fillna(g.median())
)

# Datetime features
dates = pd.to_datetime(combined["Dates"])
combined["Year"] = dates.dt.year
combined["Month"] = dates.dt.month
combined["Day"] = dates.dt.day
combined["Hour"] = dates.dt.hour
combined["Minute"] = dates.dt.minute
combined["DayOfWeek_num"] = dates.dt.dayofweek
combined["DayOfYear"] = dates.dt.dayofyear
combined["IsWeekend"] = (combined["DayOfWeek_num"] >= 5).astype(int)
combined["TimeInHours"] = combined["Hour"] + combined["Minute"] / 60.0

# Spatial features
rad = np.radians(45)
combined["X_rot45"] = combined["X"] * np.cos(rad) - combined["Y"] * np.sin(rad)
combined["Y_rot45"] = combined["X"] * np.sin(rad) + combined["Y"] * np.cos(rad)
combined["Dist_Center"] = np.sqrt(
    (combined["X"] - (-122.4194)) ** 2 + (combined["Y"] - 37.7749) ** 2
)

# Address features
combined["Is_Intersection"] = combined["Address"].str.contains("/").astype(int)
combined["Address_Clean"] = combined["Address"].apply(
    lambda x: x.split(" / ")[0].split(" Block of ")[-1] if isinstance(x, str) else ""
)

# Frequency encodings
for col in ["Address", "Address_Clean", "PdDistrict"]:
    freq = combined[col].value_counts()
    combined[f"{col}_freq"] = combined[col].map(freq)

# Convert categorical columns
cat_cols = ["PdDistrict", "DayOfWeek", "Address_Clean"]
for col in cat_cols:
    combined[col] = combined[col].astype("category")

# Drop unused columns
drop_cols = ["Dates", "Address", "Id", "id"]
features = [c for c in combined.columns if c not in drop_cols]

X = combined.iloc[:n_train][features].copy()
X_test = combined.iloc[n_train:][features].copy()

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((n_train, num_classes), dtype=np.float32)
test_probs = np.zeros((len(X_test), num_classes), dtype=np.float32)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_probs[val_idx] = val_pred
    test_probs += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

# Compute OOF Metric
cv_log_loss = log_loss(y, oof_probs)
print(f"5-Fold CV Log Loss: {cv_log_loss:.5f}")

# Prepare submission file
test_pred_labels = label_encoder.inverse_transform(np.argmax(test_probs, axis=1))
test_id = (
    test_df["Id"].values
    if "Id" in test_df.columns
    else (test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df)))
)

submission = pd.DataFrame({"id": test_id, "Category": test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
