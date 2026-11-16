import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# 1. Load Data
print("Loading data...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Target encoding
target_col = "Category"
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col])
num_classes = len(le_target.classes_)


# 3. Feature Engineering Function
def preprocess_features(df, is_train=True, district_medians=None):
    df = df.copy()

    # Parse dates
    df["Dates"] = pd.to_datetime(df["Dates"])
    df["Year"] = df["Dates"].dt.year
    df["Month"] = df["Dates"].dt.month
    df["Day"] = df["Dates"].dt.day
    df["Hour"] = df["Dates"].dt.hour
    df["Minute"] = df["Dates"].dt.minute
    df["DayOfWeek_num"] = df["Dates"].dt.dayofweek
    df["DayOfYear"] = df["Dates"].dt.dayofyear
    df["IsWeekend"] = (df["DayOfWeek_num"] >= 5).astype(int)

    # Cyclical time encodings
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek_num"] / 7.0)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek_num"] / 7.0)

    # Spatial outlier cleaning: replace Y==90 or X > -121 with district median
    if district_medians is None:
        valid_coords = df[(df["Y"] < 80) & (df["X"] < -121)]
        district_medians = (
            valid_coords.groupby("PdDistrict")[["X", "Y"]].median().to_dict()
        )

    for dist in df["PdDistrict"].unique():
        if dist in district_medians["X"]:
            mask = (df["PdDistrict"] == dist) & ((df["Y"] >= 80) | (df["X"] >= -121))
            df.loc[mask, "X"] = district_medians["X"][dist]
            df.loc[mask, "Y"] = district_medians["Y"][dist]

    # Spatial features
    sf_lat, sf_lon = 37.7749, -122.4194
    df["Dist_Center"] = np.sqrt((df["Y"] - sf_lat) ** 2 + (df["X"] - sf_lon) ** 2)
    df["Rot45_X"] = 0.7071 * df["X"] + 0.7071 * df["Y"]
    df["Rot45_Y"] = -0.7071 * df["X"] + 0.7071 * df["Y"]

    # Address features
    df["Is_Intersection"] = df["Address"].str.contains("/").astype(int)
    df["Is_Block"] = df["Address"].str.contains("Block", case=False).astype(int)

    # Categoricals
    df["PdDistrict"] = df["PdDistrict"].astype("category")
    df["DayOfWeek"] = df["DayOfWeek"].astype("category")

    feature_cols = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "DayOfWeek_num",
        "DayOfYear",
        "IsWeekend",
        "Hour_sin",
        "Hour_cos",
        "Month_sin",
        "Month_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "X",
        "Y",
        "Dist_Center",
        "Rot45_X",
        "Rot45_Y",
        "Is_Intersection",
        "Is_Block",
        "PdDistrict",
        "DayOfWeek",
    ]

    return df[feature_cols], district_medians


print("Engineering features...")
X_train, dist_medians = preprocess_features(train_df, is_train=True)
X_test, _ = preprocess_features(test_df, is_train=False, district_medians=dist_medians)

# 4. Stratified 5-Fold Cross Validation with LightGBM
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), num_classes), dtype=np.float32)
test_preds = np.zeros((len(test_df), num_classes), dtype=np.float32)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.15,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

print(f"Starting {n_splits}-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=300,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    fold_loss = log_loss(y_val, val_pred)
    print(f"Fold {fold + 1} Log Loss: {fold_loss:.4f}")

    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 5. Overall Validation Performance
cv_log_loss = log_loss(y, oof_preds)
print(f"5-Fold CV Log Loss: {cv_log_loss:.5f}")

# 6. Generate Submission
os.makedirs("./working", exist_ok=True)
predicted_class_indices = np.argmax(test_preds, axis=1)
predicted_categories = le_target.inverse_transform(predicted_class_indices)

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Category": predicted_categories}
)

submission.to_csv("./working/submission.csv", index=False)
print(
    "Submission saved successfully to ./working/submission.csv with shape:",
    submission.shape,
)
