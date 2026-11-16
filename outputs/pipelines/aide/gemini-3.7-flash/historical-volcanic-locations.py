import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "DEATHS_TOTAL_DESCRIPTION"

# Drop rows with missing target in train if any
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Encode target
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col].astype(str))
num_classes = len(target_encoder.classes_)

# 2. Preprocess and Feature Engineering
combined = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], axis=0
).reset_index(drop=True)

# Identify feature types
drop_cols = []
for c in combined.columns:
    if combined[c].nunique() <= 1:
        drop_cols.append(c)

combined = combined.drop(columns=drop_cols)

# Separate categorical and numerical columns
cat_cols = []
num_cols = []

for c in combined.columns:
    if combined[c].dtype == "object" or combined[c].dtype.name == "category":
        cat_cols.append(c)
    else:
        num_cols.append(c)

# Encode categorical features as category dtype for LightGBM
for c in cat_cols:
    combined[c] = combined[c].astype(str).fillna("MISSING")
    le = LabelEncoder()
    combined[c] = le.fit_transform(combined[c])
    combined[c] = combined[c].astype("category")

# Feature engineering: Add coordinate transformations if Latitude and Longitude exist
lat_col = next((c for c in combined.columns if "lat" in c.lower()), None)
lon_col = next(
    (c for c in combined.columns if "lon" in c.lower() or "lng" in c.lower()), None
)

if lat_col and lon_col:
    combined[f"{lat_col}_rad"] = np.radians(combined[lat_col])
    combined[f"{lon_col}_rad"] = np.radians(combined[lon_col])
    combined["coord_x"] = np.cos(combined[f"{lat_col}_rad"]) * np.cos(
        combined[f"{lon_col}_rad"]
    )
    combined["coord_y"] = np.cos(combined[f"{lat_col}_rad"]) * np.sin(
        combined[f"{lon_col}_rad"]
    )
    combined["coord_z"] = np.sin(combined[f"{lat_col}_rad"])

# Count missing values per row as an informative feature
combined["missing_count"] = combined.isnull().sum(axis=1)

# Split back to train and test
n_train = len(train_df)
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 3. Stratified 5-Fold Cross Validation with LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

# 4. Evaluation
oof_class_preds = np.argmax(oof_preds, axis=1)
overall_acc = accuracy_score(y_train, oof_class_preds)
print(f"5-Fold CV Out-Of-Fold Accuracy: {overall_acc:.5f}")

# 5. Generate Submission
test_class_preds_encoded = np.argmax(test_preds, axis=1)
test_class_preds = target_encoder.inverse_transform(test_class_preds_encoded)

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "DEATHS_TOTAL_DESCRIPTION": test_class_preds}
)

sub_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path} with shape {submission.shape}")
