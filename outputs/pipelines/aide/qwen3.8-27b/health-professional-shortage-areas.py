import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "HPSA Score"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Ensure target is numeric and usable.
if TARGET not in train.columns:
    train[TARGET] = 0.0
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET])
train = train[np.isfinite(train[TARGET])]

# Identify feature columns present in both train and test.
exclude = {TARGET, "id"}
features = [c for c in train.columns if c not in exclude and c in test.columns]

# Basic type cleanup: datetime -> numeric, object -> numeric if possible, otherwise category.
for col in features:
    if pd.api.types.is_datetime64_any_dtype(
        train[col]
    ) or pd.api.types.is_datetime64_any_dtype(test[col]):
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

for col in features:
    train_is_obj = train[col].dtype == object
    test_is_obj = test[col].dtype == object
    train_is_cat = pd.api.types.is_categorical_dtype(train[col])
    test_is_cat = pd.api.types.is_categorical_dtype(test[col])

    if train_is_obj or test_is_obj:
        train_num = pd.to_numeric(train[col], errors="coerce")
        test_num = pd.to_numeric(test[col], errors="coerce")

        # If every non-missing value is numeric, keep it numeric.
        if (
            train_num.notna().sum() == train[col].notna().sum()
            and test_num.notna().sum() == test[col].notna().sum()
        ):
            train[col] = train_num
            test[col] = test_num
        else:
            train_cat = train[col].astype("category")
            test_cat = test[col].astype("category")
            cats = set(train_cat.cat.categories) | set(test_cat.cat.categories)
            train[col] = train_cat.cat.set_categories(cats)
            test[col] = test_cat.cat.set_categories(cats)
    elif train_is_cat or test_is_cat:
        train_cat = train[col].astype("category")
        test_cat = test[col].astype("category")
        cats = set(train_cat.cat.categories) | set(test_cat.cat.categories)
        train[col] = train_cat.cat.set_categories(cats)
        test[col] = test_cat.cat.set_categories(cats)

# Remove constant or all-missing features.
clean_features = []
for col in features:
    if train[col].nunique(dropna=True) > 1:
        clean_features.append(col)
features = clean_features

# Fallback if no usable features remain.
if len(features) == 0:
    val_size = min(0.2 * len(train), len(train) - 1) if len(train) > 1 else 0
    if val_size > 0:
        train_part, val_part = train_test_split = (
            None  # placeholder to avoid unused import confusion
        )
    # Simpler deterministic fallback:
    mean_target = float(train[TARGET].mean()) if len(train) > 0 else 0.0
    if len(train) >= 2:
        val_idx = np.random.RandomState(42).choice(
            len(train), size=max(1, int(0.2 * len(train))), replace=False
        )
        val_pred = np.full(len(val_idx), mean_target)
        val_true = train.iloc[val_idx][TARGET].to_numpy()
        rmse = root_mean_squared_error(val_true, val_pred)
    else:
        rmse = 0.0

    test_pred = np.full(len(test), mean_target)
    print(f"Validation RMSE: {rmse:.6f}")

    submission = pd.DataFrame(
        {"id": np.arange(len(test), dtype=int), TARGET: test_pred}
    )
    submission.to_csv("./working/submission.csv", index=False)
    raise SystemExit(0)

# LightGBM parameters compatible with lightgbm 4.6.0.
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

num_boost_round = 3000
early_stopping_rounds = 50
n_splits = min(5, len(train)) if len(train) >= 2 else 1

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

test_X = test[features].copy()
target_min = float(train[TARGET].min())
target_max = float(train[TARGET].max())

for fold, (train_idx, val_idx) in enumerate(kf.split(train), start=1):
    train_X = train.iloc[train_idx][features].copy()
    train_y = train.iloc[train_idx][TARGET].to_numpy(dtype=float)
    val_X = train.iloc[val_idx][features].copy()
    val_y = train.iloc[val_idx][TARGET].to_numpy(dtype=float)

    categorical_feature = [
        col
        for col in features
        if pd.api.types.is_categorical_dtype(train_X[col])
        and train_X[col].nunique(dropna=True) > 0
    ]

    train_set = lgb.Dataset(
        train_X,
        label=train_y,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val_X,
        label=val_y,
        reference=train_set,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = (
        model.best_iteration
        if model.best_iteration and model.best_iteration > 0
        else num_boost_round
    )

    oof_pred[val_idx] = model.predict(val_X, num_iteration=best_iter)
    test_pred += model.predict(test_X, num_iteration=best_iter) / n_splits

    if fold % 5 == 0 or fold == n_splits:
        print(f"Fold {fold}/{n_splits} completed")

# Clip predictions to the observed target range for numerical stability.
oof_pred = np.clip(oof_pred, target_min, target_max)
test_pred = np.clip(test_pred, target_min, target_max)

validation_rmse = root_mean_squared_error(train[TARGET].to_numpy(dtype=float), oof_pred)
print(f"Validation RMSE: {validation_rmse:.6f}")

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
