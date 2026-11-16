import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "age",
    "avno60plus",
    "structure_quality",
    "month_sold",
    "LATITUDE",
    "LONGITUDE",
]

X_train = train_df[feature_cols].copy()
y_train = np.log1p(train_df["SALE_PRC"])

X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_loss = np.inf
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_cols, reference=train_data
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(50),
        lgb.log_evaluation(-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_pred)

    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        best_model = model

# Final evaluation on validation set
if best_model is not None:
    val_preds = best_model.predict(X_train.iloc[val_idx])
    final_rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], val_preds))
    print(f"Best validation RMSE (log scale): {final_rmse:.4f}")

# Generate predictions for test set
test_preds = best_model.predict(X_test)
test_preds_log = np.log1p(test_preds)

# Save submission
submission = pd.DataFrame({"id": range(len(test_preds)), "SALE_PRC": test_preds_log})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [
    "LND_SQFOOT",
    "TOT_LVG_AREA",
    "SPEC_FEAT_VAL",
    "RAIL_DIST",
    "OCEAN_DIST",
    "WATER_DIST",
    "CNTR_DIST",
    "SUBCNTR_DI",
    "HWY_DIST",
    "age",
    "avno60plus",
    "structure_quality",
    "month_sold",
    "LATITUDE",
    "LONGITUDE",
]

X_train = train_df[feature_cols].copy()
y_train = np.log1p(train_df["SALE_PRC"])

X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_loss = np.inf
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_cols, reference=train_data
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    callbacks = [
        lgb.early_stopping(50),
        lgb.log_evaluation(-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_val, val_pred)

    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        best_model = model

# Final evaluation on validation set
val_preds = best_model.predict(X_train.iloc[val_idx])
final_rmse = np.sqrt(mean_squared_error(y_train.iloc[val_idx], val_preds))
print(f"Best validation RMSE (log scale): {final_rmse:.4f}")

# Generate predictions for test set
test_preds = best_model.predict(X_test)
test_preds_log = np.log1p(test_preds)

# Save submission
submission = pd.DataFrame({"id": range(len(test_preds)), "SALE_PRC": test_preds_log})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
