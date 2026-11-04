import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Amount in USD"
train = train.dropna(subset=[target_col]).reset_index(drop=True)

y_train = train[target_col].astype(float)
X_train = train.drop(columns=[target_col]).copy()
X_test = test.copy().reindex(columns=X_train.columns)

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    if X_train[col].dtype.name == "category":
        train_categories = X_train[col].cat.categories
    else:
        train_categories = pd.unique(X_train[col].dropna())

    X_train[col] = pd.Categorical(X_train[col], categories=train_categories)
    X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

max_boost_rounds = 5000
early_stopping_rounds = 100
seeds = [42, 123, 456, 789, 1011]


def make_seed_params(seed: int) -> dict:
    p = params.copy()
    p["seed"] = seed
    p["bagging_seed"] = seed
    p["feature_fraction_seed"] = seed
    p["data_random_seed"] = seed
    return p


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(y_train), dtype=float)
best_iterations_by_seed = {seed: [] for seed in seeds}
categorical_feature = cat_cols if cat_cols else None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), start=1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    val_pred = np.zeros(len(val_idx), dtype=float)

    for seed in seeds:
        train_data = lgb.Dataset(
            X_tr,
            y_tr,
            categorical_feature=categorical_feature,
        )
        val_data = lgb.Dataset(
            X_val,
            y_val,
            reference=train_data,
            categorical_feature=categorical_feature,
        )

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
            ),
            lgb.log_evaluation(-1),
        ]

        booster = lgb.train(
            make_seed_params(seed),
            train_data,
            num_boost_round=max_boost_rounds,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = booster.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = max_boost_rounds

        best_iterations_by_seed[seed].append(best_iter)
        val_pred += booster.predict(X_val, num_iteration=best_iter)

    oof_predictions[val_idx] = val_pred / len(seeds)
    fold_rmse = float(np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx])))
    print(f"Fold {fold} seed-averaged validation RMSE: {fold_rmse:.4f}")

cv_rmse = float(np.sqrt(mean_squared_error(y_train, oof_predictions)))
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

final_rounds_by_seed = {}
for seed in seeds:
    iters = best_iterations_by_seed[seed]
    final_rounds_by_seed[seed] = int(np.clip(np.median(iters), 1, max_boost_rounds))
    print(f"Seed {seed} final boosting rounds: {final_rounds_by_seed[seed]}")

pred_test = np.zeros(len(X_test), dtype=float)

for seed in seeds:
    full_data = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=categorical_feature,
    )
    final_booster = lgb.train(
        make_seed_params(seed),
        full_data,
        num_boost_round=final_rounds_by_seed[seed],
    )
    pred_test += final_booster.predict(X_test)

pred_test /= len(seeds)

submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        "Amount in USD": pred_test,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
