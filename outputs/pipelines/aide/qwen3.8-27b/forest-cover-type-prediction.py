import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

os.makedirs("./working", exist_ok=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    dist_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]

    for col in dist_cols:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].abs())

    if all(
        c in out.columns
        for c in [
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ):
        h_hyd = out["Horizontal_Distance_To_Hydrology"]
        v_hyd = out["Vertical_Distance_To_Hydrology"]
        h_road = out["Horizontal_Distance_To_Roadways"]
        h_fire = out["Horizontal_Distance_To_Fire_Points"]

        out["Euclidean_Distance_To_Hydrology"] = np.hypot(h_hyd, v_hyd)
        out["log_Euclidean_Distance_To_Hydrology"] = np.log1p(
            out["Euclidean_Distance_To_Hydrology"]
        )

        out["Hyd_to_Road"] = h_hyd / (h_road + 1e-6)
        out["Hyd_to_Fire"] = h_hyd / (h_fire + 1e-6)
        out["Road_to_Fire"] = h_road / (h_fire + 1e-6)
        out["V_Hyd_to_H_Hyd"] = v_hyd / (h_hyd + 1e-6)

    if all(
        c in out.columns for c in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    ):
        h9 = out["Hillshade_9am"]
        h12 = out["Hillshade_Noon"]
        h15 = out["Hillshade_3pm"]

        out["Hillshade_Noon_minus_9am"] = h12 - h9
        out["Hillshade_3pm_minus_Noon"] = h15 - h12
        out["Hillshade_9am_minus_3pm"] = h9 - h15

        hillshade = out[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]]
        out["Hillshade_range"] = hillshade.max(axis=1) - hillshade.min(axis=1)

    return out


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = add_features(train)
test = add_features(test)

target = "Cover_Type"
id_col = "Id"
exclude_cols = {target, id_col}

feature_cols = [c for c in train.columns if c not in exclude_cols]

X = train[feature_cols].copy()
y = train[target].astype(int)
X_test = test.reindex(columns=feature_cols).copy()

X = X.fillna(0).astype(float)
X_test = X_test.fillna(0).astype(float)

constant_cols = X.loc[:, X.nunique() <= 1].columns.tolist()
if constant_cols:
    feature_cols = [c for c in feature_cols if c not in constant_cols]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

classes = np.sort(y.unique())
y_enc = np.searchsorted(classes, y.values).astype(int)
num_class = len(classes)

params = {
    "objective": "multiclass",
    "num_class": num_class,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
    "verbose": -1,
    "num_threads": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
best_iters = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), start=1):
    dtrain = lgb.Dataset(X.iloc[train_idx], label=y_enc[train_idx], free_raw_data=False)
    dvalid = lgb.Dataset(X.iloc[val_idx], label=y_enc[val_idx], free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iter = model.best_iteration if model.best_iteration is not None else 1
    best_iters.append(best_iter)

    val_pred_idx = np.argmax(model.predict(X.iloc[val_idx]), axis=1)
    val_pred = classes[val_pred_idx]
    acc = accuracy_score(y.iloc[val_idx], val_pred)

    fold_accs.append(acc)
    print(f"Fold {fold} accuracy: {acc:.6f}, best_iteration: {best_iter}")

cv_accuracy = float(np.mean(fold_accs))
print(f"CV accuracy: {cv_accuracy:.6f}")

final_rounds = max(100, int(np.mean(best_iters)))
dtrain_full = lgb.Dataset(X, label=y_enc, free_raw_data=False)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred_idx = np.argmax(final_model.predict(X_test), axis=1)
test_pred = classes[test_pred_idx].astype(int)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "Cover_Type": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
