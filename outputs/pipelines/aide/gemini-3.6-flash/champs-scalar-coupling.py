import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

os.makedirs("./working", exist_ok=True)

# 1. Load Parquet Data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Check for external atomic structures if coordinate columns are missing
if "x_0" not in train.columns:
    struct_path = None
    for path in [
        "./input/structures.parquet",
        "./input/original/structures.parquet",
        "./input/structures.csv",
        "./input/original/structures.csv",
    ]:
        if os.path.exists(path):
            struct_path = path
            break

    if struct_path is not None:
        structures = (
            pd.read_parquet(struct_path)
            if struct_path.endswith(".parquet")
            else pd.read_csv(struct_path)
        )

        train = train.merge(
            structures,
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        train = train.rename(
            columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}
        ).drop(columns=["atom_index"], errors="ignore")
        test = test.merge(
            structures,
            left_on=["molecule_name", "atom_index_0"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        test = test.rename(
            columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}
        ).drop(columns=["atom_index"], errors="ignore")

        train = train.merge(
            structures,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        train = train.rename(
            columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}
        ).drop(columns=["atom_index"], errors="ignore")
        test = test.merge(
            structures,
            left_on=["molecule_name", "atom_index_1"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        test = test.rename(
            columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}
        ).drop(columns=["atom_index"], errors="ignore")

# 2. Feature Engineering
combined = pd.concat([train, test], ignore_index=True)

if "x_0" in combined.columns and "x_1" in combined.columns:
    p0 = combined[["x_0", "y_0", "z_0"]].values
    p1 = combined[["x_1", "y_1", "z_1"]].values
    combined["dist"] = np.linalg.norm(p0 - p1, axis=1).astype(np.float32)
    combined["dist_x"] = np.abs(combined["x_0"] - combined["x_1"]).astype(np.float32)
    combined["dist_y"] = np.abs(combined["y_0"] - combined["y_1"]).astype(np.float32)
    combined["dist_z"] = np.abs(combined["z_0"] - combined["z_1"]).astype(np.float32)

    for atom_idx in [0, 1]:
        group_cols = ["molecule_name", f"atom_index_{atom_idx}"]
        combined[f"atom_{atom_idx}_dist_min"] = combined.groupby(group_cols)[
            "dist"
        ].transform("min")
        combined[f"atom_{atom_idx}_dist_max"] = combined.groupby(group_cols)[
            "dist"
        ].transform("max")
        combined[f"atom_{atom_idx}_dist_mean"] = combined.groupby(group_cols)[
            "dist"
        ].transform("mean")
        combined[f"atom_{atom_idx}_dist_std"] = (
            combined.groupby(group_cols)["dist"].transform("std").fillna(0)
        )
        combined[f"atom_{atom_idx}_count"] = combined.groupby(group_cols)[
            "dist"
        ].transform("count")

    combined["mol_dist_min"] = combined.groupby("molecule_name")["dist"].transform(
        "min"
    )
    combined["mol_dist_max"] = combined.groupby("molecule_name")["dist"].transform(
        "max"
    )
    combined["mol_dist_mean"] = combined.groupby("molecule_name")["dist"].transform(
        "mean"
    )

    combined["dist_diff_atom_0_min"] = combined["dist"] - combined["atom_0_dist_min"]
    combined["dist_diff_atom_1_min"] = combined["dist"] - combined["atom_1_dist_min"]

train_len = len(train)
train_df = combined.iloc[:train_len].copy()
test_df = combined.iloc[train_len:].copy()

cat_cols = [c for c in ["type", "atom_0", "atom_1"] if c in train_df.columns]
for col in cat_cols:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

ignore_cols = ["id", "molecule_name", "scalar_coupling_constant"]
features = [c for c in train_df.columns if c not in ignore_cols]

# 3. Model Training with GroupKFold
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

types = train_df["type"].unique()
gkf = GroupKFold(n_splits=5)

for ctype in types:
    type_train_mask = (train_df["type"] == ctype).values
    type_test_mask = (test_df["type"] == ctype).values

    tr_sub = train_df[type_train_mask].reset_index(drop=True)
    te_sub = test_df[type_test_mask].reset_index(drop=True)

    oof_sub = np.zeros(len(tr_sub))
    test_preds_sub = np.zeros(len(te_sub))

    groups = tr_sub["molecule_name"]

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(tr_sub, groups=groups)):
        X_tr, y_tr = (
            tr_sub.iloc[trn_idx][features],
            tr_sub.iloc[trn_idx]["scalar_coupling_constant"],
        )
        X_va, y_va = (
            tr_sub.iloc[val_idx][features],
            tr_sub.iloc[val_idx]["scalar_coupling_constant"],
        )

        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 127,
            "feature_fraction": 0.8,
            "min_child_samples": 20,
            "n_jobs": -1,
            "seed": 42 + fold,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            trn_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_sub[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        if len(te_sub) > 0:
            test_preds_sub += (
                model.predict(te_sub[features], num_iteration=model.best_iteration)
                / 5.0
            )

    oof_preds[type_train_mask] = oof_sub
    if len(te_sub) > 0:
        test_preds[type_test_mask] = test_preds_sub

val_mae = mean_absolute_error(train_df["scalar_coupling_constant"], oof_preds)
print(f"Validation MAE: {val_mae:.5f}")

# 4. Save Submission File
submission = pd.DataFrame(
    {"id": test_df["id"].astype(int), "scalar_coupling_constant": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
