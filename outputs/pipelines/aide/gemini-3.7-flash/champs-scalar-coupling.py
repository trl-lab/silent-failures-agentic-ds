import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error


def main():
    # 1. Load data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    output_path = "./working/submission.csv"

    os.makedirs("./working", exist_ok=True)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 2. Feature engineering
    def engineer_features(df):
        # Calculate spatial distance if coordinates exist
        coord_cols_0 = ["x_0", "y_0", "z_0"]
        coord_cols_1 = ["x_1", "y_1", "z_1"]

        if all(c in df.columns for c in coord_cols_0 + coord_cols_1):
            df["dx"] = df["x_0"] - df["x_1"]
            df["dy"] = df["y_0"] - df["y_1"]
            df["dz"] = df["z_0"] - df["z_1"]
            df["dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)
            df["dist_inv2"] = 1.0 / (df["dist"] ** 2 + 1e-6)
            df["dist_inv3"] = 1.0 / (df["dist"] ** 3 + 1e-6)
            df["abs_dx"] = np.abs(df["dx"])
            df["abs_dy"] = np.abs(df["dy"])
            df["abs_dz"] = np.abs(df["dz"])
        elif "dist" in df.columns:
            df["dist_inv2"] = 1.0 / (df["dist"] ** 2 + 1e-6)
            df["dist_inv3"] = 1.0 / (df["dist"] ** 3 + 1e-6)

        return df

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Target and identifier columns to exclude from training
    target_col = "scalar_coupling_constant"
    id_col = "id" if "id" in train_df.columns else None

    ignore_cols = {target_col, "id", "molecule_name", "type"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    # Convert categorical columns
    categorical_cols = [
        c
        for c in feature_cols
        if train_df[c].dtype == "object" or str(train_df[c].dtype) == "category"
    ]
    for cat_c in categorical_cols:
        train_df[cat_c] = train_df[cat_c].astype("category")
        test_df[cat_c] = test_df[cat_c].astype("category")

    # Setup cross validation
    n_splits = 5
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))

    # Check if 'type' column is available for specialized per-type models
    has_type = "type" in train_df.columns
    types = train_df["type"].unique() if has_type else [None]

    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for t in types:
        if t is not None:
            train_idx_type = train_df[train_df["type"] == t].index
            test_idx_type = test_df[test_df["type"] == t].index
            sub_train = train_df.loc[train_idx_type].reset_index(drop=True)
            sub_test = test_df.loc[test_idx_type].reset_index(drop=True)
        else:
            train_idx_type = train_df.index
            test_idx_type = test_df.index
            sub_train = train_df
            sub_test = test_df

        sub_oof = np.zeros(len(sub_train))
        sub_test_preds = np.zeros(len(sub_test))

        # Split by molecule_name if available to avoid leakage
        if "molecule_name" in sub_train.columns:
            gkf = GroupKFold(n_splits=n_splits)
            splits = gkf.split(sub_train, groups=sub_train["molecule_name"])
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = kf.split(sub_train)

        for fold, (trn_idx, val_idx) in enumerate(splits):
            X_trn, y_trn = (
                sub_train.iloc[trn_idx][feature_cols],
                sub_train.iloc[trn_idx][target_col],
            )
            X_val, y_val = (
                sub_train.iloc[val_idx][feature_cols],
                sub_train.iloc[val_idx][target_col],
            )

            model = lgb.LGBMRegressor(**lgb_params)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

            model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=callbacks)

            sub_oof[val_idx] = model.predict(X_val)
            if len(sub_test) > 0:
                sub_test_preds += model.predict(sub_test[feature_cols]) / n_splits

        oof_predictions[train_idx_type] = sub_oof
        if len(test_idx_type) > 0:
            test_predictions[test_idx_type] = sub_test_preds

    overall_mae = mean_absolute_error(train_df[target_col], oof_predictions)
    print(f"Overall OOF MAE: {overall_mae:.5f}")

    # Save submission
    sub = pd.DataFrame()
    if "id" in test_df.columns:
        sub["id"] = test_df["id"]
    else:
        sub["id"] = np.arange(len(test_df))

    sub["scalar_coupling_constant"] = test_predictions
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with shape {sub.shape}")


if __name__ == "__main__":
    main()
