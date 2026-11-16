import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def preprocess_features(df):
    df = df.copy()

    # Process timestamp features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["dayofyear"] = df["timestamp"].dt.dayofyear
        df["year_month"] = df["year"] * 100 + df["month"]
        df.drop(columns=["timestamp"], inplace=True)

    # Clean anomalous build_year
    if "build_year" in df.columns:
        df.loc[(df["build_year"] < 1600) | (df["build_year"] > 2025), "build_year"] = (
            np.nan
        )

    # Clean anomalous full_sq & life_sq
    if "life_sq" in df.columns and "full_sq" in df.columns:
        df.loc[df["life_sq"] > df["full_sq"], "life_sq"] = np.nan
        df["life_sq_ratio"] = df["life_sq"] / (df["full_sq"] + 1e-5)
        df["extra_sq"] = df["full_sq"] - df["life_sq"]

    if "kitch_sq" in df.columns and "full_sq" in df.columns:
        df.loc[df["kitch_sq"] > df["full_sq"], "kitch_sq"] = np.nan
        df["kitch_sq_ratio"] = df["kitch_sq"] / (df["full_sq"] + 1e-5)

    if "floor" in df.columns and "max_floor" in df.columns:
        df["floor_ratio"] = df["floor"] / (df["max_floor"] + 1e-5)
        df["floor_from_top"] = df["max_floor"] - df["floor"]

    return df


def main():
    print("Loading data...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "price_doc"
    id_col = "id"

    test_ids = test_df[id_col].copy()
    y_train = np.log1p(train_df[target_col].values)

    train_feats = preprocess_features(train_df.drop(columns=[target_col]))
    test_feats = preprocess_features(test_df)

    # Identify categorical columns
    cat_cols = train_feats.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in cat_cols:
        train_feats[col] = train_feats[col].astype(str)
        test_feats[col] = test_feats[col].astype(str)
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_feats[col] = oe.fit_transform(train_feats[[col]])
        test_feats[col] = oe.transform(test_feats[[col]])

    drop_cols = [id_col] if id_col in train_feats.columns else []
    X = train_feats.drop(columns=drop_cols)
    X_test = test_feats.drop(columns=drop_cols)

    # Align columns
    feature_cols = [c for c in X.columns if c in X_test.columns]
    X = X[feature_cols]
    X_test = X_test[feature_cols]

    print(
        f"Features: {len(feature_cols)}, Train shape: {X.shape}, Test shape: {X_test.shape}"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_train)):
        X_tr, y_tr = X.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X.iloc[val_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        test_pred_fold = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += test_pred_fold / kf.n_splits

        fold_rmsle = root_mean_squared_error(y_va, val_pred)
        print(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}")

    overall_rmsle = root_mean_squared_error(y_train, oof_preds)
    print(f"\nOverall OOF RMSLE: {overall_rmsle:.5f}")

    # Inverse log transform to get actual price predictions
    final_test_preds = np.expm1(test_preds)

    os.makedirs("./working", exist_ok=True)
    sub = pd.DataFrame({"id": test_ids, "price_doc": final_test_preds})
    submission_path = "./working/submission.csv"
    sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with shape {sub.shape}")


if __name__ == "__main__":
    main()
