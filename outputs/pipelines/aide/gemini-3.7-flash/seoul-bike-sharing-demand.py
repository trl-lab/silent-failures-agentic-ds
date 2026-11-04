import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def preprocess_data(df):
    df = df.copy()

    # Parse date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df.drop(columns=["Date"], inplace=True)

    # Standardize column names
    col_mapping = {col: col.strip() for col in df.columns}
    df.rename(columns=col_mapping, inplace=True)

    # Clean string / categorical variables
    categorical_cols = ["Seasons", "Holiday", "Functioning Day", "Functioning_Day"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Cyclical hour features if Hour is present
    if "Hour" in df.columns:
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

    # Weather interaction features
    temp_col = [c for c in df.columns if "Temperature" in c and "Dew" not in c]
    dew_col = [
        c
        for c in df.columns
        if "Dew point temperature" in c or "Dew_point_temperature" in c
    ]
    humidity_col = [c for c in df.columns if "Humidity" in c]
    solar_col = [c for c in df.columns if "Solar" in c]
    wind_col = [c for c in df.columns if "Wind" in c]

    if temp_col and dew_col:
        df["Temp_Dew_Diff"] = df[temp_col[0]] - df[dew_col[0]]
    if temp_col and humidity_col:
        df["Temp_Humidity_Ratio"] = df[temp_col[0]] / (df[humidity_col[0]] + 1e-5)
        df["Heat_Index_Approx"] = df[temp_col[0]] + 0.5555 * (
            6.11
            * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + df[temp_col[0]])))
            * (df[humidity_col[0]] / 100)
            - 10
        )
    if solar_col and temp_col:
        df["Solar_Temp_Interaction"] = df[solar_col[0]] * df[temp_col[0]]
    if wind_col and temp_col:
        df["Wind_Chill_Approx"] = (
            13.12
            + 0.6215 * df[temp_col[0]]
            - 11.37 * (df[wind_col[0]] ** 0.16)
            + 0.3965 * df[temp_col[0]] * (df[wind_col[0]] ** 0.16)
        )

    return df


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Rented Bike Count"
    if target_col not in train_df.columns:
        for c in train_df.columns:
            if "rented" in c.lower() or "bike" in c.lower() or "count" in c.lower():
                target_col = c
                break

    y = train_df[target_col].values
    X_train_raw = train_df.drop(columns=[target_col])
    X_test_raw = test_df.copy()

    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])

    # Preprocessing
    X_train = preprocess_data(X_train_raw)
    X_test = preprocess_data(X_test_raw)

    cat_features = [
        col for col in X_train.select_dtypes(include=["category", "object"]).columns
    ]
    for col in cat_features:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Identify functioning day column to post-process zero demand
    func_col = None
    for c in ["Functioning Day", "Functioning_Day", "FunctioningDay"]:
        if c in X_test.columns:
            func_col = c
            break

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "random_state": 42,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        val_pred = np.clip(val_pred, 0, None)

        if func_col is not None:
            is_non_func = X_va[func_col].isin(["No", "no", 0, False])
            val_pred[is_non_func] = 0

        oof_preds[val_idx] = val_pred

        fold_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        fold_test_pred = np.clip(fold_test_pred, 0, None)
        if func_col is not None:
            is_test_non_func = X_test[func_col].isin(["No", "no", 0, False])
            fold_test_pred[is_test_non_func] = 0

        test_preds += fold_test_pred / n_splits

    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(test_preds)), "Rented Bike Count": test_preds}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved successfully to ./working/submission.csv")


if __name__ == "__main__":
    main()
