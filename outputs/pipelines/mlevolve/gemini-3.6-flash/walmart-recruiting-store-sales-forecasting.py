import json
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb

warnings.filterwarnings("ignore")


def calculate_wmae(y_true, y_pred, weights):
    """Computes exact Weighted Mean Absolute Error (WMAE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # ----------------------------------------------------
    # 1. Data Loading & Initial Preprocessing
    # ----------------------------------------------------
    print("Loading datasets from ./input...")
    df_train = pd.read_parquet("./input/train.parquet")
    df_test = pd.read_parquet("./input/test.parquet")

    if "id" not in df_test.columns:
        df_test["id"] = np.arange(len(df_test))

    df_train["is_test"] = 0
    df_test["is_test"] = 1

    combined_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    combined_df["Date"] = pd.to_datetime(combined_df["Date"])
    combined_df["Store"] = combined_df["Store"].astype(int)
    combined_df["Dept"] = combined_df["Dept"].astype(int)
    combined_df["IsHoliday"] = combined_df["IsHoliday"].astype(bool)

    # Evaluation Weight (WMAE: 5.0 for holiday weeks, 1.0 otherwise)
    combined_df["weight"] = np.where(combined_df["IsHoliday"], 5.0, 1.0)

    # ----------------------------------------------------
    # 2. Date & Calendar Feature Engineering
    # ----------------------------------------------------
    combined_df["Year"] = combined_df["Date"].dt.year
    combined_df["Month"] = combined_df["Date"].dt.month
    combined_df["WeekOfYear"] = combined_df["Date"].dt.isocalendar().week.astype(int)
    combined_df["Day"] = combined_df["Date"].dt.day
    combined_df["DayOfWeek"] = combined_df["Date"].dt.dayofweek
    combined_df["DayOfYear"] = combined_df["Date"].dt.dayofyear
    combined_df["Is_Month_End"] = combined_df["Date"].dt.is_month_end.astype(int)
    combined_df["Is_Month_Start"] = combined_df["Date"].dt.is_month_start.astype(int)

    # Cyclical Calendar Transformations
    combined_df["sin_week"] = np.sin(2 * np.pi * combined_df["WeekOfYear"] / 52.0)
    combined_df["cos_week"] = np.cos(2 * np.pi * combined_df["WeekOfYear"] / 52.0)
    combined_df["sin_month"] = np.sin(2 * np.pi * combined_df["Month"] / 12.0)
    combined_df["cos_month"] = np.cos(2 * np.pi * combined_df["Month"] / 12.0)

    # Holiday Indicators & Relative Proximities
    super_bowl_weeks = [6]
    labor_day_weeks = [36]
    thanksgiving_weeks = [47]
    christmas_weeks = [51, 52]

    combined_df["is_super_bowl"] = (
        combined_df["WeekOfYear"].isin(super_bowl_weeks).astype(int)
    )
    combined_df["is_labor_day"] = (
        combined_df["WeekOfYear"].isin(labor_day_weeks).astype(int)
    )
    combined_df["is_thanksgiving"] = (
        combined_df["WeekOfYear"].isin(thanksgiving_weeks).astype(int)
    )
    combined_df["is_christmas"] = (
        combined_df["WeekOfYear"].isin(christmas_weeks).astype(int)
    )

    combined_df["weeks_to_thanksgiving"] = 47 - combined_df["WeekOfYear"]
    combined_df["weeks_to_christmas"] = 52 - combined_df["WeekOfYear"]

    # Markdown Features
    markdown_cols = [
        col for col in combined_df.columns if "MarkDown" in col or "markdown" in col
    ]
    if markdown_cols:
        for col in markdown_cols:
            combined_df[col] = combined_df[col].fillna(0.0)
        combined_df["markdown_sum"] = combined_df[markdown_cols].sum(axis=1)
        combined_df["markdown_non_zero_cnt"] = (combined_df[markdown_cols] > 0).sum(
            axis=1
        )
        combined_df["markdown_holiday_interaction"] = (
            combined_df["markdown_sum"] * combined_df["IsHoliday"].astype(float)
        )

    # ----------------------------------------------------
    # 3. Multi-Week Annual Lags & Windowed Rolling Features
    # ----------------------------------------------------
    combined_df = combined_df.sort_values(["Store", "Dept", "Date"]).reset_index(
        drop=True
    )
    grouped_sd = combined_df.groupby(["Store", "Dept"])["Weekly_Sales"]
    combined_df["sales_lag_51"] = grouped_sd.shift(51)
    combined_df["sales_lag_52"] = grouped_sd.shift(52)
    combined_df["sales_lag_53"] = grouped_sd.shift(53)
    combined_df["sales_lag_54"] = grouped_sd.shift(54)

    lag_window_matrix = combined_df[
        ["sales_lag_51", "sales_lag_52", "sales_lag_53", "sales_lag_54"]
    ]
    combined_df["sales_lag_window_mean"] = lag_window_matrix.mean(axis=1)
    combined_df["sales_lag_window_min"] = lag_window_matrix.min(axis=1)
    combined_df["sales_lag_window_max"] = lag_window_matrix.max(axis=1)
    combined_df["sales_lag_window_std"] = lag_window_matrix.std(axis=1)

    combined_df["sales_lag_52_roll_mean_4"] = combined_df.groupby(["Store", "Dept"])[
        "sales_lag_52"
    ].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    train_engineered = combined_df[combined_df["is_test"] == 0].copy()
    test_engineered = combined_df[combined_df["is_test"] == 1].copy()

    train_engineered.drop(columns=["is_test"], inplace=True)
    test_engineered.drop(columns=["is_test"], inplace=True)

    # ----------------------------------------------------
    # 4. Temporal Validation Split
    # ----------------------------------------------------
    val_cutoff_date = pd.to_datetime("2012-01-27")

    tr_data = train_engineered[train_engineered["Date"] < val_cutoff_date].copy()
    val_data = train_engineered[train_engineered["Date"] >= val_cutoff_date].copy()
    te_data = test_engineered.copy()

    # ----------------------------------------------------
    # 5. Non-Leaking Target Aggregations (Fit on tr_data)
    # ----------------------------------------------------
    print("Computing historical target statistics strictly on training split...")
    global_mean_sales = tr_data["Weekly_Sales"].mean()

    # Store-Dept holiday ratio feature fit on training set
    sd_holiday = (
        tr_data.groupby(["Store", "Dept", "IsHoliday"])["Weekly_Sales"]
        .mean()
        .unstack()
    )
    if True in sd_holiday.columns and False in sd_holiday.columns:
        sd_holiday["sd_holiday_ratio"] = (
            sd_holiday[True] / (sd_holiday[False] + 1e-5)
        ).fillna(1.0)
    else:
        sd_holiday["sd_holiday_ratio"] = 1.0
    sd_holiday_ratio_df = sd_holiday[["sd_holiday_ratio"]].reset_index()

    store_dept_week_stats = (
        tr_data.groupby(["Store", "Dept", "WeekOfYear"])["Weekly_Sales"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(
            columns={
                "mean": "sdw_sales_mean",
                "std": "sdw_sales_std",
                "median": "sdw_sales_median",
            }
        )
    )

    dept_week_stats = (
        tr_data.groupby(["Dept", "WeekOfYear"])["Weekly_Sales"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "dw_sales_mean", "std": "dw_sales_std"})
    )

    store_dept_stats = (
        tr_data.groupby(["Store", "Dept"])["Weekly_Sales"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "sd_sales_mean", "std": "sd_sales_std"})
    )

    dept_stats = (
        tr_data.groupby(["Dept"])["Weekly_Sales"]
        .agg(["mean"])
        .reset_index()
        .rename(columns={"mean": "d_sales_mean"})
    )

    store_stats = (
        tr_data.groupby(["Store"])["Weekly_Sales"]
        .agg(["mean"])
        .reset_index()
        .rename(columns={"mean": "s_sales_mean"})
    )

    def merge_target_stats(df):
        df = df.merge(
            sd_holiday_ratio_df, on=["Store", "Dept"], how="left"
        )
        df["sd_holiday_ratio"] = df["sd_holiday_ratio"].fillna(1.0)
        df["holiday_interaction_ratio"] = (
            df["sd_holiday_ratio"] * df["IsHoliday"].astype(float)
        )

        df = df.merge(
            store_dept_week_stats, on=["Store", "Dept", "WeekOfYear"], how="left"
        )
        df = df.merge(dept_week_stats, on=["Dept", "WeekOfYear"], how="left")
        df = df.merge(store_dept_stats, on=["Store", "Dept"], how="left")
        df = df.merge(dept_stats, on=["Dept"], how="left")
        df = df.merge(store_stats, on=["Store"], how="left")

        df["sdw_sales_mean"] = (
            df["sdw_sales_mean"]
            .fillna(df["dw_sales_mean"])
            .fillna(df["sd_sales_mean"])
            .fillna(df["d_sales_mean"])
            .fillna(df["s_sales_mean"])
            .fillna(global_mean_sales)
        )

        df["sdw_sales_std"] = df["sdw_sales_std"].fillna(0.0)
        df["dw_sales_std"] = df["dw_sales_std"].fillna(0.0)
        df["sd_sales_std"] = df["sd_sales_std"].fillna(0.0)

        for lag_col in ["sales_lag_51", "sales_lag_52", "sales_lag_53", "sales_lag_54"]:
            if lag_col in df.columns:
                df[lag_col] = (
                    df[lag_col].fillna(df["sdw_sales_mean"]).fillna(global_mean_sales)
                )

        for w_col in ["sales_lag_window_mean", "sales_lag_window_min", "sales_lag_window_max"]:
            if w_col in df.columns:
                df[w_col] = (
                    df[w_col].fillna(df["sdw_sales_mean"]).fillna(global_mean_sales)
                )
        if "sales_lag_window_std" in df.columns:
            df["sales_lag_window_std"] = df["sales_lag_window_std"].fillna(0.0)

        df["sales_lag_52_roll_mean_4"] = (
            df["sales_lag_52_roll_mean_4"]
            .fillna(df["sdw_sales_mean"])
            .fillna(global_mean_sales)
        )

        return df

    tr_data = merge_target_stats(tr_data)
    val_data = merge_target_stats(val_data)
    te_data = merge_target_stats(te_data)

    te_data = te_data.sort_values("id").reset_index(drop=True)

    exclude_cols = {"Date", "Weekly_Sales", "is_test", "id", "weight", "Type"}
    feature_cols = [col for col in tr_data.columns if col not in exclude_cols]

    if "Type" in tr_data.columns:
        type_mapping = {"A": 0, "B": 1, "C": 2}
        tr_data["Type_encoded"] = tr_data["Type"].map(type_mapping).fillna(-1)
        val_data["Type_encoded"] = val_data["Type"].map(type_mapping).fillna(-1)
        te_data["Type_encoded"] = te_data["Type"].map(type_mapping).fillna(-1)
        if "Type_encoded" not in feature_cols:
            feature_cols.append("Type_encoded")

    for col in feature_cols:
        if tr_data[col].dtype == "bool":
            tr_data[col] = tr_data[col].astype(int)
            val_data[col] = val_data[col].astype(int)
            te_data[col] = te_data[col].astype(int)

    # Matrix construction
    X_tr = tr_data[feature_cols].fillna(0.0).values
    y_tr = tr_data["Weekly_Sales"].values
    w_tr = tr_data["weight"].values

    X_val = val_data[feature_cols].fillna(0.0).values
    y_val = val_data["Weekly_Sales"].values
    w_val = val_data["weight"].values

    X_te = te_data[feature_cols].fillna(0.0).values
    test_ids = te_data["id"].values

    print(
        f"Train: {X_tr.shape[0]} rows | Val: {X_val.shape[0]} rows | Test: {X_te.shape[0]} rows | Features: {len(feature_cols)}"
    )

    val_preds_dict = {}
    test_preds_dict = {}

    # ----------------------------------------------------
    # 6. Model 1: LightGBM (Weighted L1 Objective)
    # ----------------------------------------------------
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=3500,
        learning_rate=0.03,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        random_state=42,
        n_jobs=-1,
    )

    lgb_model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_model.booster_.save_model("./working/lgb_model.txt")
    best_lgb = lgb.Booster(model_file="./working/lgb_model.txt")
    val_preds_dict["lgb"] = best_lgb.predict(X_val)
    test_preds_dict["lgb"] = best_lgb.predict(X_te)
    print(
        f"LightGBM Validation WMAE: {calculate_wmae(y_val, val_preds_dict['lgb'], w_val):.4f}"
    )

    # ----------------------------------------------------
    # 7. Model 2: XGBoost (Weighted L1 Objective)
    # ----------------------------------------------------
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    xgb_model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )

    xgb_model.save_model("./working/xgb_model.json")
    val_preds_dict["xgb"] = xgb_model.predict(X_val)
    test_preds_dict["xgb"] = xgb_model.predict(X_te)
    print(
        f"XGBoost Validation WMAE: {calculate_wmae(y_val, val_preds_dict['xgb'], w_val):.4f}"
    )

    # ----------------------------------------------------
    # 8. Model 3: CatBoost (Weighted MAE Objective)
    # ----------------------------------------------------
    print("Training CatBoost model...")
    cb_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=3000,
        learning_rate=0.04,
        depth=8,
        subsample=0.8,
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )

    cb_model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )

    cb_model.save_model("./working/cb_model.cbm")
    val_preds_dict["cb"] = cb_model.predict(X_val)
    test_preds_dict["cb"] = cb_model.predict(X_te)
    print(
        f"CatBoost Validation WMAE: {calculate_wmae(y_val, val_preds_dict['cb'], w_val):.4f}"
    )

    # ----------------------------------------------------
    # 9. Ensemble Optimization (SLSQP on Validation WMAE)
    # ----------------------------------------------------
    print("Optimizing ensemble weights on validation WMAE...")
    model_names = list(val_preds_dict.keys())
    val_matrix = np.column_stack([val_preds_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_preds_dict[m] for m in model_names])

    def objective_func(weights):
        weights = np.array(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        blend_val = np.dot(val_matrix, weights)
        return calculate_wmae(y_val, blend_val, w_val)

    init_weights = [1.0 / len(model_names)] * len(model_names)
    bounds = [(0.0, 1.0)] * len(model_names)

    res = minimize(
        objective_func, init_weights, method="SLSQP", bounds=bounds, tol=1e-6
    )
    opt_weights = res.x / np.sum(res.x)

    for m, w in zip(model_names, opt_weights):
        print(f"Optimal Weight - {m}: {w:.4f}")

    val_ensemble_preds = np.dot(val_matrix, opt_weights)
    test_ensemble_preds = np.dot(test_matrix, opt_weights)

    final_val_wmae = calculate_wmae(y_val, val_ensemble_preds, w_val)

    # ----------------------------------------------------
    # 10. Generate Submission File
    # ----------------------------------------------------
    submission_df = pd.DataFrame(
        {"id": test_ids.astype(int), "Weekly_Sales": test_ensemble_preds}
    )

    assert len(submission_df) == len(
        te_data
    ), "Submission length mismatch with test set!"
    assert (
        not submission_df["Weekly_Sales"].isnull().any()
    ), "Found NaN values in predictions!"

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(
        f"Saved submission file to {submission_path} with shape {submission_df.shape}"
    )

    # Required final output line
    print(f"Final Validation Score: {final_val_wmae:.5f}")


if __name__ == "__main__":
    main()
