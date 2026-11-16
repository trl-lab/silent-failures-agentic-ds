from datetime import datetime
import os
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

# Set global seed for reproducibility
SEED = 42
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -------------------------------------------------------------
# 1. Load Raw Parquet Data
# -------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Ensure Date column is datetime
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

# Ensure IsHoliday is boolean
train_df["IsHoliday"] = train_df["IsHoliday"].astype(bool)
test_df["IsHoliday"] = test_df["IsHoliday"].astype(bool)

# Ensure 0-based integer id in test
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df), dtype=np.int64)

# -------------------------------------------------------------
# 2. Out-Of-Time Temporal Validation Split
# -------------------------------------------------------------
# Hold out data from 2012-01-27 onwards to reflect the test horizon
val_cutoff_date = pd.to_datetime("2012-01-27")

tr_split = train_df[train_df["Date"] < val_cutoff_date].copy().reset_index(drop=True)
val_split = train_df[train_df["Date"] >= val_cutoff_date].copy().reset_index(drop=True)
test_split = test_df.copy().reset_index(drop=True)


# -------------------------------------------------------------
# 3. Feature Engineering
# -------------------------------------------------------------
def extract_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates calendar, cyclical, and major retail holiday features."""
    df = df.copy()

    df["Year"] = df["Date"].dt.year.astype(np.int32)
    df["Month"] = df["Date"].dt.month.astype(np.int32)
    df["Week"] = df["Date"].dt.isocalendar().week.astype(np.int32)
    df["Day"] = df["Date"].dt.day.astype(np.int32)
    df["DayOfWeek"] = df["Date"].dt.dayofweek.astype(np.int32)
    df["Quarter"] = df["Date"].dt.quarter.astype(np.int32)
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(np.int32)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(np.int32)

    # Cyclical seasonal features (Fourier terms for smooth 52-week cycles)
    df["sin_week"] = np.sin(2 * np.pi * df["Week"] / 52.0).astype(np.float32)
    df["cos_week"] = np.cos(2 * np.pi * df["Week"] / 52.0).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12.0).astype(np.float32)

    # Walmart holiday week markers
    df["is_superbowl_week"] = (df["Week"] == 6).astype(np.int32)
    df["is_labor_day_week"] = (df["Week"] == 36).astype(np.int32)
    df["is_thanksgiving_week"] = (df["Week"] == 47).astype(np.int32)
    df["is_christmas_week"] = (df["Week"].isin([51, 52])).astype(np.int32)

    # Lead / Lag indicator around major shopping events
    df["is_pre_thanksgiving"] = (df["Week"] == 46).astype(np.int32)
    df["is_post_thanksgiving"] = (df["Week"] == 48).astype(np.int32)
    df["is_pre_christmas"] = (df["Week"] == 50).astype(np.int32)
    df["is_post_christmas"] = (df["Week"] == 1).astype(np.int32)

    # Store-Dept interaction identifier
    df["Store_Dept"] = (df["Store"].astype(str) + "_" + df["Dept"].astype(str)).astype(
        "category"
    )

    # Handle MarkDown features if present in dataset
    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    if len(markdown_cols) > 0:
        for m_col in markdown_cols:
            df[f"{m_col}_isna"] = df[m_col].isna().astype(np.int32)
            df[m_col] = df[m_col].fillna(0.0)
            df[f"{m_col}_log"] = np.log1p(np.maximum(0, df[m_col])).astype(np.float32)
        df["Total_MarkDown"] = df[markdown_cols].sum(axis=1).astype(np.float32)
        df["Total_MarkDown_log"] = np.log1p(np.maximum(0, df["Total_MarkDown"])).astype(
            np.float32
        )
        df["MarkDown_Count"] = (df[markdown_cols] > 0).sum(axis=1).astype(np.int32)

    # Handle store type if present
    if "Type" in df.columns:
        type_mapping = {"A": 1, "B": 2, "C": 3}
        df["Type_Encoded"] = df["Type"].map(type_mapping).fillna(0).astype(np.int32)

    return df


tr_feat = extract_calendar_features(tr_split)
val_feat = extract_calendar_features(val_split)
test_feat = extract_calendar_features(test_split)

# Target Aggregations Strictly computed on the training partition (No Leakage)
store_dept_stats = (
    tr_split.groupby(["Store", "Dept"])["Weekly_Sales"]
    .agg(
        store_dept_sales_mean="mean",
        store_dept_sales_std="std",
        store_dept_sales_median="median",
        store_dept_sales_min="min",
        store_dept_sales_max="max",
        store_dept_sales_count="count",
    )
    .reset_index()
)

dept_stats = (
    tr_split.groupby(["Dept"])["Weekly_Sales"]
    .agg(
        dept_sales_mean="mean",
        dept_sales_std="std",
        dept_sales_median="median",
        dept_sales_min="min",
        dept_sales_max="max",
    )
    .reset_index()
)

store_stats = (
    tr_split.groupby(["Store"])["Weekly_Sales"]
    .agg(store_sales_mean="mean", store_sales_std="std", store_sales_median="median")
    .reset_index()
)

dept_week_stats = (
    tr_feat.groupby(["Dept", "Week"])["Weekly_Sales"]
    .agg(dept_week_sales_mean="mean", dept_week_sales_median="median")
    .reset_index()
)

store_dept_holiday_stats = (
    tr_feat.groupby(["Store", "Dept", "IsHoliday"])["Weekly_Sales"]
    .agg(store_dept_holiday_sales_mean="mean")
    .reset_index()
)

store_dept_month_stats = (
    tr_feat.groupby(["Store", "Dept", "Month"])["Weekly_Sales"]
    .agg(store_dept_month_sales_mean="mean")
    .reset_index()
)

global_mean = float(tr_split["Weekly_Sales"].mean())
global_median = float(tr_split["Weekly_Sales"].median())


def merge_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Merges training-derived aggregations and fills missing values cleanly."""
    df = df.copy()

    df = df.merge(store_dept_stats, on=["Store", "Dept"], how="left")
    df = df.merge(dept_stats, on=["Dept"], how="left")
    df = df.merge(store_stats, on=["Store"], how="left")
    df = df.merge(dept_week_stats, on=["Dept", "Week"], how="left")
    df = df.merge(
        store_dept_holiday_stats, on=["Store", "Dept", "IsHoliday"], how="left"
    )
    df = df.merge(store_dept_month_stats, on=["Store", "Dept", "Month"], how="left")

    # Impute missing hierarchical aggregations with broader fallbacks
    df["store_dept_sales_mean"] = (
        df["store_dept_sales_mean"].fillna(df["dept_sales_mean"]).fillna(global_mean)
    )
    df["store_dept_sales_median"] = (
        df["store_dept_sales_median"]
        .fillna(df["dept_sales_median"])
        .fillna(global_median)
    )
    df["store_dept_sales_std"] = (
        df["store_dept_sales_std"].fillna(df["dept_sales_std"]).fillna(0.0)
    )
    df["store_dept_sales_min"] = (
        df["store_dept_sales_min"].fillna(df["dept_sales_min"]).fillna(0.0)
    )
    df["store_dept_sales_max"] = (
        df["store_dept_sales_max"].fillna(df["dept_sales_max"]).fillna(0.0)
    )
    df["store_dept_sales_count"] = df["store_dept_sales_count"].fillna(0.0)

    df["dept_sales_mean"] = df["dept_sales_mean"].fillna(global_mean)
    df["dept_sales_std"] = df["dept_sales_std"].fillna(0.0)
    df["dept_sales_median"] = df["dept_sales_median"].fillna(global_median)
    df["dept_sales_min"] = df["dept_sales_min"].fillna(0.0)
    df["dept_sales_max"] = df["dept_sales_max"].fillna(0.0)

    df["store_sales_mean"] = df["store_sales_mean"].fillna(global_mean)
    df["store_sales_std"] = df["store_sales_std"].fillna(0.0)
    df["store_sales_median"] = df["store_sales_median"].fillna(global_median)

    df["dept_week_sales_mean"] = df["dept_week_sales_mean"].fillna(
        df["dept_sales_mean"]
    )
    df["dept_week_sales_median"] = df["dept_week_sales_median"].fillna(
        df["dept_sales_median"]
    )

    df["store_dept_holiday_sales_mean"] = df["store_dept_holiday_sales_mean"].fillna(
        df["store_dept_sales_mean"]
    )
    df["store_dept_month_sales_mean"] = df["store_dept_month_sales_mean"].fillna(
        df["store_dept_sales_mean"]
    )

    # Interaction ratios
    df["store_dept_to_dept_ratio"] = (
        df["store_dept_sales_mean"] / (df["dept_sales_mean"] + 1e-5)
    ).astype(np.float32)
    df["store_dept_to_store_ratio"] = (
        df["store_dept_sales_mean"] / (df["store_sales_mean"] + 1e-5)
    ).astype(np.float32)
    df["seasonal_dept_week_ratio"] = (
        df["dept_week_sales_mean"] / (df["dept_sales_mean"] + 1e-5)
    ).astype(np.float32)

    return df


tr_feat = merge_aggregations(tr_feat)
val_feat = merge_aggregations(val_feat)
test_feat = merge_aggregations(test_feat)

# Assign official competition sample weights (5x for holiday weeks, 1x for regular weeks)
tr_feat["sample_weight"] = np.where(tr_feat["IsHoliday"], 5.0, 1.0).astype(np.float32)
val_feat["sample_weight"] = np.where(val_feat["IsHoliday"], 5.0, 1.0).astype(np.float32)
test_feat["sample_weight"] = np.where(test_feat["IsHoliday"], 5.0, 1.0).astype(
    np.float32
)

# Convert IsHoliday to int32
tr_feat["IsHoliday"] = tr_feat["IsHoliday"].astype(np.int32)
val_feat["IsHoliday"] = val_feat["IsHoliday"].astype(np.int32)
test_feat["IsHoliday"] = test_feat["IsHoliday"].astype(np.int32)

# Ensure categorical types
for col in ["Store", "Dept", "Store_Dept"]:
    if col in tr_feat.columns:
        tr_feat[col] = tr_feat[col].astype("category")
        val_feat[col] = val_feat[col].astype("category")
        test_feat[col] = test_feat[col].astype("category")

# Persist processed sets to working directory
tr_feat.to_parquet("./working/train_processed.parquet", index=False)
val_feat.to_parquet("./working/val_processed.parquet", index=False)
test_feat.to_parquet("./working/test_processed.parquet", index=False)


# -------------------------------------------------------------
# 4. Evaluation Metric & LightGBM Dataset Builder
# -------------------------------------------------------------
def calculate_wmae(
    y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray
) -> float:
    """Computes official Weighted Mean Absolute Error (WMAE)."""
    weights = np.where(is_holiday.astype(bool), 5.0, 1.0)
    wmae_score = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    return float(wmae_score)


def lgb_wmae_metric(
    preds: np.ndarray, train_data: lgb.Dataset
) -> Tuple[str, float, bool]:
    """Custom WMAE evaluation metric callback for LightGBM."""
    labels = train_data.get_label()
    weights = train_data.get_weight()
    if weights is None:
        weights = np.ones_like(labels)
    wmae = np.sum(weights * np.abs(labels - preds)) / np.sum(weights)
    return "wmae", float(wmae), False


CATEGORICAL_FEATURES: List[str] = ["Store", "Dept", "Store_Dept"]
IGNORED_COLUMNS: List[str] = ["Date", "Weekly_Sales", "sample_weight", "id", "Type"]


def get_feature_columns(df_columns: List[str]) -> List[str]:
    """Extracts valid input feature columns excluding targets and metadata."""
    return [col for col in df_columns if col not in IGNORED_COLUMNS]


def get_lgb_model_params(seed: int = 42) -> Dict[str, Any]:
    """LightGBM hyperparameters optimized for WMAE minimization."""
    return {
        "objective": "regression_l1",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def build_lgb_dataset(
    df: pd.DataFrame, feature_cols: List[str], target_col: str = "Weekly_Sales"
) -> lgb.Dataset:
    """Builds a LightGBM Dataset with sample weights and categorical support."""
    X = df[feature_cols]
    y = df[target_col] if target_col in df.columns else None
    weights = df["sample_weight"].values if "sample_weight" in df.columns else None
    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_cols]

    return lgb.Dataset(
        data=X,
        label=y,
        weight=weights,
        categorical_feature=categorical,
        free_raw_data=False,
    )


# -------------------------------------------------------------
# 5. Model Training & Evaluation
# -------------------------------------------------------------
feature_cols = get_feature_columns(list(tr_feat.columns))

dtrain = build_lgb_dataset(tr_feat, feature_cols=feature_cols)
dval = build_lgb_dataset(val_feat, feature_cols=feature_cols)

lgb_params = get_lgb_model_params(seed=SEED)

evals_result = {}
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=0),
    lgb.record_evaluation(evals_result),
]

model = lgb.train(
    params=lgb_params,
    train_set=dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    feval=lgb_wmae_metric,
    callbacks=callbacks,
)

# Hold-out validation inference and exact WMAE score calculation
X_val = val_feat[feature_cols]
val_preds = model.predict(X_val, num_iteration=model.best_iteration)

val_wmae = calculate_wmae(
    y_true=val_feat["Weekly_Sales"].values,
    y_pred=val_preds,
    is_holiday=val_feat["IsHoliday"].values,
)

# -------------------------------------------------------------
# 6. Test Inference & Submission Generation
# -------------------------------------------------------------
X_test = test_feat[feature_cols]
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

if "id" in test_feat.columns:
    test_ids = test_feat["id"].values
else:
    test_ids = np.arange(len(test_preds), dtype=np.int64)

submission_df = pd.DataFrame({"id": test_ids, "Weekly_Sales": test_preds})

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {val_wmae}")
