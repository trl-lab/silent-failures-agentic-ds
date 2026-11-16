import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Define file paths and target column
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
target_col = "Estimated Percentage of Patients with Controlled Blood Pressure"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target and features
y = train_df[target_col].astype(int).values
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])


def preprocess_features(df_train, df_test):
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # Missing value count feature
    df_tr["missing_count"] = df_tr.isna().sum(axis=1)
    df_te["missing_count"] = df_te.isna().sum(axis=1)

    cat_cols = []
    num_cols = []

    for col in list(df_tr.columns):
        if col == "missing_count":
            num_cols.append(col)
            continue

        # Check if column is datetime
        if (
            pd.api.types.is_datetime64_any_dtype(df_tr[col])
            or "date" in col.lower()
            or "time" in col.lower()
        ):
            try:
                dt_tr = pd.to_datetime(df_tr[col], errors="coerce")
                dt_te = pd.to_datetime(df_te[col], errors="coerce")
                if dt_tr.notna().sum() > 0:
                    df_tr[f"{col}_year"] = dt_tr.dt.year.fillna(-1).astype(int)
                    df_tr[f"{col}_month"] = dt_tr.dt.month.fillna(-1).astype(int)
                    df_tr[f"{col}_day"] = dt_tr.dt.day.fillna(-1).astype(int)
                    df_tr[f"{col}_dayofweek"] = dt_tr.dt.dayofweek.fillna(-1).astype(
                        int
                    )
                    df_tr[f"{col}_ts"] = dt_tr.astype("int64") // 10**9

                    df_te[f"{col}_year"] = dt_te.dt.year.fillna(-1).astype(int)
                    df_te[f"{col}_month"] = dt_te.dt.month.fillna(-1).astype(int)
                    df_te[f"{col}_day"] = dt_te.dt.day.fillna(-1).astype(int)
                    df_te[f"{col}_dayofweek"] = dt_te.dt.dayofweek.fillna(-1).astype(
                        int
                    )
                    df_te[f"{col}_ts"] = dt_te.astype("int64") // 10**9

                    num_cols.extend(
                        [
                            f"{col}_year",
                            f"{col}_month",
                            f"{col}_day",
                            f"{col}_dayofweek",
                            f"{col}_ts",
                        ]
                    )

                    df_tr.drop(columns=[col], inplace=True)
                    df_te.drop(columns=[col], inplace=True)
                    continue
            except Exception:
                pass

        # Handle Categorical / Object columns
        if (
            df_tr[col].dtype == "object"
            or df_tr[col].dtype.name == "category"
            or pd.api.types.is_string_dtype(df_tr[col])
        ):
            combined = pd.concat([df_tr[col], df_te[col]], axis=0).astype(str)

            # Frequency encoding
            freq_map = combined.value_counts(normalize=True).to_dict()
            df_tr[f"{col}_freq"] = df_tr[col].astype(str).map(freq_map).fillna(0)
            df_te[f"{col}_freq"] = df_te[col].astype(str).map(freq_map).fillna(0)
            num_cols.append(f"{col}_freq")

            categories = combined.unique()
            cat_type = pd.CategoricalDtype(categories=categories)
            df_tr[col] = df_tr[col].astype(str).astype(cat_type)
            df_te[col] = df_te[col].astype(str).astype(cat_type)
            cat_cols.append(col)
        else:
            # Numerical columns
            df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce")
            df_te[col] = pd.to_numeric(df_te[col], errors="coerce")
            num_cols.append(col)

    return df_tr, df_te, cat_cols


X, X_test, cat_cols = preprocess_features(X, X_test)

# Setup 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1200,
}

# Perform Cross-Validation with nested Target Encoding
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_v = X.iloc[val_idx].copy()
    y_v = y[val_idx]
    X_te = X_test.copy()

    # Out-of-fold target encoding for train and mapping for val & test
    smooth_weight = 10.0
    global_mean = y_tr.mean()

    for col in cat_cols:
        tr_series = X_tr[col].astype(str)
        v_series = X_v[col].astype(str)
        te_series = X_te[col].astype(str)

        # Full train stats for val and test
        stats = (
            pd.DataFrame({"cat": tr_series, "y": y_tr})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        te_mapping = (stats["count"] * stats["mean"] + smooth_weight * global_mean) / (
            stats["count"] + smooth_weight
        )
        te_dict = te_mapping.to_dict()

        X_v[f"{col}_te"] = v_series.map(te_dict).fillna(global_mean).astype(float)
        X_te[f"{col}_te"] = te_series.map(te_dict).fillna(global_mean).astype(float)

        # Inner CV for X_tr to prevent target leakage
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=fold)
        tr_te_values = pd.Series(index=X_tr.index, dtype=float)

        for in_tr_idx, in_val_idx in inner_kf.split(X_tr):
            in_tr_cat = tr_series.iloc[in_tr_idx]
            in_tr_y = y_tr[in_tr_idx]
            in_stats = (
                pd.DataFrame({"cat": in_tr_cat, "y": in_tr_y})
                .groupby("cat")["y"]
                .agg(["count", "mean"])
            )
            in_global_mean = in_tr_y.mean()
            in_mapping = (
                in_stats["count"] * in_stats["mean"] + smooth_weight * in_global_mean
            ) / (in_stats["count"] + smooth_weight)
            in_dict = in_mapping.to_dict()
            tr_te_values.iloc[in_val_idx] = (
                tr_series.iloc[in_val_idx]
                .map(in_dict)
                .fillna(in_global_mean)
                .astype(float)
            )

        X_tr[f"{col}_te"] = tr_te_values.values

    model = lgb.LGBMClassifier(**lgb_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=callbacks)

    val_pred = model.predict_proba(X_v)[:, 1]
    oof_preds[val_idx] = val_pred

    fold_auc = roc_auc_score(y_v, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_te)[:, 1] / n_splits

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC-AUC: {overall_auc:.5f}")

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)
print(
    f"Submission successfully saved to {submission_path} with shape {submission.shape}"
)
