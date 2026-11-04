import gc
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def reduce_mem_usage(df, verbose=False):
    """Iterate through all columns of a dataframe and downcast numeric types."""
    for col in df.columns:
        col_type = df[col].dtype
        if (
            col_type != object
            and not isinstance(df[col].dtype, pd.CategoricalDtype)
            and not pd.api.types.is_datetime64_any_dtype(col_type)
        ):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df


def get_lgbm_params(custom_params: dict = None) -> dict:
    """Returns competitive hyperparameters for LightGBM on IEEE-CIS Fraud Detection."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 256,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return params


def compute_roc_auc_metric(y_true, y_pred_prob) -> float:
    """Computes standard ROC-AUC score matching the official competition evaluation metric."""
    return float(roc_auc_score(y_true, y_pred_prob))


def compute_temporal_velocity(df, uid_cols, time_col="TransactionDT", windows=None):
    """Computes rolling time-window transaction counts per UID."""
    if windows is None:
        windows = {
            "6h": 6 * 3600,
            "24h": 24 * 3600,
            "72h": 72 * 3600,
            "168h": 168 * 3600,
        }
    timestamps = df[time_col].values
    n_rows = len(df)
    for uid_col in uid_cols:
        if uid_col not in df.columns:
            continue
        uid_indices = df.groupby(uid_col, sort=False).indices
        counts_dict = {w_name: np.zeros(n_rows, dtype=np.int32) for w_name in windows}
        for _, idxs in uid_indices.items():
            if len(idxs) == 1:
                for w_name in windows:
                    counts_dict[w_name][idxs] = 1
                continue
            ts = timestamps[idxs]
            for w_name, w_secs in windows.items():
                left_pos = np.searchsorted(ts, ts - w_secs, side="left")
                counts_dict[w_name][idxs] = np.arange(len(ts), dtype=np.int32) - left_pos + 1
        for w_name in windows:
            df[f"{uid_col}_count_{w_name}"] = counts_dict[w_name].astype(np.int32)
    return df


class FraudEnsembleModel:
    """Heterogeneous ensemble combining LightGBM deep trees, XGBoost hist booster, and ExtraTrees LightGBM."""

    def __init__(
        self,
        n_estimators: int = 3000,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.weights = [0.45, 0.35, 0.20]
        self.lgb_model_1 = None
        self.xgb_model = None
        self.lgb_model_et = None

    def fit(self, dtrain, dval, X_v, y_v):
        val_preds_list = []
        X_tr = dtrain.data
        y_tr = dtrain.get_label()

        # Model 1: Deep Leaf-wise LightGBM
        print("Training Ensemble Model 1/3: Deep Leaf-wise LightGBM...")
        callbacks_1 = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]
        params_1 = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 256,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.lgb_model_1 = lgb.train(
            params=params_1,
            train_set=dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks_1,
        )
        val_p1 = self.lgb_model_1.predict(
            X_v, num_iteration=self.lgb_model_1.best_iteration
        )
        val_preds_list.append(val_p1)
        print(
            f"Model 1 (LightGBM Deep) Best Iteration: {self.lgb_model_1.best_iteration} | Val AUC: {compute_roc_auc_metric(y_v, val_p1):.6f}"
        )

        # Model 2: Depth-constrained regularized XGBoost (hist booster)
        print("Training Ensemble Model 2/3: Regularized XGBoost (hist)...")
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 9,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
            "random_state": 4242,
            "n_jobs": -1,
        }
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
        dval_xgb = xgb.DMatrix(X_v, label=y_v)
        self.xgb_model = xgb.train(
            params=xgb_params,
            dtrain=dtrain_xgb,
            num_boost_round=self.n_estimators,
            evals=[(dtrain_xgb, "train"), (dval_xgb, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
        )
        val_p2 = self.xgb_model.predict(
            dval_xgb, iteration_range=(0, self.xgb_model.best_iteration + 1)
        )
        val_preds_list.append(val_p2)
        print(
            f"Model 2 (XGBoost Hist) Best Iteration: {self.xgb_model.best_iteration} | Val AUC: {compute_roc_auc_metric(y_v, val_p2):.6f}"
        )
        del dtrain_xgb, dval_xgb
        gc.collect()

        # Model 3: Randomized ExtraTrees LightGBM
        print("Training Ensemble Model 3/3: ExtraTrees LightGBM...")
        callbacks_3 = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=100),
        ]
        params_3 = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "extra_trees": True,
            "learning_rate": 0.03,
            "num_leaves": 256,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": 2024,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.lgb_model_et = lgb.train(
            params=params_3,
            train_set=dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=callbacks_3,
        )
        val_p3 = self.lgb_model_et.predict(
            X_v, num_iteration=self.lgb_model_et.best_iteration
        )
        val_preds_list.append(val_p3)
        print(
            f"Model 3 (LightGBM ExtraTrees) Best Iteration: {self.lgb_model_et.best_iteration} | Val AUC: {compute_roc_auc_metric(y_v, val_p3):.6f}"
        )

        blended_val = np.zeros(len(y_v), dtype=np.float64)
        for w, p in zip(self.weights, val_preds_list):
            blended_val += w * (rankdata(p) / len(p))
        return blended_val

    def predict(self, X):
        p1 = self.lgb_model_1.predict(
            X, num_iteration=self.lgb_model_1.best_iteration
        )
        dX = xgb.DMatrix(X)
        p2 = self.xgb_model.predict(
            dX, iteration_range=(0, self.xgb_model.best_iteration + 1)
        )
        del dX
        gc.collect()
        p3 = self.lgb_model_et.predict(
            X, num_iteration=self.lgb_model_et.best_iteration
        )

        preds = (
            self.weights[0] * (rankdata(p1) / len(p1))
            + self.weights[1] * (rankdata(p2) / len(p2))
            + self.weights[2] * (rankdata(p3) / len(p3))
        )
        return preds


def build_model(
    params: dict = None,
    n_estimators: int = 3000,
    early_stopping_rounds: int = 50,
) -> FraudEnsembleModel:
    """Factory function to build and configure the fraud classification ensemble model."""
    return FraudEnsembleModel(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )


# -------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# -------------------------------------------------------------
print("Loading train and test parquet datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Sort train chronologically by TransactionDT to preserve temporal structure
train_df = train_df.sort_values("TransactionDT").reset_index(drop=True)

# Separate target and identification
target_col = "isFraud"
y_train_full = train_df[target_col].copy()

# Concatenate for consistent categorical and structural transformations
n_train = len(train_df)
combined = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)
del train_df, test_df
gc.collect()

print(f"Combined data shape: {combined.shape}")

# Temporal Feature Engineering
combined["DT_day"] = (combined["TransactionDT"] // 86400).astype(np.int16)
combined["DT_hour"] = ((combined["TransactionDT"] // 3600) % 24).astype(np.int8)
combined["DT_day_of_week"] = ((combined["TransactionDT"] // 86400) % 7).astype(np.int8)

# Stationary Delta Features (D-columns drift normalization D1-D15)
d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in combined.columns]
for d in d_cols:
    combined[f"{d}_norm"] = combined[d] - combined["DT_day"]

# Transaction Amount Transformations
combined["TransactionAmt_log"] = np.log1p(combined["TransactionAmt"]).astype(np.float32)
combined["TransactionAmt_decimal"] = (
    combined["TransactionAmt"] - np.floor(combined["TransactionAmt"])
).astype(np.float32)
combined["TransactionAmt_cents"] = (
    (combined["TransactionAmt"] * 100).round() % 100
).astype(np.float32)

# Email Domain Features & Risk Grouping
email_mapping = {
    "gmail": "google", "att.net": "att", "twc.com": "twc",
    "scranton.edu": "edu", "optonline.net": "other",
    "hotmail.co.uk": "microsoft", "comcast.net": "comcast",
    "yahoo.com.mx": "yahoo", "yahoo.fr": "yahoo",
    "yahoo.es": "yahoo", "yahoo.de": "yahoo",
    "hotmail.de": "microsoft", "chase.com": "chase",
    "yahoo.co.uk": "yahoo", "yahoo.com": "yahoo",
    "protonmail.com": "proton", "hotmail.com": "microsoft",
    "gmail.com": "google", "live.com": "microsoft",
    "frontiernet.net": "other", "rocketmail.com": "yahoo",
    "live.com.mx": "microsoft", "icloud.com": "apple",
    "q.com": "other", "windstream.net": "other",
    "aim.com": "aol", "earthlink.net": "other",
    "netzero.com": "other", "bellsouth.net": "other",
    "web.de": "other", "live.fr": "microsoft",
    "ymail.com": "yahoo", "cableone.net": "other",
    "sbcglobal.net": "att", "sc.rr.com": "other",
    "suddenlink.net": "other", "centurylink.net": "other",
    "embarqmail.com": "other", "roadrunner.com": "other",
    "mail.com": "other", "verizon.net": "verizon",
    "cfl.rr.com": "other", "frontier.com": "other",
    "anonymous.com": "anon", "mac.com": "apple",
    "cox.net": "cox", "dodo.com.au": "other",
    "outlook.es": "microsoft", "aol.com": "aol",
    "broadstripe.net": "other", "charter.net": "charter",
    "juno.com": "other", "gmx.de": "other",
    "outlook.com": "microsoft", "msn.com": "microsoft",
    "servicios-ta.com": "other", "netzero.net": "other",
    "ptd.net": "other",
}

for email_col in ["P_emaildomain", "R_emaildomain"]:
    if email_col in combined.columns:
        combined[f"{email_col}_prefix"] = (
            combined[email_col]
            .astype(str)
            .apply(
                lambda x: x.split(".")[0] if pd.notna(x) and x != "nan" else "missing"
            )
        )
        combined[f"{email_col}_suffix"] = (
            combined[email_col]
            .astype(str)
            .apply(
                lambda x: (
                    x.split(".")[-1]
                    if pd.notna(x) and x != "nan" and "." in x
                    else "missing"
                )
            )
        )
        combined[f"{email_col}_group"] = (
            combined[email_col].map(email_mapping).fillna("other")
        )

if "P_emaildomain" in combined.columns and "R_emaildomain" in combined.columns:
    combined["email_match"] = (
        combined["P_emaildomain"].astype(str) == combined["R_emaildomain"].astype(str)
    ).astype(np.int8)
    combined.loc[
        combined["P_emaildomain"].isna() | combined["R_emaildomain"].isna(),
        "email_match",
    ] = -1

# C-column Interaction Ratios and Differences
c_pairs = [("C1", "C2"), ("C13", "C1"), ("C14", "C1"), ("C6", "C1"), ("C11", "C1"), ("C5", "C1"), ("C8", "C1")]
for c1, c2 in c_pairs:
    if c1 in combined.columns and c2 in combined.columns:
        combined[f"{c1}_ratio_{c2}"] = (
            (combined[c1] + 1e-3) / (combined[c2] + 1e-3)
        ).astype(np.float32)
        combined[f"{c1}_diff_{c2}"] = (combined[c1] - combined[c2]).astype(np.float32)

# Missingness indicators across column blocks
v_cols = [c for c in combined.columns if c.startswith("V")]
d_cols_all = [c for c in combined.columns if c.startswith("D") and not c.endswith("_norm")]
m_cols = [c for c in combined.columns if c.startswith("M")]

combined["V_nan_count"] = combined[v_cols].isna().sum(axis=1).astype(np.int16)
combined["D_nan_count"] = combined[d_cols_all].isna().sum(axis=1).astype(np.int16)
combined["M_nan_count"] = combined[m_cols].isna().sum(axis=1).astype(np.int16)
combined["total_nan_count"] = (
    combined["V_nan_count"] + combined["D_nan_count"] + combined["M_nan_count"]
).astype(np.int16)

# High-order User/Entity Interaction Keys & IEEE-CIS Pseudo-UIDs
if "card1" in combined.columns and "card2" in combined.columns:
    combined["uid_c1_c2"] = (
        combined["card1"].astype(str) + "_" + combined["card2"].astype(str)
    )
if "card1" in combined.columns and "addr1" in combined.columns:
    combined["uid_c1_a1"] = (
        combined["card1"].astype(str) + "_" + combined["addr1"].astype(str)
    )
if (
    "card1" in combined.columns
    and "addr1" in combined.columns
    and "P_emaildomain" in combined.columns
):
    combined["uid_c1_a1_pemail"] = (
        combined["card1"].astype(str)
        + "_"
        + combined["addr1"].astype(str)
        + "_"
        + combined["P_emaildomain"].astype(str)
    )
if (
    "card1" in combined.columns
    and "addr1" in combined.columns
    and "D1_norm" in combined.columns
):
    combined["uid_c1_a1_d1"] = (
        combined["card1"].astype(str)
        + "_"
        + combined["addr1"].astype(str)
        + "_"
        + combined["D1_norm"].round().astype(str)
    )
if (
    "card1" in combined.columns
    and "card2" in combined.columns
    and "card3" in combined.columns
    and "card5" in combined.columns
    and "addr1" in combined.columns
):
    combined["uid_card_full"] = (
        combined["card1"].astype(str)
        + "_"
        + combined["card2"].astype(str)
        + "_"
        + combined["card3"].astype(str)
        + "_"
        + combined["card5"].astype(str)
        + "_"
        + combined["addr1"].astype(str)
    )

# Transaction Velocity & Sequential Dynamics (Lag and Lead differences per UID)
for uid_col in ["uid_c1_a1", "uid_c1_a1_d1", "uid_card_full"]:
    if uid_col in combined.columns:
        combined[f"{uid_col}_dt_diff_prev"] = (
            combined["TransactionDT"] - combined.groupby(uid_col)["TransactionDT"].shift(1)
        ).astype(np.float32)
        combined[f"{uid_col}_dt_diff_next"] = (
            combined.groupby(uid_col)["TransactionDT"].shift(-1) - combined["TransactionDT"]
        ).astype(np.float32)
        combined[f"{uid_col}_amt_diff_prev"] = (
            combined["TransactionAmt"] - combined.groupby(uid_col)["TransactionAmt"].shift(1)
        ).astype(np.float32)
        combined[f"{uid_col}_amt_diff_next"] = (
            combined.groupby(uid_col)["TransactionAmt"].shift(-1) - combined["TransactionAmt"]
        ).astype(np.float32)

# Temporal Velocity Window Features (counts in preceding 6h, 24h, 72h, 168h per UID)
velocity_uids = ["card1", "uid_c1_a1", "uid_c1_a1_d1", "uid_card_full"]
combined = compute_temporal_velocity(combined, uid_cols=velocity_uids, time_col="TransactionDT")

# Frequency Encoding
freq_cols = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "uid_c1_c2",
    "uid_c1_a1",
    "uid_c1_a1_pemail",
    "uid_c1_a1_d1",
    "uid_card_full",
]
for col in freq_cols:
    if col in combined.columns:
        freq = combined[col].value_counts(dropna=False)
        combined[f"{col}_fq_enc"] = combined[col].map(freq).astype(np.float32)

# Multi-statistic Group Aggregations strictly computed on Training split
train_subset = combined.iloc[:n_train]
group_pairs = [
    ("card1", "TransactionAmt"),
    ("uid_c1_a1", "TransactionAmt"),
    ("uid_c1_c2", "TransactionAmt"),
    ("uid_c1_a1_d1", "TransactionAmt"),
    ("uid_card_full", "TransactionAmt"),
    ("card1", "D1_norm"),
    ("card1", "D2_norm"),
    ("card1", "D4_norm"),
    ("card1", "D10_norm"),
    ("card1", "D15_norm"),
    ("uid_c1_a1", "D1_norm"),
    ("uid_c1_a1", "D2_norm"),
    ("uid_c1_a1", "D4_norm"),
    ("uid_c1_a1", "D10_norm"),
    ("uid_c1_a1", "D15_norm"),
    ("uid_card_full", "D1_norm"),
    ("uid_card_full", "D2_norm"),
    ("uid_card_full", "D4_norm"),
    ("uid_card_full", "D10_norm"),
    ("uid_card_full", "D15_norm"),
    ("uid_c1_a1_d1", "D2_norm"),
    ("uid_c1_a1_d1", "D4_norm"),
    ("uid_c1_a1_d1", "D10_norm"),
    ("uid_c1_a1_d1", "D15_norm"),
    ("card1", "C1"),
    ("card1", "C2"),
    ("card1", "C13"),
    ("card1", "C14"),
    ("uid_c1_a1", "C1"),
    ("uid_c1_a1", "C2"),
    ("uid_c1_a1", "C13"),
    ("uid_c1_a1", "C14"),
    ("uid_card_full", "C1"),
    ("uid_card_full", "C2"),
    ("uid_card_full", "C13"),
    ("uid_card_full", "C14"),
    ("uid_c1_a1_d1", "C1"),
    ("uid_c1_a1_d1", "C13"),
    ("uid_c1_a1_d1", "C14"),
]

for group_col, num_col in group_pairs:
    if group_col in combined.columns and num_col in combined.columns:
        grp = train_subset.groupby(group_col)[num_col]
        mean_s = grp.mean()
        std_s = grp.std()
        min_s = grp.min()
        max_s = grp.max()

        mean_val = combined[group_col].map(mean_s).astype(np.float32)
        std_val = combined[group_col].map(std_s).astype(np.float32)
        min_val = combined[group_col].map(min_s).astype(np.float32)
        max_val = combined[group_col].map(max_s).astype(np.float32)

        combined[f"{num_col}_mean_by_{group_col}"] = mean_val
        combined[f"{num_col}_std_by_{group_col}"] = std_val
        combined[f"{num_col}_min_by_{group_col}"] = min_val
        combined[f"{num_col}_max_by_{group_col}"] = max_val

        combined[f"{num_col}_diff_mean_by_{group_col}"] = (
            combined[num_col] - mean_val
        ).astype(np.float32)
        combined[f"{num_col}_ratio_mean_by_{group_col}"] = (
            combined[num_col] / (mean_val + 1e-5)
        ).astype(np.float32)
        combined[f"{num_col}_zscore_by_{group_col}"] = (
            (combined[num_col] - mean_val) / (std_val + 1e-5)
        ).astype(np.float32)

# Nunique group aggregations computed on train subset
nunique_pairs = [
    ("card1", "addr1"),
    ("uid_c1_a1", "dist1"),
    ("uid_c1_a1", "P_emaildomain"),
    ("uid_card_full", "dist1"),
    ("uid_card_full", "P_emaildomain"),
]
for group_col, cat_col in nunique_pairs:
    if group_col in combined.columns and cat_col in combined.columns:
        nu_map = train_subset.groupby(group_col)[cat_col].nunique()
        combined[f"{cat_col}_nunique_by_{group_col}"] = combined[group_col].map(nu_map).astype(np.float32)

# Encode Categorical Columns
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    combined[col] = combined[col].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined[col] = encoder.fit_transform(combined[[col]]).astype(np.int32)

# Memory optimization
combined = reduce_mem_usage(combined)

# Split back to Train and Test
X_train_full = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined.iloc[n_train:].copy().reset_index(drop=True)
del combined, train_subset
gc.collect()

# Create Strict Temporal Validation Split (80% Train, 20% Out-Of-Time Validation)
split_idx = int(0.80 * len(X_train_full))
X_train = X_train_full.iloc[:split_idx].copy().reset_index(drop=True)
y_train = y_train_full.iloc[:split_idx].copy().reset_index(drop=True)
X_val = X_train_full.iloc[split_idx:].copy().reset_index(drop=True)
y_val = y_train_full.iloc[split_idx:].copy().reset_index(drop=True)
del X_train_full
gc.collect()

print(f"Train split shape: {X_train.shape} | Fraud rate: {y_train.mean():.4f}")
print(f"Validation split shape: {X_val.shape} | Fraud rate: {y_val.mean():.4f}")
print(f"Test shape: {X_test.shape}")

# -------------------------------------------------------------
# 2. Model Training & Evaluation
# -------------------------------------------------------------
drop_cols = ["isFraud"]
feature_cols = [c for c in X_train.columns if c not in drop_cols]

X_tr = X_train[feature_cols]
y_tr = y_train

X_v = X_val[feature_cols]
y_v = y_val

X_te = X_test[feature_cols]

print(f"Training features count: {len(feature_cols)}")

# Build LightGBM Datasets
dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
dval = lgb.Dataset(X_v, label=y_v, reference=dtrain, free_raw_data=False)

# Model Configuration and Multi-Model Ensemble Training
ensemble_model = build_model(n_estimators=3000, early_stopping_rounds=50)

print("Initiating FraudEnsembleModel training with out-of-time validation...")
val_preds = ensemble_model.fit(dtrain=dtrain, dval=dval, X_v=X_v, y_v=y_v)

# Save ensemble model artifact
model_path = "./working/ensemble_fraud_model.joblib"
joblib.dump(ensemble_model, model_path)
print(f"Ensemble model saved successfully to {model_path}")

# Out-of-time Validation Evaluation
val_auc = compute_roc_auc_metric(y_v, val_preds)
print(f"Ensemble Validation ROC-AUC: {val_auc:.6f}")

# Model Inference on Test Set
test_preds = ensemble_model.predict(X_te)

# Validate prediction integrity
assert len(test_preds) == len(X_te), "Prediction count mismatch with test dataset."
assert not np.isnan(test_preds).any(), "NaN values encountered in test predictions."

# Create and format submission file
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=np.int64),
        "isFraud": test_preds.astype(np.float32),
    }
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")

# Final validation score reporting
print(f"Final Validation Score: {val_auc}")
