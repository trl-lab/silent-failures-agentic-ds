import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load raw datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Retrieve test ids
if "Id" in test_df.columns:
    test_ids = test_df["Id"].values
elif "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# 2. Target Label Encoding
target_col = "Category"
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(train_df[target_col])
num_classes = len(label_encoder.classes_)

# 3. Stratified Train / Validation Split
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.20, random_state=42, stratify=y_all
)

train_sub = train_df.iloc[train_idx].copy().reset_index(drop=True)
val_sub = train_df.iloc[val_idx].copy().reset_index(drop=True)
test_sub = test_df.copy().reset_index(drop=True)

y_train = y_all[train_idx]
y_val = y_all[val_idx]

# 4. Spatial Outlier Cleaning
valid_mask = (train_sub["Y"] < 80) & (train_sub["X"] < -121)
district_medians = (
    train_sub[valid_mask].groupby("PdDistrict")[["X", "Y"]].median().to_dict()
)
global_x_median = train_sub[valid_mask]["X"].median()
global_y_median = train_sub[valid_mask]["Y"].median()


def clean_coordinates(df):
    x_clean = df["X"].copy()
    y_clean = df["Y"].copy()
    outlier_mask = (df["Y"] >= 80) | (df["X"] >= -121)

    for idx in df[outlier_mask].index:
        district = df.loc[idx, "PdDistrict"]
        if district in district_medians["X"]:
            x_clean.loc[idx] = district_medians["X"][district]
            y_clean.loc[idx] = district_medians["Y"][district]
        else:
            x_clean.loc[idx] = global_x_median
            y_clean.loc[idx] = global_y_median

    df["X_clean"] = x_clean
    df["Y_clean"] = y_clean
    return df


train_sub = clean_coordinates(train_sub)
val_sub = clean_coordinates(val_sub)
test_sub = clean_coordinates(test_sub)

# 5. Frequency Encodings & Feature Engineering
from sklearn.model_selection import StratifiedKFold

address_counts = train_sub["Address"].value_counts().to_dict()
pddistrict_counts = train_sub["PdDistrict"].value_counts().to_dict()

train_hours = pd.to_datetime(train_sub["Dates"]).dt.hour
district_hour_str = train_sub["PdDistrict"].astype(str) + "_" + train_hours.astype(str)
district_hour_counts = district_hour_str.value_counts().to_dict()

address_hour_str = train_sub["Address"].astype(str) + "_" + train_hours.astype(str)
address_hour_counts = address_hour_str.value_counts().to_dict()

STREET_TYPES = [
    "ST",
    "AV",
    "WAY",
    "BLVD",
    "RD",
    "DR",
    "LN",
    "CT",
    "PL",
    "HWY",
    "EXPY",
    "TER",
    "ALLEY",
    "HY",
    "PKWY",
    "PZ",
    "TR",
]


def extract_street_type(address):
    address_upper = str(address).upper()
    for st_type in STREET_TYPES:
        if f" {st_type}" in address_upper or f"/{st_type}" in address_upper:
            return st_type
    return "OTHER"


def extract_street_name(address):
    address_str = str(address).upper()
    cleaned = re.sub(r"^\d+\s+BLOCK\s+OF\s+", "", address_str)
    return cleaned.strip()


def engineer_features(df):
    features = pd.DataFrame(index=df.index)

    # Temporal Features
    dates = pd.to_datetime(df["Dates"])
    features["year"] = dates.dt.year
    features["month"] = dates.dt.month
    features["day"] = dates.dt.day
    features["hour"] = dates.dt.hour
    features["minute"] = dates.dt.minute
    features["dayofweek"] = dates.dt.dayofweek
    features["dayofyear"] = dates.dt.dayofyear
    features["is_weekend"] = (features["dayofweek"] >= 5).astype(int)
    features["time_in_minutes"] = features["hour"] * 60 + features["minute"]
    features["trend"] = (
        features["year"]
        + (features["month"] - 1) / 12.0
        + (features["day"] - 1) / 365.0
    )

    # Cyclical Temporal Encodings
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24.0)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12.0)
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7.0)

    # Spatial Features & Spatial Grid Binning
    x = df["X_clean"].values
    y = df["Y_clean"].values
    features["X"] = x
    features["Y"] = y

    # Fine-grained Spatial Grid Bins
    features["X_grid"] = np.round(x, 3)
    features["Y_grid"] = np.round(y, 3)
    features["grid_id"] = (
        features["X_grid"].astype(str) + "_" + features["Y_grid"].astype(str)
    )

    # Rotated Coordinates
    for angle in [15, 30, 45, 60]:
        rad = np.radians(angle)
        features[f"rot_{angle}_x"] = x * np.cos(rad) - y * np.sin(rad)
        features[f"rot_{angle}_y"] = x * np.sin(rad) + y * np.cos(rad)

    # Landmarks & Center Distances
    center_x, center_y = -122.4194, 37.7749
    features["dist_center"] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    features["angle_center"] = np.arctan2(y - center_y, x - center_x)

    landmarks = {
        "tenderloin": (-122.4133, 37.7836),
        "financial": (-122.4014, 37.7946),
        "mission": (-122.4194, 37.7599),
        "sfpd_hq": (-122.4036, 37.7754),
    }
    lm_dists = []
    for lm_name, (lm_x, lm_y) in landmarks.items():
        d = np.sqrt((x - lm_x) ** 2 + (y - lm_y) ** 2)
        features[f"dist_{lm_name}"] = d
        lm_dists.append(d)

    # Landmark spatial interactions
    lm_dists_arr = np.column_stack(lm_dists)
    features["min_landmark_dist"] = np.min(lm_dists_arr, axis=1)
    features["mean_landmark_dist"] = np.mean(lm_dists_arr, axis=1)
    features["std_landmark_dist"] = np.std(lm_dists_arr, axis=1)
    features["dist_ratio_tenderloin_center"] = features["dist_tenderloin"] / (
        features["dist_center"] + 1e-5
    )
    features["dist_ratio_financial_center"] = features["dist_financial"] / (
        features["dist_center"] + 1e-5
    )
    features["x_dist_center"] = x * features["dist_center"]
    features["y_dist_center"] = y * features["dist_center"]

    # Address & Categorical Features
    address_str = df["Address"].astype(str)
    features["is_intersection"] = address_str.str.contains("/").astype(int)
    features["is_block"] = address_str.str.contains("Block", case=False).astype(int)

    features["address_freq"] = df["Address"].map(address_counts).fillna(1).astype(int)
    features["pddistrict_freq"] = (
        df["PdDistrict"].map(pddistrict_counts).fillna(1).astype(int)
    )

    # Spatio-Temporal Interaction Frequencies
    dh_key = df["PdDistrict"].astype(str) + "_" + features["hour"].astype(str)
    features["district_hour_freq"] = (
        dh_key.map(district_hour_counts).fillna(1).astype(int)
    )

    ah_key = df["Address"].astype(str) + "_" + features["hour"].astype(str)
    features["address_hour_freq"] = (
        ah_key.map(address_hour_counts).fillna(1).astype(int)
    )

    block_numbers = (
        address_str.str.extract(r"(\d+)\s+Block", expand=False).fillna(-1).astype(float)
    )
    features["block_number"] = block_numbers

    features["PdDistrict"] = df["PdDistrict"].astype("category")
    features["DayOfWeek"] = df["DayOfWeek"].astype("category")
    features["street_type"] = address_str.apply(extract_street_type).astype("category")

    # High-cardinality interaction features for Target Encoding
    features["street_name"] = address_str.apply(extract_street_name)
    features["Address_raw"] = address_str
    features["pd_hour"] = (
        df["PdDistrict"].astype(str) + "_" + features["hour"].astype(str)
    )

    return features


X_train = engineer_features(train_sub)
X_val = engineer_features(val_sub)
X_test = engineer_features(test_sub)

# Out-Of-Fold Target Probability Encoding with Additive Bayesian Smoothing (Batched)
te_cols = ["Address_raw", "street_name", "pd_hour", "grid_id"]
smoothing_weight = 10.0

global_class_priors = np.bincount(y_train, minlength=num_classes) / len(y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

te_train_dict = {}
te_val_dict = {}
te_test_dict = {}

for col in te_cols:
    train_te_feats = np.zeros((len(X_train), num_classes), dtype=np.float32)

    for tr_idx, fold_val_idx in skf.split(X_train, y_train):
        fold_tr_df = X_train.iloc[tr_idx]
        fold_tr_y = y_train[tr_idx]

        for k in range(num_classes):
            is_class_k = (fold_tr_y == k).astype(int)
            stats = pd.DataFrame(
                {"cat_val": fold_tr_df[col], "target": is_class_k}
            ).groupby("cat_val")["target"].agg(["count", "sum"])

            smoothed_prob = (
                stats["sum"] + smoothing_weight * global_class_priors[k]
            ) / (stats["count"] + smoothing_weight)
            mapping = smoothed_prob.to_dict()

            fold_val_cats = X_train.iloc[fold_val_idx][col]
            train_te_feats[fold_val_idx, k] = (
                fold_val_cats.map(mapping).fillna(global_class_priors[k]).values
            )

    for k in range(num_classes):
        te_train_dict[f"{col}_te_class_{k}"] = train_te_feats[:, k]

    # Target encodings for Validation and Test set using full training data
    for k in range(num_classes):
        is_class_k = (y_train == k).astype(int)
        stats = pd.DataFrame(
            {"cat_val": X_train[col], "target": is_class_k}
        ).groupby("cat_val")["target"].agg(["count", "sum"])

        smoothed_prob = (
            stats["sum"] + smoothing_weight * global_class_priors[k]
        ) / (stats["count"] + smoothing_weight)
        mapping = smoothed_prob.to_dict()

        te_val_dict[f"{col}_te_class_{k}"] = (
            X_val[col].map(mapping).fillna(global_class_priors[k]).values
        )
        te_test_dict[f"{col}_te_class_{k}"] = (
            X_test[col].map(mapping).fillna(global_class_priors[k]).values
        )

# Batch concatenate target encoding features to avoid DataFrame fragmentation
te_train_df = pd.DataFrame(te_train_dict, index=X_train.index)
te_val_df = pd.DataFrame(te_val_dict, index=X_val.index)
te_test_df = pd.DataFrame(te_test_dict, index=X_test.index)

X_train = pd.concat([X_train, te_train_df], axis=1)
X_val = pd.concat([X_val, te_val_df], axis=1)
X_test = pd.concat([X_test, te_test_df], axis=1)

# Drop raw high-cardinality temporary string columns after target encoding
X_train.drop(columns=["Address_raw", "street_name", "pd_hour", "grid_id"], inplace=True)
X_val.drop(columns=["Address_raw", "street_name", "pd_hour", "grid_id"], inplace=True)
X_test.drop(columns=["Address_raw", "street_name", "pd_hour", "grid_id"], inplace=True)


# 6. PyTorch Deep Neural Architecture Definition (Model Design Architecture)
class TabularEmbeddingNet(nn.Module):
    def __init__(
        self,
        num_cont_features,
        cat_cardinalities=None,
        num_classes=39,
        hidden_dims=(512, 256, 128),
        dropout_rate=0.25,
    ):
        super(TabularEmbeddingNet, self).__init__()

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        if cat_cardinalities is not None:
            for num_categories in cat_cardinalities:
                emb_dim = min(50, max(4, (num_categories + 1) // 2))
                self.embeddings.append(nn.Embedding(num_categories, emb_dim))
                total_emb_dim += emb_dim

        self.cont_bn = nn.BatchNorm1d(num_cont_features)
        in_dim = num_cont_features + total_emb_dim

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x_cont, x_cat=None):
        if x_cont.shape[0] > 1:
            x_cont_norm = self.cont_bn(x_cont)
        else:
            x_cont_norm = x_cont

        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = []
            for idx, emb in enumerate(self.embeddings):
                emb_outs.append(emb(x_cat[:, idx]))
            x_emb = torch.cat(emb_outs, dim=1)
            x_all = torch.cat([x_cont_norm, x_emb], dim=1)
        else:
            x_all = x_cont_norm

        features = self.backbone(x_all)
        logits = self.classifier(features)
        return logits


# 7. Tuned LightGBM + XGBoost Multi-Seed Ensemble Definition & Training
seeds = [42, 123]
val_preds_list = []
test_preds_list = []

for seed in seeds:
    # Tuned LightGBM (reduced leaf count to prevent target encoding overfitting)
    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_preds_list.append(lgb_model.predict_proba(X_val))
    test_preds_list.append(lgb_model.predict_proba(X_test))

    # XGBoost classifier for depth-wise complementary decision boundaries
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds_list.append(xgb_model.predict_proba(X_val))
    test_preds_list.append(xgb_model.predict_proba(X_test))

# 8. Model Inference & Ensemble Validation Evaluation
val_preds = np.mean(val_preds_list, axis=0)
val_score = log_loss(y_val, val_preds, labels=np.arange(num_classes))

test_preds = np.mean(test_preds_list, axis=0)
predicted_class_indices = np.argmax(test_preds, axis=1)
predicted_class_names = label_encoder.inverse_transform(predicted_class_indices)

# 9. Format & Save Submission File
sub_df = pd.DataFrame({"id": test_ids, "Category": predicted_class_names})
submission_path = "./submission/submission.csv"
sub_df.to_csv(submission_path, index=False)

# 10. Print Final Validation Metric
print(f"Final Validation Score: {val_score}")
