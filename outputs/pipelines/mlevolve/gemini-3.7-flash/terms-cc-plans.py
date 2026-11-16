import json
import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ---------------------------------------------------------
# 1. Directory and Data Loading
# ---------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

TARGET_COL = "APR"
if TARGET_COL not in train_df.columns:
    possible_targets = [c for c in train_df.columns if c.upper() == "APR"]
    if possible_targets:
        TARGET_COL = possible_targets[0]
    else:
        raise ValueError("Target column APR not found in training data.")

y_train = train_df[TARGET_COL].copy().astype(np.float32)

if "id" in test_df.columns:
    test_ids = test_df["id"].copy().values
else:
    test_ids = np.arange(len(test_df), dtype=int)


# ---------------------------------------------------------
# 2. Data Cleaning, Range Extraction and Type Casting
# ---------------------------------------------------------
train_features = pd.DataFrame(index=train_df.index)
test_features = pd.DataFrame(index=test_df.index)

train_features["row_missing_count"] = train_df.isna().sum(axis=1).astype(np.float32)
test_features["row_missing_count"] = test_df.isna().sum(axis=1).astype(np.float32)

def parse_range_values(series: pd.Series):
    low_vals, high_vals, mid_vals, span_vals = [], [], [], []
    has_range_pattern = 0
    range_regex = re.compile(r"(\d+\.?\d*)\s*(?:%|\$)?\s*(?:-|to)\s*(\d+\.?\d*)\s*(?:%|\$)?", re.IGNORECASE)
    single_num_regex = re.compile(r"(\d+\.?\d*)")

    for val in series:
        if pd.isna(val):
            low_vals.append(np.nan)
            high_vals.append(np.nan)
            mid_vals.append(np.nan)
            span_vals.append(np.nan)
            continue
        s = str(val).strip()
        match = range_regex.search(s)
        if match:
            has_range_pattern += 1
            l_num, h_num = float(match.group(1)), float(match.group(2))
            low_vals.append(l_num)
            high_vals.append(h_num)
            mid_vals.append((l_num + h_num) / 2.0)
            span_vals.append(abs(h_num - l_num))
        else:
            s_match = single_num_regex.search(s)
            if s_match:
                num = float(s_match.group(1))
                low_vals.append(num)
                high_vals.append(num)
                mid_vals.append(num)
                span_vals.append(0.0)
            else:
                low_vals.append(np.nan)
                high_vals.append(np.nan)
                mid_vals.append(np.nan)
                span_vals.append(np.nan)

    return has_range_pattern > 0, pd.Series(low_vals, index=series.index), pd.Series(high_vals, index=series.index), pd.Series(mid_vals, index=series.index), pd.Series(span_vals, index=series.index)


def clean_boolean_series(series: pd.Series):
    bool_map = {
        "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
        "t": 1.0, "f": 0.0, "y": 1.0, "n": 0.0, "1": 1.0, "0": 0.0
    }
    s_lower = series.dropna().astype(str).str.strip().str.lower()
    if len(s_lower) > 0 and s_lower.isin(bool_map.keys()).mean() > 0.8:
        return series.astype(str).str.strip().str.lower().map(bool_map)
    return None


feature_cols = [c for c in train_df.columns if c != TARGET_COL]

for col in feature_cols:
    if train_df[col].dtype == object or str(train_df[col].dtype) == "string":
        bool_tr = clean_boolean_series(train_df[col])
        if bool_tr is not None:
            train_df[col] = bool_tr
            if col in test_df.columns:
                test_df[col] = clean_boolean_series(test_df[col])
            continue

        has_range, tr_low, tr_high, tr_mid, tr_span = parse_range_values(train_df[col])
        if has_range:
            _, te_low, te_high, te_mid, te_span = parse_range_values(test_df[col] if col in test_df else pd.Series([], dtype=object))
            train_features[f"{col}_range_low"] = tr_low.fillna(tr_low.median() if not pd.isna(tr_low.median()) else 0.0).astype(np.float32)
            train_features[f"{col}_range_high"] = tr_high.fillna(tr_high.median() if not pd.isna(tr_high.median()) else 0.0).astype(np.float32)
            train_features[f"{col}_range_mid"] = tr_mid.fillna(tr_mid.median() if not pd.isna(tr_mid.median()) else 0.0).astype(np.float32)
            train_features[f"{col}_range_span"] = tr_span.fillna(0.0).astype(np.float32)

            test_features[f"{col}_range_low"] = te_low.fillna(tr_low.median() if not pd.isna(tr_low.median()) else 0.0).astype(np.float32)
            test_features[f"{col}_range_high"] = te_high.fillna(tr_high.median() if not pd.isna(tr_high.median()) else 0.0).astype(np.float32)
            test_features[f"{col}_range_mid"] = te_mid.fillna(tr_mid.median() if not pd.isna(tr_mid.median()) else 0.0).astype(np.float32)
            test_features[f"{col}_range_span"] = te_span.fillna(0.0).astype(np.float32)

        cleaned_tr = (
            train_df[col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        converted = pd.to_numeric(cleaned_tr, errors="coerce")
        if converted.notna().sum() > 0.5 * len(train_df):
            train_df[col] = converted
            if col in test_df.columns:
                cleaned_te = (
                    test_df[col].astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                test_df[col] = pd.to_numeric(cleaned_te, errors="coerce")

numeric_cols = []
categorical_cols = []
text_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        numeric_cols.append(col)
    else:
        sample_str = train_df[col].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 40:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

# ---------------------------------------------------------
# 3. Numeric Transformations & Advanced Interactions
# ---------------------------------------------------------
for col in numeric_cols:
    med = train_df[col].median()
    if pd.isna(med):
        med = 0.0

    if train_df[col].isna().sum() > 0 or (col in test_df and test_df[col].isna().sum() > 0):
        train_features[f"{col}_isnan"] = train_df[col].isna().astype(np.int8)
        test_features[f"{col}_isnan"] = test_df[col].isna().astype(np.int8) if col in test_df else 1

    train_features[col] = train_df[col].fillna(med).astype(np.float32)
    test_features[col] = test_df[col].fillna(med).astype(np.float32) if col in test_df else med

    if (train_features[col] >= 0).all() and train_features[col].skew() > 1.2:
        train_features[f"{col}_log1p"] = np.log1p(train_features[col]).astype(np.float32)
        test_features[f"{col}_log1p"] = np.log1p(test_features[col].clip(lower=0)).astype(np.float32)

    # Quantile Monotonic Transformation
    qt = QuantileTransformer(n_quantiles=min(100, len(train_df)), output_distribution="normal", random_state=42)
    train_features[f"{col}_quantile"] = qt.fit_transform(train_features[[col]]).astype(np.float32).ravel()
    test_features[f"{col}_quantile"] = qt.transform(test_features[[col]]).astype(np.float32).ravel()

    # Power Transformation
    try:
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        train_features[f"{col}_yeojohnson"] = pt.fit_transform(train_features[[col]]).astype(np.float32).ravel()
        test_features[f"{col}_yeojohnson"] = pt.transform(test_features[[col]]).astype(np.float32).ravel()
    except Exception:
        pass

if len(numeric_cols) >= 2:
    train_features["num_row_mean"] = train_features[numeric_cols].mean(axis=1).astype(np.float32)
    train_features["num_row_std"] = train_features[numeric_cols].std(axis=1).fillna(0).astype(np.float32)
    train_features["num_row_min"] = train_features[numeric_cols].min(axis=1).astype(np.float32)
    train_features["num_row_max"] = train_features[numeric_cols].max(axis=1).astype(np.float32)

    test_features["num_row_mean"] = test_features[numeric_cols].mean(axis=1).astype(np.float32)
    test_features["num_row_std"] = test_features[numeric_cols].std(axis=1).fillna(0).astype(np.float32)
    test_features["num_row_min"] = test_features[numeric_cols].min(axis=1).astype(np.float32)
    test_features["num_row_max"] = test_features[numeric_cols].max(axis=1).astype(np.float32)

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col1, col2 = numeric_cols[i], numeric_cols[j]
        diff_tr = train_features[col1] - train_features[col2]
        diff_te = test_features[col1] - test_features[col2]

        train_features[f"diff_{col1}_{col2}"] = diff_tr.astype(np.float32)
        test_features[f"diff_{col1}_{col2}"] = diff_te.astype(np.float32)

        train_features[f"abs_diff_{col1}_{col2}"] = diff_tr.abs().astype(np.float32)
        test_features[f"abs_diff_{col1}_{col2}"] = diff_te.abs().astype(np.float32)

        train_features[f"ratio_{col1}_{col2}"] = (train_features[col1] / (train_features[col2].abs() + 1e-5)).astype(np.float32)
        test_features[f"ratio_{col1}_{col2}"] = (test_features[col1] / (test_features[col2].abs() + 1e-5)).astype(np.float32)

# ---------------------------------------------------------
# 4. Categorical Feature Engineering & Out-Of-Fold Encodings
# ---------------------------------------------------------
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = np.zeros(len(train_df), dtype=int)

for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
    folds[val_idx] = fold_idx

for col in categorical_cols:
    train_s = train_df[col].fillna("missing").astype(str)
    test_s = (
        test_df[col].fillna("missing").astype(str)
        if col in test_df
        else pd.Series(["missing"] * len(test_df))
    )

    freq_map = train_s.value_counts(normalize=True).to_dict()
    train_features[f"{col}_freq"] = train_s.map(freq_map).fillna(0).astype(np.float32)
    test_features[f"{col}_freq"] = test_s.map(freq_map).fillna(0).astype(np.float32)

    le = LabelEncoder()
    le.fit(list(train_s.unique()) + ["__unseen__"])
    train_features[f"{col}_label"] = le.transform(train_s).astype(np.float32)
    test_mapped = test_s.map(lambda s: s if s in le.classes_ else "__unseen__")
    test_features[f"{col}_label"] = le.transform(test_mapped).astype(np.float32)

    global_mean = float(y_train.mean())
    global_std = float(y_train.std())
    global_min = float(y_train.min())
    global_max = float(y_train.max())

    for smooth_weight in [5.0, 15.0, 30.0]:
        oof_target_enc = np.zeros(len(train_df), dtype=np.float32)
        for trn_idx, val_idx in kf.split(train_df):
            fold_train_s = train_s.iloc[trn_idx]
            fold_train_y = y_train.iloc[trn_idx]
            counts = fold_train_s.value_counts()
            means = fold_train_y.groupby(fold_train_s).mean()
            smoothed = (counts * means + smooth_weight * global_mean) / (
                counts + smooth_weight
            )
            oof_target_enc[val_idx] = (
                train_s.iloc[val_idx].map(smoothed).fillna(global_mean).values
            )

        full_counts = train_s.value_counts()
        full_means = y_train.groupby(train_s).mean()
        full_smoothed = (full_counts * full_means + smooth_weight * global_mean) / (
            full_counts + smooth_weight
        )

        train_features[f"{col}_target_enc_s{int(smooth_weight)}"] = oof_target_enc
        test_features[f"{col}_target_enc_s{int(smooth_weight)}"] = (
            test_s.map(full_smoothed).fillna(global_mean).astype(np.float32)
        )

    # Multi-statistic out-of-fold target encoding (std, min, max)
    oof_target_std = np.zeros(len(train_df), dtype=np.float32)
    oof_target_min = np.zeros(len(train_df), dtype=np.float32)
    oof_target_max = np.zeros(len(train_df), dtype=np.float32)

    for trn_idx, val_idx in kf.split(train_df):
        fold_train_s = train_s.iloc[trn_idx]
        fold_train_y = y_train.iloc[trn_idx]

        f_std = fold_train_y.groupby(fold_train_s).std().fillna(0)
        f_min = fold_train_y.groupby(fold_train_s).min()
        f_max = fold_train_y.groupby(fold_train_s).max()

        oof_target_std[val_idx] = train_s.iloc[val_idx].map(f_std).fillna(global_std).values
        oof_target_min[val_idx] = train_s.iloc[val_idx].map(f_min).fillna(global_min).values
        oof_target_max[val_idx] = train_s.iloc[val_idx].map(f_max).fillna(global_max).values

    full_std = y_train.groupby(train_s).std().fillna(0)
    full_min = y_train.groupby(train_s).min()
    full_max = y_train.groupby(train_s).max()

    train_features[f"{col}_target_std"] = oof_target_std
    test_features[f"{col}_target_std"] = test_s.map(full_std).fillna(global_std).astype(np.float32)

    train_features[f"{col}_target_min"] = oof_target_min
    test_features[f"{col}_target_min"] = test_s.map(full_min).fillna(global_min).astype(np.float32)

    train_features[f"{col}_target_max"] = oof_target_max
    test_features[f"{col}_target_max"] = test_s.map(full_max).fillna(global_max).astype(np.float32)

    if len(numeric_cols) > 0 and train_s.nunique() <= 50:
        for num_c in numeric_cols:
            grp_mean = train_df.groupby(train_s)[num_c].mean().to_dict()
            grp_std = train_df.groupby(train_s)[num_c].std().fillna(0).to_dict()
            grp_min = train_df.groupby(train_s)[num_c].min().to_dict()
            grp_max = train_df.groupby(train_s)[num_c].max().to_dict()
            grp_median = train_df.groupby(train_s)[num_c].median().to_dict()

            default_mean = float(train_features[num_c].mean())
            default_min = float(train_features[num_c].min())
            default_max = float(train_features[num_c].max())
            default_median = float(train_features[num_c].median())

            train_features[f"{col}_{num_c}_grp_mean"] = (
                train_s.map(grp_mean).fillna(default_mean).astype(np.float32)
            )
            train_features[f"{col}_{num_c}_grp_std"] = (
                train_s.map(grp_std).fillna(0).astype(np.float32)
            )
            train_features[f"{col}_{num_c}_grp_min"] = (
                train_s.map(grp_min).fillna(default_min).astype(np.float32)
            )
            train_features[f"{col}_{num_c}_grp_max"] = (
                train_s.map(grp_max).fillna(default_max).astype(np.float32)
            )
            train_features[f"{col}_{num_c}_grp_median"] = (
                train_s.map(grp_median).fillna(default_median).astype(np.float32)
            )

            test_features[f"{col}_{num_c}_grp_mean"] = (
                test_s.map(grp_mean).fillna(default_mean).astype(np.float32)
            )
            test_features[f"{col}_{num_c}_grp_std"] = (
                test_s.map(grp_std).fillna(0).astype(np.float32)
            )
            test_features[f"{col}_{num_c}_grp_min"] = (
                test_s.map(grp_min).fillna(default_min).astype(np.float32)
            )
            test_features[f"{col}_{num_c}_grp_max"] = (
                test_s.map(grp_max).fillna(default_max).astype(np.float32)
            )
            test_features[f"{col}_{num_c}_grp_median"] = (
                test_s.map(grp_median).fillna(default_median).astype(np.float32)
            )

top_cats = categorical_cols[:6]
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        c1, c2 = top_cats[i], top_cats[j]
        train_inter = (
            train_df[c1].fillna("missing").astype(str)
            + "___"
            + train_df[c2].fillna("missing").astype(str)
        )
        test_inter = (
            test_df[c1].fillna("missing").astype(str)
            + "___"
            + test_df[c2].fillna("missing").astype(str)
            if (c1 in test_df and c2 in test_df)
            else pd.Series(["missing___missing"] * len(test_df))
        )

        global_mean = float(y_train.mean())
        smooth_weight = 10.0
        oof_inter_target = np.zeros(len(train_df), dtype=np.float32)

        for trn_idx, val_idx in kf.split(train_df):
            fold_inter_s = train_inter.iloc[trn_idx]
            fold_y = y_train.iloc[trn_idx]
            counts = fold_inter_s.value_counts()
            means = fold_y.groupby(fold_inter_s).mean()
            smoothed = (counts * means + smooth_weight * global_mean) / (
                counts + smooth_weight
            )
            oof_inter_target[val_idx] = (
                train_inter.iloc[val_idx].map(smoothed).fillna(global_mean).values
            )

        full_counts = train_inter.value_counts()
        full_means = y_train.groupby(train_inter).mean()
        full_smoothed = (full_counts * full_means + smooth_weight * global_mean) / (
            full_counts + smooth_weight
        )

        train_features[f"inter_{c1}_{c2}_target_enc"] = oof_inter_target
        test_features[f"inter_{c1}_{c2}_target_enc"] = (
            test_inter.map(full_smoothed).fillna(global_mean).astype(np.float32)
        )

# ---------------------------------------------------------
# 5. Text Feature Engineering
# ---------------------------------------------------------
for t_col in text_cols:
    train_t = train_df[t_col].fillna("").astype(str)
    test_t = (
        test_df[t_col].fillna("").astype(str)
        if t_col in test_df
        else pd.Series([""] * len(test_df))
    )

    train_features[f"{t_col}_char_len"] = train_t.str.len().astype(np.float32)
    train_features[f"{t_col}_word_count"] = train_t.apply(
        lambda s: len(s.split())
    ).astype(np.float32)
    train_features[f"{t_col}_num_digits"] = train_t.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(np.float32)

    test_features[f"{t_col}_char_len"] = test_t.str.len().astype(np.float32)
    test_features[f"{t_col}_word_count"] = test_t.apply(
        lambda s: len(s.split())
    ).astype(np.float32)
    test_features[f"{t_col}_num_digits"] = test_t.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(np.float32)

    n_comp = min(5, max(1, len(train_df) - 1))
    tfidf = TfidfVectorizer(max_features=250, ngram_range=(1, 2), stop_words="english")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)

    train_tfidf = tfidf.fit_transform(train_t)
    train_svd = svd.fit_transform(train_tfidf)

    test_tfidf = tfidf.transform(test_t)
    test_svd = svd.transform(test_tfidf)

    for comp_idx in range(train_svd.shape[1]):
        train_features[f"{t_col}_svd_{comp_idx}"] = train_svd[:, comp_idx].astype(
            np.float32
        )
        test_features[f"{t_col}_svd_{comp_idx}"] = test_svd[:, comp_idx].astype(
            np.float32
        )

    oof_ridge = np.zeros(len(train_df), dtype=np.float32)
    test_ridge = np.zeros(len(test_df), dtype=np.float32)
    for trn_idx, val_idx in kf.split(train_df):
        r_model = Ridge(alpha=1.0)
        r_model.fit(train_tfidf[trn_idx], y_train.iloc[trn_idx])
        oof_ridge[val_idx] = r_model.predict(train_tfidf[val_idx])
        test_ridge += r_model.predict(test_tfidf) / n_splits
    train_features[f"{t_col}_ridge_pred"] = oof_ridge
    test_features[f"{t_col}_ridge_pred"] = test_ridge

engineered_feature_cols = [c for c in train_features.columns]

for col in engineered_feature_cols:
    if col not in test_features.columns:
        test_features[col] = 0.0

X = train_features[engineered_feature_cols].copy().astype(np.float32)
y = y_train.values.astype(np.float32)
test_X = test_features[engineered_feature_cols].copy().astype(np.float32)


# ---------------------------------------------------------
# 6. Model Architectures & Factories
# ---------------------------------------------------------
class TabularResNetBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dropout(self.act(self.norm1(self.linear1(x))))
        out = self.norm2(self.linear2(out))
        return self.act(out + residual)


class TabularResNet(nn.Module):
    def __init__(self, in_features: int, hidden_dims=[256, 128, 64], dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Mish(),
            nn.Dropout(dropout / 2.0),
        )
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                TabularResNetBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout=dropout
                )
            )
        self.res_blocks = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.head(x).squeeze(-1)


def train_pytorch_nn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    seed: int = 42,
    epochs: int = 120,
    batch_size: int = 64,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=0.0, neginf=0.0)

    t_X_tr = torch.tensor(X_tr_s, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
    t_X_val = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    t_X_te = torch.tensor(X_te_s, dtype=torch.float32).to(device)

    dataset = TensorDataset(t_X_tr, t_y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularResNet(
        in_features=X_tr.shape[1], hidden_dims=[256, 128, 64], dropout=0.2
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(t_X_val)
            val_loss = criterion(val_preds, t_y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    with torch.no_grad():
        oof_pred = model(t_X_val).cpu().numpy()
        test_pred = model(t_X_te).cpu().numpy()

    return oof_pred, test_pred


def get_lgb_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_lgb_dart_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="dart",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=25,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_samples=20,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_xgb_shallow_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=50,
    )


def get_hgb_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=1000,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=random_state,
    )


def get_ridge_model(random_state: int = 42) -> Ridge:
    return Ridge(alpha=10.0, random_state=random_state)


# ---------------------------------------------------------
# 7. Multi-Model 10-Fold Cross Validation Training
# ---------------------------------------------------------
oof_preds_lgb = np.zeros(len(train_df), dtype=np.float32)
oof_preds_lgb_dart = np.zeros(len(train_df), dtype=np.float32)
oof_preds_xgb = np.zeros(len(train_df), dtype=np.float32)
oof_preds_xgb_sh = np.zeros(len(train_df), dtype=np.float32)
oof_preds_cb = np.zeros(len(train_df), dtype=np.float32)
oof_preds_hgb = np.zeros(len(train_df), dtype=np.float32)
oof_preds_nn = np.zeros(len(train_df), dtype=np.float32)
oof_preds_ridge = np.zeros(len(train_df), dtype=np.float32)

test_preds_lgb = np.zeros(len(test_df), dtype=np.float32)
test_preds_lgb_dart = np.zeros(len(test_df), dtype=np.float32)
test_preds_xgb = np.zeros(len(test_df), dtype=np.float32)
test_preds_xgb_sh = np.zeros(len(test_df), dtype=np.float32)
test_preds_cb = np.zeros(len(test_df), dtype=np.float32)
test_preds_hgb = np.zeros(len(test_df), dtype=np.float32)
test_preds_nn = np.zeros(len(test_df), dtype=np.float32)
test_preds_ridge = np.zeros(len(test_df), dtype=np.float32)

for fold in range(n_splits):
    trn_mask = folds != fold
    val_mask = folds == fold

    X_train_fold, y_train_fold = X.iloc[trn_mask], y[trn_mask]
    X_val_fold, y_val_fold = X.iloc[val_mask], y[val_mask]

    # --- A. LightGBM Standard ---
    lgb_model = get_lgb_model(random_state=42 + fold)
    lgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_p_lgb = lgb_model.predict(X_val_fold)
    oof_preds_lgb[val_mask] = val_p_lgb
    test_preds_lgb += lgb_model.predict(test_X) / n_splits

    # --- B. LightGBM DART ---
    lgb_dart = get_lgb_dart_model(random_state=42 + fold)
    lgb_dart.fit(X_train_fold, y_train_fold)
    val_p_lgb_dart = lgb_dart.predict(X_val_fold)
    oof_preds_lgb_dart[val_mask] = val_p_lgb_dart
    test_preds_lgb_dart += lgb_dart.predict(test_X) / n_splits

    # --- C. XGBoost Standard ---
    xgb_model = get_xgb_model(random_state=42 + fold)
    xgb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_p_xgb = xgb_model.predict(X_val_fold)
    oof_preds_xgb[val_mask] = val_p_xgb
    test_preds_xgb += xgb_model.predict(test_X) / n_splits

    # --- D. XGBoost Shallow ---
    xgb_sh = get_xgb_shallow_model(random_state=42 + fold)
    xgb_sh.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_p_xgb_sh = xgb_sh.predict(X_val_fold)
    oof_preds_xgb_sh[val_mask] = val_p_xgb_sh
    test_preds_xgb_sh += xgb_sh.predict(test_X) / n_splits

    # --- E. CatBoost ---
    cb_model = get_catboost_model(random_state=42 + fold)
    cb_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_p_cb = cb_model.predict(X_val_fold)
    oof_preds_cb[val_mask] = val_p_cb
    test_preds_cb += cb_model.predict(test_X) / n_splits

    # --- F. HistGradientBoosting ---
    hgb_model = get_hgb_model(random_state=42 + fold)
    hgb_model.fit(X_train_fold, y_train_fold)
    val_p_hgb = hgb_model.predict(X_val_fold)
    oof_preds_hgb[val_mask] = val_p_hgb
    test_preds_hgb += hgb_model.predict(test_X) / n_splits

    # --- G. PyTorch Tabular ResNet ---
    oof_p_nn, test_p_nn = train_pytorch_nn(
        X_train_fold.values,
        y_train_fold,
        X_val_fold.values,
        y_val_fold,
        test_X.values,
        seed=42 + fold,
        epochs=120,
        batch_size=64,
    )
    oof_preds_nn[val_mask] = oof_p_nn
    test_preds_nn += test_p_nn / n_splits

    # --- H. Scaled Ridge Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    test_scaled = scaler.transform(test_X)

    ridge_model = get_ridge_model(random_state=42 + fold)
    ridge_model.fit(X_train_scaled, y_train_fold)
    val_p_ridge = ridge_model.predict(X_val_scaled)
    oof_preds_ridge[val_mask] = val_p_ridge
    test_preds_ridge += ridge_model.predict(test_scaled) / n_splits

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val_fold, val_p_lgb))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val_fold, val_p_xgb))
    fold_cb_rmse = np.sqrt(mean_squared_error(y_val_fold, val_p_cb))
    fold_nn_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_p_nn))
    print(
        f"Fold {fold+1}/{n_splits} Validation RMSE -> LGB: {fold_lgb_rmse:.4f} | XGB: {fold_xgb_rmse:.4f} | CB: {fold_cb_rmse:.4f} | NN: {fold_nn_rmse:.4f}"
    )

# ---------------------------------------------------------
# 8. Out-of-Fold Ensemble Optimization & Submission
# ---------------------------------------------------------
oof_matrix = np.column_stack([
    oof_preds_lgb,
    oof_preds_lgb_dart,
    oof_preds_xgb,
    oof_preds_xgb_sh,
    oof_preds_cb,
    oof_preds_hgb,
    oof_preds_nn,
    oof_preds_ridge,
])
test_matrix = np.column_stack([
    test_preds_lgb,
    test_preds_lgb_dart,
    test_preds_xgb,
    test_preds_xgb_sh,
    test_preds_cb,
    test_preds_hgb,
    test_preds_nn,
    test_preds_ridge,
])


def objective(weights):
    w = np.array(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y, pred))


n_models = oof_matrix.shape[1]
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

best_loss = float("inf")
best_weights = np.ones(n_models) / n_models

for seed_offset in range(15):
    np.random.seed(42 + seed_offset)
    init_w = np.random.uniform(0.1, 1.0, size=n_models)
    init_w = init_w / np.sum(init_w)
    res = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if res.success and res.fun < best_loss:
        best_loss = res.fun
        best_weights = res.x

best_weights = np.maximum(0, best_weights)
best_weights = best_weights / np.sum(best_weights)

oof_ensemble = np.dot(oof_matrix, best_weights)
final_score = np.sqrt(mean_squared_error(y, oof_ensemble))

final_test_preds = np.dot(test_matrix, best_weights)
final_test_preds = np.clip(final_test_preds, a_min=0.0, a_max=None)

submission_df = pd.DataFrame({"id": test_ids, "APR": final_test_preds})
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Ensemble Model Weights: {best_weights}")
print(f"Final Validation Score: {final_score}")