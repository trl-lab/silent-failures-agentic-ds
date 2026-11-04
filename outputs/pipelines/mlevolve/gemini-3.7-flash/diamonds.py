import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

# =========================================================================
# 0. Global Setup & Seed Configuration
# =========================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Data Ingestion & Physical Preprocessing
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet").reset_index(drop=True)
test_df = pd.read_parquet("./input/test.parquet").reset_index(drop=True)

CUT_MAP = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
COLOR_MAP = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
CLARITY_MAP = {
    "I1": 1,
    "SI2": 2,
    "SI1": 3,
    "VS2": 4,
    "VS1": 5,
    "VVS2": 6,
    "VVS1": 7,
    "IF": 8,
}


def clean_and_impute_dimensions(
    df: pd.DataFrame, ref_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Cleans anomalous zero or negative dimensions using physical relationships."""
    df = df.copy()
    if ref_df is None:
        ref_df = df

    for col in ["x", "y", "z"]:
        df[col] = df[col].apply(lambda v: np.nan if v <= 0 or v > 30 else v)

    # Impute missing z using theoretical depth percentage formula: depth = 200 * z / (x + y)
    mask_z_missing = (
        df["z"].isna() & df["x"].notna() & df["y"].notna() & df["depth"].notna()
    )
    df.loc[mask_z_missing, "z"] = (
        df.loc[mask_z_missing, "depth"]
        * (df.loc[mask_z_missing, "x"] + df.loc[mask_z_missing, "y"])
        / 200.0
    )

    # Compute reference ratios from clean subset
    ref_clean = ref_df[(ref_df["x"] > 0) & (ref_df["y"] > 0) & (ref_df["z"] > 0)].copy()
    ref_clean["vol_approx"] = ref_clean["x"] * ref_clean["y"] * ref_clean["z"]
    carat_vol_ratio = (ref_clean["carat"] / ref_clean["vol_approx"]).median()
    median_l_w_ratio = (ref_clean["x"] / ref_clean["y"]).median()

    for idx in df[df["x"].isna() | df["y"].isna() | df["z"].isna()].index:
        c = df.loc[idx, "carat"]
        est_vol = c / (carat_vol_ratio if carat_vol_ratio > 0 else 0.006)
        r = (est_vol / (median_l_w_ratio * 0.6)) ** (1.0 / 3.0)
        if pd.isna(df.loc[idx, "y"]):
            df.loc[idx, "y"] = max(r, 0.1)
        if pd.isna(df.loc[idx, "x"]):
            df.loc[idx, "x"] = max(r * median_l_w_ratio, 0.1)
        if pd.isna(df.loc[idx, "z"]):
            d = df.loc[idx, "depth"] if pd.notna(df.loc[idx, "depth"]) else 61.8
            df.loc[idx, "z"] = max(
                d * (df.loc[idx, "x"] + df.loc[idx, "y"]) / 200.0, 0.1
            )

    return df


train_df = clean_and_impute_dimensions(train_df, train_df)
test_df = clean_and_impute_dimensions(test_df, train_df)


# =========================================================================
# 2. Physics & Gemological Feature Engineering
# =========================================================================
def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics, gemological geometry, non-linear transformations, and interaction features."""
    df = df.copy()
    df["cut"] = df["cut"].astype(str)
    df["color"] = df["color"].astype(str)
    df["clarity"] = df["clarity"].astype(str)

    # Ordinal numerical mappings
    df["cut_num"] = df["cut"].map(CUT_MAP).astype(float)
    df["color_num"] = df["color"].map(COLOR_MAP).astype(float)
    df["clarity_num"] = df["clarity"].map(CLARITY_MAP).astype(float)

    # Composite Quality Scores
    df["quality_composite_add"] = (
        0.20 * df["cut_num"] + 0.35 * df["color_num"] + 0.45 * df["clarity_num"]
    )
    df["quality_composite_mul"] = df["cut_num"] * df["color_num"] * df["clarity_num"]
    df["color_clarity_idx"] = df["color_num"] * df["clarity_num"]
    df["cut_clarity_idx"] = df["cut_num"] * df["clarity_num"]
    df["cut_color_idx"] = df["cut_num"] * df["color_num"]

    # Non-linear carat transformations
    df["carat_cbrt"] = np.cbrt(df["carat"])
    df["carat_sqrt"] = np.sqrt(df["carat"])
    df["carat_sq"] = df["carat"] ** 2
    df["carat_cubed"] = df["carat"] ** 3
    df["log_carat"] = np.log1p(df["carat"])

    # 3D Physical Volumes & Approximations
    df["vol_box"] = df["x"] * df["y"] * df["z"]
    df["vol_ellipsoid"] = (np.pi / 6.0) * df["x"] * df["y"] * df["z"]

    r_girdle = (df["x"] + df["y"]) / 4.0
    r_table = r_girdle * (df["table"] / 100.0)
    df["vol_cone_approx"] = (np.pi * df["z"] / 3.0) * (
        r_girdle**2 + r_girdle * r_table + r_table**2
    )
    df["log_vol_box"] = np.log1p(np.clip(df["vol_box"], 0, None))

    # Surface Area and Density approximations
    df["surface_area"] = 2.0 * (
        df["x"] * df["y"] + df["y"] * df["z"] + df["z"] * df["x"]
    )
    df["carat_density"] = df["carat"] / (df["vol_box"] + 1e-4)
    df["carat_to_surface"] = df["carat"] / (df["surface_area"] + 1e-4)

    # Dimension ratios, symmetry, and proportions
    df["dim_mean"] = (df["x"] + df["y"] + df["z"]) / 3.0
    df["dim_std"] = np.std(df[["x", "y", "z"]].values, axis=1)
    df["dim_max"] = df[["x", "y", "z"]].max(axis=1)
    df["dim_min"] = df[["x", "y", "z"]].min(axis=1)
    df["dim_range"] = df["dim_max"] - df["dim_min"]

    df["aspect_ratio"] = df["x"] / (df["y"] + 1e-4)
    df["z_to_x_ratio"] = df["z"] / (df["x"] + 1e-4)
    df["z_to_y_ratio"] = df["z"] / (df["y"] + 1e-4)
    df["table_to_depth_ratio"] = df["table"] / (df["depth"] + 1e-4)

    # Optical proportion discrepancy
    df["depth_calc"] = 200.0 * df["z"] / (df["x"] + df["y"] + 1e-4)
    df["depth_discrepancy"] = np.abs(df["depth"] - df["depth_calc"])

    # Table physical dimensions
    df["table_width_mm"] = (df["table"] / 100.0) * ((df["x"] + df["y"]) / 2.0)
    df["table_area_approx"] = np.pi * ((df["table_width_mm"] / 2.0) ** 2)

    # Inter-attribute feature interactions
    df["carat_x_cut"] = df["carat"] * df["cut_num"]
    df["carat_x_color"] = df["carat"] * df["color_num"]
    df["carat_x_clarity"] = df["carat"] * df["clarity_num"]
    df["carat_x_quality_add"] = df["carat"] * df["quality_composite_add"]
    df["carat_x_quality_mul"] = df["carat"] * df["quality_composite_mul"]
    df["carat_sq_x_clarity"] = df["carat_sq"] * df["clarity_num"]
    df["carat_sq_x_color"] = df["carat_sq"] * df["color_num"]

    # String interactions
    df["cut_color"] = df["cut"].astype(str) + "_" + df["color"].astype(str)
    df["cut_clarity"] = df["cut"].astype(str) + "_" + df["clarity"].astype(str)
    df["color_clarity"] = df["color"].astype(str) + "_" + df["clarity"].astype(str)
    df["cut_color_clarity"] = (
        df["cut"].astype(str)
        + "_"
        + df["color"].astype(str)
        + "_"
        + df["clarity"].astype(str)
    )

    return df


train_fe = compute_domain_features(train_df)
test_fe = compute_domain_features(test_df)

cat_cols = [
    "cut",
    "color",
    "clarity",
    "cut_color",
    "cut_clarity",
    "color_clarity",
    "cut_color_clarity",
]
for col in cat_cols:
    freq_map = train_fe[col].astype(str).value_counts(normalize=True).to_dict()
    train_fe[f"{col}_freq"] = (
        train_fe[col].astype(str).map(freq_map).astype(float).fillna(0.0)
    )
    test_fe[f"{col}_freq"] = (
        test_fe[col].astype(str).map(freq_map).astype(float).fillna(0.0)
    )

# =========================================================================
# 3. Stratified 5-Fold Splitting & Leak-Free Fold Aggregations
# =========================================================================
N_SPLITS = 5
train_fe["price_bin"] = pd.qcut(
    train_fe["price"], q=20, labels=False, duplicates="drop"
)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_fe["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_fe, train_fe["price_bin"])):
    train_fe.loc[val_idx, "fold"] = fold
train_fe = train_fe.drop(columns=["price_bin"])

agg_keys = [
    "clarity",
    "color",
    "cut",
    "color_clarity",
    "cut_clarity",
    "cut_color_clarity",
]

for agg_key in agg_keys:
    train_fe[f"{agg_key}_mean_price_per_carat"] = np.nan
    test_fe[f"{agg_key}_mean_price_per_carat"] = np.nan
    train_fe[f"{agg_key}_carat_mean"] = np.nan
    train_fe[f"{agg_key}_carat_std"] = np.nan
    test_fe[f"{agg_key}_carat_mean"] = np.nan
    test_fe[f"{agg_key}_carat_std"] = np.nan

    for fold in range(N_SPLITS):
        fold_train = train_fe[train_fe["fold"] != fold].copy()
        fold_val_idx = train_fe[train_fe["fold"] == fold].index

        fold_train["price_per_carat"] = fold_train["price"] / (
            fold_train["carat"] + 1e-4
        )
        group_stats = (
            fold_train.groupby(agg_key)
            .agg(
                mean_ppc=("price_per_carat", "mean"),
                carat_mean=("carat", "mean"),
                carat_std=("carat", "std"),
            )
            .reset_index()
        )

        val_merged = train_fe.loc[fold_val_idx, [agg_key]].merge(
            group_stats, on=agg_key, how="left"
        )
        train_fe.loc[fold_val_idx, f"{agg_key}_mean_price_per_carat"] = val_merged[
            "mean_ppc"
        ].values
        train_fe.loc[fold_val_idx, f"{agg_key}_carat_mean"] = val_merged[
            "carat_mean"
        ].values
        train_fe.loc[fold_val_idx, f"{agg_key}_carat_std"] = (
            val_merged["carat_std"].fillna(0).values
        )

    # Full training stats for test set
    full_train = train_fe.copy()
    full_train["price_per_carat"] = full_train["price"] / (full_train["carat"] + 1e-4)
    full_group_stats = (
        full_train.groupby(agg_key)
        .agg(
            mean_ppc=("price_per_carat", "mean"),
            carat_mean=("carat", "mean"),
            carat_std=("carat", "std"),
        )
        .reset_index()
    )

    test_merged = test_fe[[agg_key]].merge(full_group_stats, on=agg_key, how="left")
    test_fe[f"{agg_key}_mean_price_per_carat"] = test_merged["mean_ppc"].values
    test_fe[f"{agg_key}_carat_mean"] = test_merged["carat_mean"].values
    test_fe[f"{agg_key}_carat_std"] = test_merged["carat_std"].fillna(0).values

for agg_key in agg_keys:
    for metric in ["mean_price_per_carat", "carat_mean", "carat_std"]:
        col_name = f"{agg_key}_{metric}"
        global_med = train_fe[col_name].median()
        train_fe[col_name] = train_fe[col_name].fillna(global_med)
        test_fe[col_name] = test_fe[col_name].fillna(global_med)

# Categorical column conversions with synchronized categories
for col in cat_cols:
    all_cats = pd.Series(
        np.unique(
            np.concatenate(
                [train_fe[col].dropna().unique(), test_fe[col].dropna().unique()]
            )
        )
    ).tolist()
    train_fe[col] = pd.Categorical(train_fe[col], categories=all_cats)
    test_fe[col] = pd.Categorical(test_fe[col], categories=all_cats)

num_cols = [c for c in train_fe.columns if c not in cat_cols + ["price", "fold"]]
all_features = cat_cols + num_cols

train_df_str = train_fe.copy()
test_df_str = test_fe.copy()
for col in cat_cols:
    train_df_str[col] = train_df_str[col].astype(str)
    test_df_str[col] = test_df_str[col].astype(str)


# =========================================================================
# 4. Neural Network Architecture: DiamondTabularAttentionResNet
# =========================================================================
class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.embedding_dim = embedding_dim
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )
        self.num_linear = nn.Linear(num_continuous, num_continuous * embedding_dim)
        self.num_bias = nn.Parameter(torch.zeros(num_continuous, embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_num.shape[0]
        cat_tokens = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_tokens.append(emb_layer(x_cat[:, i]).unsqueeze(1))

        if len(cat_tokens) > 0:
            cat_tokens = torch.cat(cat_tokens, dim=1)
        else:
            cat_tokens = torch.empty(
                batch_size, 0, self.embedding_dim, device=x_num.device
            )

        num_proj = self.num_linear(x_num).view(
            batch_size, self.num_continuous, self.embedding_dim
        )
        num_tokens = self.norm(num_proj + self.num_bias.unsqueeze(0))
        all_tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        return all_tokens, num_tokens


class TabularAttentionBlock(nn.Module):
    def __init__(self, emb_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.drop1(self.act1(self.norm1(self.fc1(x))))
        out = self.drop2(self.act2(self.norm2(self.fc2(out))))
        return out + residual


class DiamondTabularAttentionResNet(nn.Module):
    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: List[int],
        embedding_dim: int = 32,
        num_heads: int = 4,
        dense_hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.15,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            num_continuous=num_continuous,
            cat_cardinalities=cat_cardinalities,
            embedding_dim=embedding_dim,
        )
        self.attention_block = TabularAttentionBlock(
            emb_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        total_features = len(cat_cardinalities) + num_continuous
        flattened_dim = total_features * embedding_dim + num_continuous

        self.cont_norm = nn.BatchNorm1d(num_continuous)
        layers = []
        prev_dim = flattened_dim
        for hidden_dim in dense_hidden_dims:
            layers.append(ResidualDenseBlock(prev_dim, hidden_dim, dropout=dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.tokenizer(x_num, x_cat)
        attn_tokens = self.attention_block(tokens)
        flattened_tokens = attn_tokens.reshape(attn_tokens.shape[0], -1)
        normed_num = self.cont_norm(x_num)
        combined_repr = torch.cat([flattened_tokens, normed_num], dim=1)
        latent = self.backbone(combined_repr)
        return self.head(latent).squeeze(-1)


# =========================================================================
# 5. Tree Model Builders
# =========================================================================
def create_lgb_model(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=3500,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def create_xgb_model(seed: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=3500,
        random_state=seed,
        n_jobs=-1,
    )


def create_catboost_model(seed: int = 42) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=3.0,
        iterations=3500,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )


# =========================================================================
# 6. Multi-Model 5-Fold Cross Validation Training
# =========================================================================
models_list = ["lgb", "xgb", "cat", "nn"]
oof_preds = {m: np.zeros(len(train_fe)) for m in models_list}
test_preds = {m: np.zeros(len(test_fe)) for m in models_list}
y_true = train_fe["price"].values

print(f"Training multi-model ensemble across {N_SPLITS} folds on {device}...")

for fold in range(N_SPLITS):
    trn_mask = train_fe["fold"] != fold
    val_mask = train_fe["fold"] == fold

    y_trn = y_true[trn_mask]
    y_val = y_true[val_mask]

    # --- A. LightGBM ---
    X_trn_lgb = train_fe.loc[trn_mask, all_features]
    X_val_lgb = train_fe.loc[val_mask, all_features]
    X_tst_lgb = test_fe[all_features]

    lgb_model = create_lgb_model(seed=SEED + fold)
    lgb_model.fit(
        X_trn_lgb,
        y_trn,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )
    val_pred_lgb = lgb_model.predict(X_val_lgb)
    oof_preds["lgb"][val_mask] = val_pred_lgb
    test_preds["lgb"] += lgb_model.predict(X_tst_lgb) / N_SPLITS

    # --- B. XGBoost ---
    X_trn_xgb = train_fe.loc[trn_mask, all_features].copy()
    X_val_xgb = train_fe.loc[val_mask, all_features].copy()
    X_tst_xgb = test_fe[all_features].copy()

    xgb_model = create_xgb_model(seed=SEED + fold)
    xgb_model.set_params(early_stopping_rounds=60, enable_categorical=True)
    xgb_model.fit(X_trn_xgb, y_trn, eval_set=[(X_val_xgb, y_val)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_val_xgb)
    oof_preds["xgb"][val_mask] = val_pred_xgb
    test_preds["xgb"] += xgb_model.predict(X_tst_xgb) / N_SPLITS

    # --- C. CatBoost ---
    X_trn_cat = train_df_str.loc[trn_mask, all_features]
    X_val_cat = train_df_str.loc[val_mask, all_features]
    X_tst_cat = test_df_str[all_features]

    cat_model = create_catboost_model(seed=SEED + fold)
    cat_model.fit(
        X_trn_cat,
        y_trn,
        eval_set=(X_val_cat, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=60,
        verbose=False,
    )
    val_pred_cat = cat_model.predict(X_val_cat)
    oof_preds["cat"][val_mask] = val_pred_cat
    test_preds["cat"] += cat_model.predict(X_tst_cat) / N_SPLITS

    # --- D. PyTorch Tabular Attention ResNet ---
    encoded_cat_trn = np.zeros((len(y_trn), len(cat_cols)), dtype=np.int64)
    encoded_cat_val = np.zeros((len(y_val), len(cat_cols)), dtype=np.int64)
    encoded_cat_tst = np.zeros((len(test_fe), len(cat_cols)), dtype=np.int64)
    cat_cardinalities = []

    for i, col in enumerate(cat_cols):
        trn_vals = train_df_str.loc[trn_mask, col].values
        val_vals = train_df_str.loc[val_mask, col].values
        tst_vals = test_df_str[col].values

        unique_vals = np.unique(trn_vals)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        unknown_idx = len(unique_vals)

        encoded_cat_trn[:, i] = np.array(
            [mapping.get(v, unknown_idx) for v in trn_vals], dtype=np.int64
        )
        encoded_cat_val[:, i] = np.array(
            [mapping.get(v, unknown_idx) for v in val_vals], dtype=np.int64
        )
        encoded_cat_tst[:, i] = np.array(
            [mapping.get(v, unknown_idx) for v in tst_vals], dtype=np.int64
        )
        cat_cardinalities.append(len(unique_vals) + 1)

    scaler = StandardScaler()
    scaled_num_trn = scaler.fit_transform(
        train_fe.loc[trn_mask, num_cols].fillna(0).values
    )
    scaled_num_val = scaler.transform(train_fe.loc[val_mask, num_cols].fillna(0).values)
    scaled_num_tst = scaler.transform(test_fe[num_cols].fillna(0).values)

    t_trn_num = torch.tensor(scaled_num_trn, dtype=torch.float32)
    t_trn_cat = torch.tensor(encoded_cat_trn, dtype=torch.long)
    t_trn_y = torch.tensor(y_trn, dtype=torch.float32)

    t_val_num = torch.tensor(scaled_num_val, dtype=torch.float32)
    t_val_cat = torch.tensor(encoded_cat_val, dtype=torch.long)
    t_val_y = torch.tensor(y_val, dtype=torch.float32)

    t_tst_num = torch.tensor(scaled_num_tst, dtype=torch.float32)
    t_tst_cat = torch.tensor(encoded_cat_tst, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(t_trn_num, t_trn_cat, t_trn_y),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(t_val_num, t_val_cat, t_val_y),
        batch_size=512,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(t_tst_num, t_tst_cat),
        batch_size=512,
        shuffle=False,
    )

    nn_model = DiamondTabularAttentionResNet(
        num_continuous=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        embedding_dim=32,
        num_heads=4,
        dense_hidden_dims=[512, 256, 128],
        dropout=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    best_val_rmse = float("inf")
    best_nn_state = None

    for epoch in range(25):
        nn_model.train()
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num, batch_cat, batch_y = (
                batch_num.to(device),
                batch_cat.to(device),
                batch_y.to(device),
            )
            optimizer.zero_grad()
            pred = nn_model(batch_num, batch_cat)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_num, batch_cat, _ in val_loader:
                batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
                p = nn_model(batch_num, batch_cat)
                val_preds_list.extend(p.cpu().numpy().tolist())

        val_rmse_epoch = np.sqrt(mean_squared_error(y_val, val_preds_list))
        if val_rmse_epoch < best_val_rmse:
            best_val_rmse = val_rmse_epoch
            best_nn_state = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_state.items()})
    nn_model.eval()

    val_preds_final = []
    with torch.no_grad():
        for batch_num, batch_cat, _ in val_loader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            val_preds_final.extend(
                nn_model(batch_num, batch_cat).cpu().numpy().tolist()
            )

    tst_preds_final = []
    with torch.no_grad():
        for batch_num, batch_cat in test_loader:
            batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
            tst_preds_final.extend(
                nn_model(batch_num, batch_cat).cpu().numpy().tolist()
            )

    oof_preds["nn"][val_mask] = np.array(val_preds_final)
    test_preds["nn"] += np.array(tst_preds_final) / N_SPLITS

    print(
        f"Fold {fold + 1}/{N_SPLITS} Completed - LGB: {np.sqrt(mean_squared_error(y_val, val_pred_lgb)):.2f}, "
        f"XGB: {np.sqrt(mean_squared_error(y_val, val_pred_xgb)):.2f}, "
        f"CAT: {np.sqrt(mean_squared_error(y_val, val_pred_cat)):.2f}, "
        f"NN: {best_val_rmse:.2f}"
    )


# =========================================================================
# 7. Out-Of-Fold Evaluation & SLSQP Optimal Ensembling
# =========================================================================
oof_matrix = np.column_stack([oof_preds[m] for m in models_list])
test_matrix = np.column_stack([test_preds[m] for m in models_list])


def objective_fn(weights):
    w = np.array(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y_true, pred))


init_weights = np.ones(len(models_list)) / len(models_list)
bounds = [(0.0, 1.0) for _ in range(len(models_list))]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective_fn,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-9},
)

optimal_weights = opt_res.x / np.sum(opt_res.x)

blended_oof = np.dot(oof_matrix, optimal_weights)
min_train_price = float(np.min(y_true))
blended_oof = np.clip(blended_oof, min_train_price, None)
final_score = np.sqrt(mean_squared_error(y_true, blended_oof))

# =========================================================================
# 8. Test Submission Generation
# =========================================================================
final_test_pred = np.dot(test_matrix, optimal_weights)
final_test_pred = np.clip(final_test_pred, min_train_price, None)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_pred)),
        "price": final_test_pred,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
