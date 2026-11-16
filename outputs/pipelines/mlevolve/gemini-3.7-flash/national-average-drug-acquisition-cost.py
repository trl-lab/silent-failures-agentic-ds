import json
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for strict reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official competition evaluation metric: Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_ndc_components(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hierarchical manufacturer, product, and package codes from NDC numbers."""
    features = pd.DataFrame(index=df.index)
    ndc_col = next(
        (
            c
            for c in df.columns
            if "ndc" in c.lower() and "desc" not in c.lower() and "desc" not in c
        ),
        None,
    )

    if ndc_col and ndc_col in df.columns:
        ndc_str = (
            df[ndc_col].astype(str).str.replace(r"[^0-9]", "", regex=True).str.zfill(11)
        )
        features["ndc_labeler"] = ndc_str.str[:5]
        features["ndc_product"] = ndc_str.str[5:9]
        features["ndc_package"] = ndc_str.str[9:11]
        features["ndc_product_9"] = ndc_str.str[:9]
        features["ndc_labeler_num"] = pd.to_numeric(
            features["ndc_labeler"], errors="coerce"
        ).fillna(0)
        features["ndc_product_num"] = pd.to_numeric(
            features["ndc_product"], errors="coerce"
        ).fillna(0)
        features["ndc_package_num"] = pd.to_numeric(
            features["ndc_package"], errors="coerce"
        ).fillna(0)
        features["ndc_product_9_num"] = pd.to_numeric(
            features["ndc_product_9"], errors="coerce"
        ).fillna(0)
    return features


def extract_dosage_and_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract dosage values, units of measure, and drug delivery forms from text fields."""
    features = pd.DataFrame(index=df.index)

    desc_col = next(
        (c for c in df.columns if "desc" in c.lower() or "drug" in c.lower()), None
    )
    generic_col = next((c for c in df.columns if "generic" in c.lower()), None)

    combined_text = pd.Series("", index=df.index)
    if desc_col and desc_col in df.columns:
        desc_text = df[desc_col].fillna("").astype(str).str.upper()
        combined_text = combined_text + " " + desc_text
        features["desc_len"] = desc_text.str.len()
        features["desc_word_count"] = desc_text.str.split().str.len()
        features["desc_digit_count"] = desc_text.str.count(r"\d")
        features["desc_has_slash"] = desc_text.str.contains(r"/", regex=True).astype(
            int
        )
        features["desc_has_hyphen"] = desc_text.str.contains(r"-", regex=True).astype(
            int
        )

    if generic_col and generic_col in df.columns:
        gen_text = df[generic_col].fillna("").astype(str).str.upper()
        combined_text = combined_text + " " + gen_text
        features["gen_len"] = gen_text.str.len()
        features["gen_word_count"] = gen_text.str.split().str.len()
        if desc_col and desc_col in df.columns:
            features["desc_equals_generic"] = (
                df[desc_col].astype(str).str.upper() == gen_text
            ).astype(int)

    # Strength extraction and standardized unit normalization
    strength_pattern = r"(\d+(?:\.\d+)?)\s*(MG|MCG|G|GM|ML|%|MEQ|UNIT|UNITS|IU|MCG/ML|MG/ML|MG/GM|MCG/ACT|ACT|HR|HOUR)"
    extracted = combined_text.str.extract(strength_pattern)
    features["strength_num"] = pd.to_numeric(extracted[0], errors="coerce").fillna(0.0)
    features["strength_num_log"] = np.log1p(features["strength_num"])
    features["strength_unit"] = extracted[1].fillna("NONE").astype(str)

    # Normalized metric strength (standardizing g, mcg, %, mg to mg base)
    normalized_mg = features["strength_num"].copy()
    mcg_mask = features["strength_unit"] == "MCG"
    g_mask = features["strength_unit"].isin(["G", "GM"])
    pct_mask = features["strength_unit"] == "%"
    normalized_mg.loc[mcg_mask] = normalized_mg.loc[mcg_mask] / 1000.0
    normalized_mg.loc[g_mask] = normalized_mg.loc[g_mask] * 1000.0
    normalized_mg.loc[pct_mask] = normalized_mg.loc[pct_mask] * 10.0
    features["strength_normalized_mg"] = normalized_mg
    features["strength_normalized_mg_log"] = np.log1p(np.maximum(0, normalized_mg))

    # Combination drug secondary strength
    second_strength_pattern = r"(\d+(?:\.\d+)?)\s*[-/]\s*(\d+(?:\.\d+)?)"
    extracted_combo = combined_text.str.extract(second_strength_pattern)
    features["is_combo_drug"] = extracted_combo[0].notnull().astype(int)
    features["strength_num_secondary"] = pd.to_numeric(
        extracted_combo[1], errors="coerce"
    ).fillna(0.0)

    # Pharmacological active stem / suffix extraction
    pharm_stems = {
        "stem_mab": r"\w+MAB\b",
        "stem_nib": r"\w+NIB\b",
        "stem_statin": r"\w+STATIN\b",
        "stem_cillin": r"\w+CILLIN\b",
        "stem_olol": r"\w+OLOL\b",
        "stem_sartan": r"\w+SARTAN\b",
        "stem_prazole": r"\w+PRAZOLE\b",
        "stem_vir": r"\w+VIR\b",
        "stem_glutide": r"\w+GLUTIDE\b",
        "stem_pril": r"\w+PRIL\b",
        "stem_dipine": r"\w+DIPINE\b",
        "stem_afil": r"\w+AFIL\b",
        "stem_gliptin": r"\w+GLIPTIN\b",
        "stem_gliflozin": r"\w+GLIFLOZIN\b",
        "stem_dronate": r"\w+DRONATE\b",
        "stem_coxib": r"\w+COXIB\b",
        "stem_asone": r"\w+ASONE\b",
        "stem_olone": r"\w+OLONE\b",
    }
    for stem_name, pattern in pharm_stems.items():
        features[stem_name] = combined_text.str.contains(pattern, regex=True).astype(int)

    # Pharmaceutical formulations and delivery modes
    forms = {
        "form_tablet": r"\b(?:TAB|TABLET|TABS|CAPLET)\b",
        "form_capsule": r"\b(?:CAP|CAPSULE|CAPS)\b",
        "form_injection": r"\b(?:INJ|INJECTION|VIAL|AMP|AMPULE|SYRINGE)\b",
        "form_solution": r"\b(?:SOL|SOLUTION|SOLN|ORAL SOL)\b",
        "form_suspension": r"\b(?:SUSP|SUSPENSION)\b",
        "form_cream": r"\b(?:CREAM|CRM)\b",
        "form_ointment": r"\b(?:OINT|OINTMENT)\b",
        "form_gel": r"\b(?:GEL|JELLY)\b",
        "form_patch": r"\b(?:PATCH|TRANSDERMAL)\b",
        "form_drops": r"\b(?:DROPS|OPHTH|OTIC)\b",
        "form_spray": r"\b(?:SPRAY|AEROSOL|INHALER|INHALATION|AER)\b",
        "form_extended_release": r"\b(?:ER|XR|CR|SR|XL|SA|TR|EXTENDED|RELEASE|DELAYED|DR)\b",
        "form_chewable": r"\b(?:CHEW|CHEWABLE)\b",
        "form_topical": r"\b(?:TOPICAL|TOP)\b",
        "form_oral": r"\b(?:ORAL)\b",
        "form_suppository": r"\b(?:SUPP|SUPPOSITORY)\b",
        "form_powder": r"\b(?:POWDER|POWD)\b",
    }

    for feat_name, pattern in forms.items():
        features[feat_name] = combined_text.str.contains(pattern, regex=True).astype(
            int
        )

    # Chemical salt indicators and specialized release modifiers
    salts_and_release = {
        "salt_hcl": r"\b(?:HCL|HYDROCHLORIDE)\b",
        "salt_sodium": r"\b(?:SODIUM|SOD)\b",
        "salt_potassium": r"\b(?:POTASSIUM|POT)\b",
        "salt_tartrate": r"\b(?:TARTRATE)\b",
        "salt_acetate": r"\b(?:ACETATE)\b",
        "salt_succinate": r"\b(?:SUCCINATE)\b",
        "salt_phosphate": r"\b(?:PHOSPHATE|PHOSP)\b",
        "salt_sulfate": r"\b(?:SULFATE|SULPHATE|SO4)\b",
        "salt_citrate": r"\b(?:CITRATE)\b",
        "salt_maleate": r"\b(?:MALEATE)\b",
        "salt_fumarate": r"\b(?:FUMARATE)\b",
        "salt_mesylate": r"\b(?:MESYLATE)\b",
        "salt_hydrobromide": r"\b(?:HYDROBROMIDE|HBR)\b",
        "form_odt": r"\b(?:ODT|DISPERSIBLE|ORALLY DISINTEGRATING)\b",
        "form_ir": r"\b(?:IR|IMMEDIATE RELEASE)\b",
        "form_er": r"\b(?:ER)\b",
        "form_cr": r"\b(?:CR)\b",
        "form_xr": r"\b(?:XR)\b",
    }

    for salt_name, pattern in salts_and_release.items():
        features[salt_name] = combined_text.str.contains(
            pattern, regex=True
        ).astype(int)

    # Strength-to-unit interactions
    unit_col = next(
        (
            c
            for c in df.columns
            if "unit" in c.lower()
            and "nadac" not in c.lower()
            and "price" not in c.lower()
        ),
        None,
    )
    if unit_col and unit_col in df.columns:
        unit_str = df[unit_col].astype(str).str.upper()
        features["unit_is_ea"] = unit_str.str.contains(
            r"\bEA\b", regex=True
        ).astype(int)
        features["unit_is_ml"] = unit_str.str.contains(
            r"\bML\b", regex=True
        ).astype(int)
        features["unit_is_gm"] = unit_str.str.contains(
            r"\b(?:GM|G)\b", regex=True
        ).astype(int)
        features["strength_times_is_ea"] = (
            features["strength_num"] * features["unit_is_ea"]
        )
        features["strength_times_is_ml"] = (
            features["strength_num"] * features["unit_is_ml"]
        )
        features["strength_times_is_gm"] = (
            features["strength_num"] * features["unit_is_gm"]
        )

    return features


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar, cyclic, elapsed time, and pairwise date delta features."""
    features = pd.DataFrame(index=df.index)
    date_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ["date", "time", "as_of", "effective"])
    ]

    parsed_dates_dict = {}
    for c in date_cols:
        parsed_dates = pd.to_datetime(df[c], errors="coerce")
        if parsed_dates.notnull().sum() > 0:
            parsed_dates_dict[c] = parsed_dates
            prefix = re.sub(r"[^a-zA-Z0-9]", "_", c.lower()).strip("_")
            features[f"{prefix}_year"] = parsed_dates.dt.year.fillna(-1).astype(int)
            features[f"{prefix}_month"] = parsed_dates.dt.month.fillna(-1).astype(int)
            features[f"{prefix}_day"] = parsed_dates.dt.day.fillna(-1).astype(int)
            features[f"{prefix}_dayofweek"] = parsed_dates.dt.dayofweek.fillna(
                -1
            ).astype(int)
            features[f"{prefix}_dayofyear"] = parsed_dates.dt.dayofyear.fillna(
                -1
            ).astype(int)
            features[f"{prefix}_quarter"] = parsed_dates.dt.quarter.fillna(-1).astype(
                int
            )
            features[f"{prefix}_is_month_start"] = (
                parsed_dates.dt.is_month_start.fillna(False).astype(int)
            )
            features[f"{prefix}_is_month_end"] = parsed_dates.dt.is_month_end.fillna(
                False
            ).astype(int)

            min_date = parsed_dates.min()
            features[f"{prefix}_days_elapsed"] = (
                (parsed_dates - min_date).dt.days.fillna(-1).astype(int)
            )

            month_norm = 2 * np.pi * parsed_dates.dt.month.fillna(1) / 12.0
            features[f"{prefix}_month_sin"] = np.sin(month_norm)
            features[f"{prefix}_month_cos"] = np.cos(month_norm)

    parsed_col_names = list(parsed_dates_dict.keys())
    for i in range(len(parsed_col_names)):
        for j in range(i + 1, len(parsed_col_names)):
            c1, c2 = parsed_col_names[i], parsed_col_names[j]
            p1 = re.sub(r"[^a-zA-Z0-9]", "_", c1.lower()).strip("_")
            p2 = re.sub(r"[^a-zA-Z0-9]", "_", c2.lower()).strip("_")
            diff_days = (
                (parsed_dates_dict[c1] - parsed_dates_dict[c2]).dt.days.fillna(0.0)
            )
            features[f"date_diff_{p1}_{p2}_days"] = diff_days.astype(np.float32)

    return features


def extract_tfidf_svd_and_ridge_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    folds: list,
    n_components: int = 32,
    seed: int = 42,
) -> tuple:
    """Extract TruncatedSVD components and Out-of-Fold Ridge regression price predictions on TF-IDF text features."""
    desc_col_train = next(
        (c for c in train_df.columns if "desc" in c.lower() or "drug" in c.lower()),
        None,
    )
    gen_col_train = next(
        (c for c in train_df.columns if "generic" in c.lower()), None
    )

    desc_col_test = next(
        (c for c in test_df.columns if "desc" in c.lower() or "drug" in c.lower()),
        None,
    )
    gen_col_test = next((c for c in test_df.columns if "generic" in c.lower()), None)

    train_text = (
        (
            train_df[desc_col_train].fillna("").astype(str)
            if desc_col_train
            else pd.Series("", index=train_df.index)
        )
        + " "
        + (
            train_df[gen_col_train].fillna("").astype(str)
            if gen_col_train
            else pd.Series("", index=train_df.index)
        )
    )

    test_text = (
        (
            test_df[desc_col_test].fillna("").astype(str)
            if desc_col_test
            else pd.Series("", index=test_df.index)
        )
        + " "
        + (
            test_df[gen_col_test].fillna("").astype(str)
            if gen_col_test
            else pd.Series("", index=test_df.index)
        )
    )

    # Word & Char n-gram TF-IDF
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=5000,
        sublinear_tf=True,
    )

    train_w = word_tfidf.fit_transform(train_text)
    test_w = word_tfidf.transform(test_text)
    train_c = char_tfidf.fit_transform(train_text)
    test_c = char_tfidf.transform(test_text)

    train_tfidf = hstack([train_w, train_c]).tocsr()
    test_tfidf = hstack([test_w, test_c]).tocsr()

    # Truncated SVD components
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_svd_arr = svd.fit_transform(train_tfidf)
    test_svd_arr = svd.transform(test_tfidf)

    cols = [f"svd_text_{i}" for i in range(n_components)]
    train_feat_df = pd.DataFrame(
        train_svd_arr, columns=cols, index=train_df.index, dtype=np.float32
    )
    test_feat_df = pd.DataFrame(
        test_svd_arr, columns=cols, index=test_df.index, dtype=np.float32
    )

    # Supervised Out-of-Fold Ridge Text Regression
    train_ridge_oof = np.zeros(len(train_df), dtype=np.float32)
    test_ridge_accum = np.zeros(len(test_df), dtype=np.float32)
    y_train = train_df[target_col].values

    for train_idx, val_idx in folds:
        ridge_model = Ridge(alpha=5.0, random_state=seed)
        ridge_model.fit(train_tfidf[train_idx], y_train[train_idx])
        train_ridge_oof[val_idx] = ridge_model.predict(train_tfidf[val_idx])
        test_ridge_accum += ridge_model.predict(test_tfidf) / len(folds)

    train_feat_df["oof_text_ridge_pred"] = train_ridge_oof
    test_feat_df["oof_text_ridge_pred"] = test_ridge_accum

    return train_feat_df, test_feat_df


class ResidualBlock(nn.Module):
    """Residual MLP block with dense layer, LayerNorm, SiLU activation, and dropout."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act(self.ln1(self.fc1(x))))
        out = self.ln2(self.fc2(out))
        return self.act(out + residual)


class ResidualTabularNN(nn.Module):
    """Deep Residual Tabular Neural Network for continuous representation learning."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def train_residual_tabular_nn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 256,
) -> tuple:
    """Train Residual Tabular PyTorch NN with AdamW and Cosine Annealing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_x_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    test_x_t = torch.tensor(X_test_sc, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = ResidualTabularNN(in_features=X_tr.shape[1], hidden_dim=256, num_blocks=3).to(
        device
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_rmse = float("inf")
    best_val_preds = np.zeros(len(y_val), dtype=np.float32)
    best_test_preds = np.zeros(len(X_test), dtype=np.float32)

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_x_t).cpu().numpy()
            val_rmse = compute_rmse(y_val, val_out)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_preds = val_out
                best_test_preds = model(test_x_t).cpu().numpy()

    return np.clip(best_val_preds, 0.0, None), np.clip(best_test_preds, 0.0, None)


def compute_out_of_fold_target_encodings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list,
    target_col: str,
    folds: list,
    smoothing: float = 10.0,
) -> tuple:
    """Compute leakage-free out-of-fold smoothed multi-statistic target encodings."""
    train_te = pd.DataFrame(index=train_df.index)
    test_te = pd.DataFrame(index=test_df.index)

    global_mean = float(train_df[target_col].mean())
    global_std = float(train_df[target_col].std(ddof=0))
    global_min = float(train_df[target_col].min())
    global_max = float(train_df[target_col].max())

    for col in cat_cols:
        if col not in train_df.columns:
            continue

        train_te[f"{col}_te_mean"] = 0.0
        train_te[f"{col}_te_std"] = 0.0
        train_te[f"{col}_te_min"] = 0.0
        train_te[f"{col}_te_max"] = 0.0

        for train_idx, val_idx in folds:
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            stats = fold_train.groupby(col)[target_col].agg(
                ["count", "mean", "std", "min", "max"]
            )
            smoothed_mean = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            smoothed_std = stats["std"].fillna(global_std)
            smoothed_min = stats["min"].fillna(global_min)
            smoothed_max = stats["max"].fillna(global_max)

            train_te.loc[val_idx, f"{col}_te_mean"] = (
                fold_val[col].map(smoothed_mean).fillna(global_mean).values
            )
            train_te.loc[val_idx, f"{col}_te_std"] = (
                fold_val[col].map(smoothed_std).fillna(global_std).values
            )
            train_te.loc[val_idx, f"{col}_te_min"] = (
                fold_val[col].map(smoothed_min).fillna(global_min).values
            )
            train_te.loc[val_idx, f"{col}_te_max"] = (
                fold_val[col].map(smoothed_max).fillna(global_max).values
            )

        full_stats = train_df.groupby(col)[target_col].agg(
            ["count", "mean", "std", "min", "max"]
        )
        full_smoothed_mean = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        test_te[f"{col}_te_mean"] = (
            test_df[col].map(full_smoothed_mean).fillna(global_mean).values
        )
        test_te[f"{col}_te_std"] = (
            test_df[col]
            .map(full_stats["std"].fillna(global_std))
            .fillna(global_std)
            .values
        )
        test_te[f"{col}_te_min"] = (
            test_df[col]
            .map(full_stats["min"].fillna(global_min))
            .fillna(global_min)
            .values
        )
        test_te[f"{col}_te_max"] = (
            test_df[col]
            .map(full_stats["max"].fillna(global_max))
            .fillna(global_max)
            .values
        )

    return train_te, test_te


def main():
    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    train_df = pd.read_parquet(train_path).reset_index(drop=True)
    test_df = pd.read_parquet(test_path).reset_index(drop=True)

    target_col = next(
        (c for c in train_df.columns if "nadac" in c.lower() or "target" in c.lower()),
        "NADAC Per Unit",
    )
    y_train = train_df[target_col].values.astype(np.float32)

    # 2. Extract Subcomponent Domain Features
    train_ndc = parse_ndc_components(train_df)
    test_ndc = parse_ndc_components(test_df)

    train_text = extract_dosage_and_form_features(train_df)
    test_text = extract_dosage_and_form_features(test_df)

    train_temporal = extract_temporal_features(train_df)
    test_temporal = extract_temporal_features(test_df)

    train_svd, test_svd = extract_tfidf_svd_and_ridge_features(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        folds=list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train_df)),
        n_components=32,
        seed=SEED,
    )

    combined_train = pd.concat(
        [train_df, train_ndc, train_text, train_temporal, train_svd], axis=1
    )
    combined_test = pd.concat(
        [test_df, test_ndc, test_text, test_temporal, test_svd], axis=1
    )

    # 3. Categorical Interaction & Frequency Encoding
    class_col = next((c for c in combined_train.columns if "class" in c.lower()), None)
    unit_col = next(
        (
            c
            for c in combined_train.columns
            if "unit" in c.lower()
            and "nadac" not in c.lower()
            and "price" not in c.lower()
        ),
        None,
    )

    cat_candidates = [
        c
        for c in combined_train.columns
        if (
            combined_train[c].dtype == "object"
            or combined_train[c].dtype.name == "category"
        )
        and c != target_col
    ]

    if class_col and unit_col:
        combined_train["class_x_unit"] = (
            combined_train[class_col].astype(str)
            + "_"
            + combined_train[unit_col].astype(str)
        )
        combined_test["class_x_unit"] = (
            combined_test[class_col].astype(str)
            + "_"
            + combined_test[unit_col].astype(str)
        )
        cat_candidates.append("class_x_unit")

    if "ndc_labeler" in combined_train.columns and unit_col:
        combined_train["labeler_x_unit"] = (
            combined_train["ndc_labeler"].astype(str)
            + "_"
            + combined_train[unit_col].astype(str)
        )
        combined_test["labeler_x_unit"] = (
            combined_test["ndc_labeler"].astype(str)
            + "_"
            + combined_test[unit_col].astype(str)
        )
        cat_candidates.append("labeler_x_unit")

    if "ndc_product_9" in combined_train.columns and unit_col:
        combined_train["ndc9_x_unit"] = (
            combined_train["ndc_product_9"].astype(str)
            + "_"
            + combined_train[unit_col].astype(str)
        )
        combined_test["ndc9_x_unit"] = (
            combined_test["ndc_product_9"].astype(str)
            + "_"
            + combined_test[unit_col].astype(str)
        )
        cat_candidates.append("ndc9_x_unit")

    for col in cat_candidates:
        if col in combined_train.columns:
            freq = combined_train[col].value_counts(normalize=True).to_dict()
            combined_train[f"{col}_freq"] = (
                combined_train[col].map(freq).fillna(0).astype(np.float32)
            )
            combined_test[f"{col}_freq"] = (
                combined_test[col].map(freq).fillna(0).astype(np.float32)
            )

            le = LabelEncoder()
            train_vals = combined_train[col].astype(str)
            test_vals = combined_test[col].astype(str)
            le.fit(
                list(train_vals.unique()) + list(test_vals.unique()) + ["__UNSEEN__"]
            )
            combined_train[f"{col}_encoded"] = le.transform(train_vals)
            combined_test[f"{col}_encoded"] = le.transform(test_vals)

    # 4. K-Fold Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(combined_train))

    fold_assignments = np.zeros(len(combined_train), dtype=int)
    for fold_num, (_, val_idx) in enumerate(fold_indices):
        fold_assignments[val_idx] = fold_num
    combined_train["fold"] = fold_assignments

    # 5. Out-of-fold Target Encoding
    target_enc_cols = [
        c
        for c in [
            "ndc_product_9",
            class_col,
            unit_col,
            "class_x_unit",
            "strength_unit",
            "ndc_labeler",
            "labeler_x_unit",
            "ndc9_x_unit",
        ]
        if c and c in combined_train.columns
    ]
    train_te, test_te = compute_out_of_fold_target_encodings(
        train_df=combined_train,
        test_df=combined_test,
        cat_cols=target_enc_cols,
        target_col=target_col,
        folds=fold_indices,
    )

    combined_train = pd.concat([combined_train, train_te], axis=1)
    combined_test = pd.concat([combined_test, test_te], axis=1)

    # 6. Feature Selection and Imputation
    exclude_cols = (
        {target_col, "id", "fold"}
        | set(cat_candidates)
        | {
            c
            for c in combined_train.columns
            if any(k in c.lower() for k in ["date", "time", "as_of", "effective"])
            and combined_train[c].dtype == "object"
        }
    )

    feature_cols = [
        c
        for c in combined_train.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(combined_train[c])
    ]

    for c in feature_cols:
        col_median = combined_train[c].replace([np.inf, -np.inf], np.nan).median()
        if pd.isna(col_median):
            col_median = 0.0
        combined_train[c] = (
            combined_train[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(col_median)
            .astype(np.float32)
        )
        combined_test[c] = (
            combined_test[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(col_median)
            .astype(np.float32)
        )

    X = combined_train[feature_cols].values
    y = combined_train[target_col].values.astype(np.float32)
    X_test = combined_test[feature_cols].values
    folds = combined_train["fold"].values
    n_splits = len(np.unique(folds))

    # 7. Multi-Model Training & Out-of-fold Inference (LightGBM, XGBoost, Residual Tabular NN)
    model_names = ["lgb", "xgb", "nn"]
    oof_preds = {
        name: np.zeros(len(combined_train), dtype=np.float32) for name in model_names
    }
    test_preds = {
        name: np.zeros(len(combined_test), dtype=np.float32) for name in model_names
    }

    for fold_idx in range(n_splits):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # 1) LightGBM Regressor with extra_trees randomized splits
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            extra_trees=True,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED + fold_idx,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_val = np.clip(lgb_model.predict(X_val), 0.0, None)
        oof_preds["lgb"][val_mask] = lgb_val
        test_preds["lgb"] += np.clip(lgb_model.predict(X_test), 0.0, None) / n_splits

        # 2) XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=SEED + fold_idx,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        xgb_val = np.clip(xgb_model.predict(X_val), 0.0, None)
        oof_preds["xgb"][val_mask] = xgb_val
        test_preds["xgb"] += np.clip(xgb_model.predict(X_test), 0.0, None) / n_splits

        # 3) Residual Tabular Neural Network
        nn_val, nn_test = train_residual_tabular_nn(
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            seed=SEED + fold_idx,
            epochs=45,
            batch_size=256,
        )
        oof_preds["nn"][val_mask] = nn_val
        test_preds["nn"] += nn_test / n_splits

        print(
            f"Fold {fold_idx + 1}/{n_splits} - "
            f"LGB: {compute_rmse(y_val, lgb_val):.5f} | "
            f"XGB: {compute_rmse(y_val, xgb_val):.5f} | "
            f"NN: {compute_rmse(y_val, nn_val):.5f}"
        )

    for name in model_names:
        print(f"OOF RMSE ({name}): {compute_rmse(y, oof_preds[name]):.5f}")

    # Optimize Ensemble Blend Weights
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
    test_matrix = np.column_stack([test_preds[name] for name in model_names])

    def blend_loss(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = np.sum(oof_matrix * w, axis=1)
        return compute_rmse(y, pred)

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1)] * len(model_names)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(
        blend_loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = res.x / np.sum(res.x)
    print(f"Optimal Blend Weights: {dict(zip(model_names, best_weights))}")

    oof_ensemble = np.clip(np.sum(oof_matrix * best_weights, axis=1), 0.0, None)
    final_test_predictions = np.clip(
        np.sum(test_matrix * best_weights, axis=1), 0.0, None
    )

    final_oof_rmse = compute_rmse(y, oof_ensemble)

    # 8. Create Submission File
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame(
        {"id": np.arange(len(combined_test)), "NADAC Per Unit": final_test_predictions}
    )
    submission_df.to_csv(submission_path, index=False)

    # 9. Output Final Validation Score
    print(f"Final Validation Score: {final_oof_rmse}")


if __name__ == "__main__":
    main()