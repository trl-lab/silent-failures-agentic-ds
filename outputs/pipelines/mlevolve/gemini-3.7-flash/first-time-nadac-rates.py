import json
import os
import random
import re
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack as sparse_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Create required output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ==============================================================================
# 1. Feature Engineering & Preprocessing Helper Functions
# ==============================================================================
def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map columns to standardized stripped naming."""
    col_map = {col: col.strip() for col in df.columns}
    return df.rename(columns=col_map)


def find_column(df: pd.DataFrame, candidates: list):
    """Find the first matching column name from candidates (fuzzy normalized)."""
    def normalize_str(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    norm_to_orig = {normalize_str(c): c for c in df.columns}
    for cand in candidates:
        norm_cand = normalize_str(cand)
        if norm_cand in norm_to_orig:
            return norm_to_orig[norm_cand]
    # Substring fallback
    for cand in candidates:
        norm_cand = normalize_str(cand)
        for k, orig in norm_to_orig.items():
            if norm_cand in k or k in norm_cand:
                return orig
    return None


def extract_ndc_components(series: pd.Series) -> pd.DataFrame:
    """Extract standard 11-digit NDC segments (labeler, product, package)."""
    clean_ndc = series.astype(str).str.replace(r"[^0-9]", "", regex=True)
    clean_ndc = clean_ndc.str.zfill(11)

    labeler = clean_ndc.str.slice(0, 5)
    product = clean_ndc.str.slice(5, 9)
    package = clean_ndc.str.slice(9, 11)

    labeler_num = pd.to_numeric(labeler, errors="coerce").fillna(-1).astype(np.int32)
    product_num = pd.to_numeric(product, errors="coerce").fillna(-1).astype(np.int32)
    package_num = pd.to_numeric(package, errors="coerce").fillna(-1).astype(np.int32)

    return pd.DataFrame(
        {
            "ndc_labeler": labeler_num,
            "ndc_product": product_num,
            "ndc_package": package_num,
        },
        index=series.index,
    )


def set_torch_seed(seed: int = 42):
    """Ensure exact reproducibility across PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_hierarchical_target_aggregations(
    train_series: pd.Series,
    y_train_series: pd.Series,
    val_series: pd.Series,
    test_series: pd.Series,
    smoothing: float = 10.0,
):
    """Compute Bayesian smoothed mean, standard deviation, and count statistics without leakage."""
    global_mean = float(y_train_series.mean())
    global_std = float(y_train_series.std()) if len(y_train_series) > 1 else 0.0

    grouped = y_train_series.groupby(train_series)
    counts = grouped.count()
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)

    # Bayesian smoothed mean
    smoothed_means = (
        (counts * means + smoothing * global_mean) / (counts + smoothing)
    ).to_dict()

    # Bayesian smoothed standard deviation
    smoothed_stds = (
        np.sqrt((counts * (stds ** 2) + smoothing * (global_std ** 2)) / (counts + smoothing))
    ).to_dict()

    # Log counts
    log_counts = np.log1p(counts).to_dict()

    def map_and_fill(s: pd.Series):
        s_mean = s.map(smoothed_means).fillna(global_mean).astype(np.float32)
        s_std = s.map(smoothed_stds).fillna(global_std).astype(np.float32)
        s_cnt = s.map(log_counts).fillna(0.0).astype(np.float32)
        return np.column_stack([s_mean.values, s_std.values, s_cnt.values])

    tr_arr = map_and_fill(train_series)
    va_arr = map_and_fill(val_series)
    te_arr = map_and_fill(test_series)

    return tr_arr, va_arr, te_arr


def extract_text_domain_features(descriptions: pd.Series) -> pd.DataFrame:
    """Extract dosage, concentration ratios, packaging, administration routes, and chemical tokens."""
    s = descriptions.fillna("").astype(str).str.upper()
    features = pd.DataFrame(index=descriptions.index)

    # Text length & complexity statistics
    features["desc_len"] = s.str.len().astype(np.float32)
    features["desc_word_count"] = s.str.split().apply(len).astype(np.float32)
    features["desc_num_digits"] = s.apply(lambda x: sum(c.isdigit() for c in x)).astype(np.float32)
    features["desc_has_slash"] = s.str.contains("/", regex=False).astype(np.float32)
    features["desc_has_dash"] = s.str.contains("-", regex=False).astype(np.float32)
    features["desc_has_plus"] = s.str.contains(r"\+|\bAND\b|\bW/", regex=True).astype(np.float32)
    features["desc_has_pct"] = s.str.contains("%", regex=False).astype(np.float32)

    # Extended Concentration & Multi-part strength parsing
    def parse_advanced_strength(text):
        mg_ml_match = re.search(r"(\d+\.?\d*)\s*MG\s*/\s*(\d+\.?\d*)\s*ML", text)
        mg_ml_ratio = (
            float(mg_ml_match.group(1)) / (float(mg_ml_match.group(2)) + 1e-5)
            if mg_ml_match
            else 0.0
        )

        mcg_act_match = re.search(r"(\d+\.?\d*)\s*MCG\s*/\s*(?:ACTUATION|ACT|SPRAY|INH)", text)
        mcg_per_act = float(mcg_act_match.group(1)) if mcg_act_match else 0.0

        pct_match = re.search(r"(\d+\.?\d*)\s*%", text)
        pct_conc = float(pct_match.group(1)) if pct_match else 0.0

        # Package quantity / volume match
        qty_match = re.search(r"\b(\d+)\s*(?:TAB|TABLET|CAP|CAPSULE|VIAL|BOTTLE|AMPULE|SYRINGE|PACK|COUNT|CT|EA|BLISTER|FILM|PATCH)S?\b", text)
        pkg_qty = float(qty_match.group(1)) if qty_match else 0.0

        vol_match = re.search(r"(\d+\.?\d*)\s*(?:ML|LITER|L|GM|G|OZ|FL\s*OZ)\b", text)
        pkg_vol = float(vol_match.group(1)) if vol_match else 0.0

        # General numerical strengths
        matches = re.findall(
            r"(\d+\.?\d*)\s*(?:MG|MCG|GM|G|ML|MEQ|%|UNITS|UNIT|IU|ACTUATION|MCG/ML|MG/ML)",
            text,
        )
        if matches:
            nums = [float(m) for m in matches if float(m) > 0]
            if nums:
                primary = nums[0]
                secondary = nums[1] if len(nums) > 1 else 0.0
                total = sum(nums)
                return primary, secondary, total, len(nums), mg_ml_ratio, mcg_per_act, pct_conc, pkg_qty, pkg_vol

        num_matches = re.findall(r"\b\d+\.?\d*\b", text)
        if num_matches:
            nums = [float(m) for m in num_matches if float(m) > 0]
            if nums:
                return nums[0], 0.0, nums[0], 1, mg_ml_ratio, mcg_per_act, pct_conc, pkg_qty, pkg_vol

        return 0.0, 0.0, 0.0, 0, mg_ml_ratio, mcg_per_act, pct_conc, pkg_qty, pkg_vol

    strength_results = [parse_advanced_strength(t) for t in s]
    features["primary_strength"] = np.array([r[0] for r in strength_results], dtype=np.float32)
    features["secondary_strength"] = np.array([r[1] for r in strength_results], dtype=np.float32)
    features["total_strength"] = np.array([r[2] for r in strength_results], dtype=np.float32)
    features["log_primary_strength"] = np.log1p(features["primary_strength"]).astype(np.float32)
    features["log_total_strength"] = np.log1p(features["total_strength"]).astype(np.float32)
    features["num_strength_values"] = np.array([r[3] for r in strength_results], dtype=np.float32)
    features["mg_per_ml_ratio"] = np.array([r[4] for r in strength_results], dtype=np.float32)
    features["mcg_per_actuation"] = np.array([r[5] for r in strength_results], dtype=np.float32)
    features["pct_concentration"] = np.array([r[6] for r in strength_results], dtype=np.float32)
    features["extracted_pkg_qty"] = np.array([r[7] for r in strength_results], dtype=np.float32)
    features["extracted_pkg_vol"] = np.array([r[8] for r in strength_results], dtype=np.float32)

    features["strength_ratio"] = np.where(
        features["secondary_strength"] > 0,
        features["primary_strength"] / (features["secondary_strength"] + 1e-5),
        1.0,
    ).astype(np.float32)

    # Formulation & Administration Route Flags
    route_keywords = {
        "route_oral": r"\b(?:ORAL|PO|TAB|TABLET|CAP|CAPSULE|SOLN|SOLUTION|SYRUP|ELIXIR|CHEW|SUSP)\b",
        "route_topical": r"\b(?:TOPICAL|TOP|CREAM|CRM|OINT|OINTMENT|GEL|LOTION|SHAMPOO|FOAM|PATCH|TRANSDERMAL)\b",
        "route_inj": r"\b(?:INJ|INJECTION|IV|INTRAVENOUS|IM|INTRAMUSCULAR|SC|SUBCUTANEOUS|VIAL|SYRINGE|AMP|AUTO-INJECTOR)\b",
        "route_ophth": r"\b(?:OPHTH|OPHTHALMIC|EYE|DROPS)\b",
        "route_otic": r"\b(?:OTIC|EAR)\b",
        "route_nasal": r"\b(?:NASAL|NOSE|SPRAY)\b",
        "route_inh": r"\b(?:INH|INHALER|INHALATION|AEROSOL|NEBULIZER|NEB|RESPIRATORY)\b",
        "route_rectal": r"\b(?:RECTAL|SUPP|SUPPOSITORY|ENEMA)\b",
        "route_sublingual": r"\b(?:SUBLINGUAL|SL|BUCCAL)\b",
        "form_er": r"\b(?:ER|XR|SR|DR|EXTENDED RELEASE|DELAYED RELEASE|CR|CONTINUOUS RELEASE|24HR|12HR)\b",
        "form_generic_indicator": r"\b(?:GENERIC|CORRESPONDING)\b",
        "form_powder": r"\b(?:POWDER|POWD|RECONSTITUTED|PACKET)\b",
    }
    for col_name, pattern in route_keywords.items():
        features[col_name] = s.str.contains(pattern, regex=True).astype(np.float32)

    # Strength unit flags
    unit_keywords = {
        "unit_mg": r"\bMG\b|\bMG/",
        "unit_mcg": r"\bMCG\b|\bMCG/",
        "unit_gm": r"\bGM\b|\bG\b|\bGRAM\b",
        "unit_ml": r"\bML\b",
        "unit_pct": r"%",
        "unit_meq": r"\bMEQ\b",
        "unit_units": r"\bUNITS\b|\bUNIT\b|\bIU\b",
    }
    for col_name, pattern in unit_keywords.items():
        features[col_name] = s.str.contains(pattern, regex=True).astype(np.float32)

    # Active ingredient chemical root tokens
    def get_stem_tokens(text):
        tokens = re.findall(r"[A-Z]+", text)
        first = tokens[0] if len(tokens) > 0 else "UNKNOWN"
        second = tokens[1] if len(tokens) > 1 else ""
        two_tok = f"{first}_{second}" if second else first
        return first, two_tok

    stem_data = [get_stem_tokens(t) for t in s]
    features["drug_first_word"] = [item[0] for item in stem_data]
    features["drug_stem_pair"] = [item[1] for item in stem_data]
    features["first_word_len"] = features["drug_first_word"].apply(len).astype(np.float32)

    return features


class TabularResNet(nn.Module):
    """PyTorch Tabular ResNet architecture with skip connections, batch normalization, and dropout."""
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h + self.act(self.block1(h))
        h = h + self.act(self.block2(h))
        out = self.head(h)
        return out.squeeze(-1)


def train_pytorch_resnet(
    X_tr_sc: np.ndarray,
    y_tr: np.ndarray,
    X_va_sc: np.ndarray,
    y_va: np.ndarray,
    X_te_sc: np.ndarray,
    seed: int = 42,
    epochs: int = 110,
    lr: float = 3e-3,
):
    """Train PyTorch Tabular ResNet model using AdamW and Cosine Annealing."""
    set_torch_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_features = X_tr_sc.shape[1]
    model = TabularResNet(in_features=in_features, hidden_dim=128, dropout_rate=0.2).to(device)

    dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_va_preds = np.zeros(len(y_va), dtype=np.float32)
    best_te_preds = np.zeros(len(X_te_sc), dtype=np.float32)

    X_va_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_pred = np.maximum(0.0, model(X_va_t).cpu().numpy())
            val_rmse = np.sqrt(mean_squared_error(y_va, va_pred))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_va_preds = va_pred
                best_te_preds = np.maximum(0.0, model(X_te_t).cpu().numpy())

    return best_va_preds, best_te_preds


def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract cyclical and hierarchical temporal components."""
    if date_col is None or date_col not in df.columns:
        return pd.DataFrame(index=df.index)

    dt = pd.to_datetime(df[date_col], errors="coerce")
    ref_date = pd.Timestamp("2020-01-01")

    feats = pd.DataFrame(index=df.index)
    feats[f"{date_col}_year"] = dt.dt.year.fillna(2020).astype(np.int32)
    feats[f"{date_col}_month"] = dt.dt.month.fillna(1).astype(np.int32)
    feats[f"{date_col}_day"] = dt.dt.day.fillna(1).astype(np.int32)
    feats[f"{date_col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(np.int32)
    feats[f"{date_col}_dayofyear"] = dt.dt.dayofyear.fillna(1).astype(np.int32)
    feats[f"{date_col}_quarter"] = dt.dt.quarter.fillna(1).astype(np.int32)
    feats[f"{date_col}_is_month_end"] = dt.dt.is_month_end.fillna(False).astype(np.int8)
    feats[f"{date_col}_days_since_ref"] = (
        ((dt - ref_date).dt.days).fillna(0).astype(np.float32)
    )

    # Cyclical encodings
    month_rad = 2 * np.pi * feats[f"{date_col}_month"] / 12.0
    feats[f"{date_col}_month_sin"] = np.sin(month_rad).astype(np.float32)
    feats[f"{date_col}_month_cos"] = np.cos(month_rad).astype(np.float32)

    doy_rad = 2 * np.pi * feats[f"{date_col}_dayofyear"] / 365.25
    feats[f"{date_col}_doy_sin"] = np.sin(doy_rad).astype(np.float32)
    feats[f"{date_col}_doy_cos"] = np.cos(doy_rad).astype(np.float32)

    return feats


# ==============================================================================
# 2. Main Data Processing & Feature Pipeline
# ==============================================================================
def process_data_and_train():
    print("Loading raw datasets from ./input...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df = standardize_col_names(train_df)
    test_df = standardize_col_names(test_df)

    # Identify target column
    target_col = find_column(
        train_df, ["NADAC Rate", "NADAC_Rate", "NADAC_Per_Unit", "nadac_rate", "target"]
    )
    if target_col is None:
        raise ValueError(
            f"Target column not found in train. Available columns: {list(train_df.columns)}"
        )

    y_train = train_df[target_col].astype(np.float32).values

    # Identify key domain columns
    desc_col = find_column(
        train_df,
        [
            "NDC Description",
            "NDC_Description",
            "Description",
            "ndc_description",
            "drug_name",
        ],
    )
    ndc_col = find_column(train_df, ["NDC", "ndc", "ndc_code", "National_Drug_Code"])
    eff_date_col = find_column(
        train_df,
        [
            "Effective Date",
            "Effective_Date",
            "As of Date",
            "effective_date",
            "Rate_Date",
        ],
    )
    gen_eff_date_col = find_column(
        train_df,
        [
            "Corresponding Generic Drug Effective Date",
            "Corresponding_Generic_Drug_Effective_Date",
        ],
    )
    gen_rate_col = find_column(
        train_df,
        [
            "Corresponding Generic Drug NADAC Per Unit",
            "Corresponding_Generic_Drug_NADAC_Per_Unit",
        ],
    )
    unit_col = find_column(train_df, ["Pricing Unit", "Pricing_Unit", "pricing_unit"])
    pharm_col = find_column(
        train_df,
        ["Pharmacy Type Indicator", "Pharmacy_Type_Indicator", "pharmacy_type"],
    )
    otc_col = find_column(train_df, ["OTC", "otc", "otc_indicator"])
    class_col = find_column(
        train_df,
        [
            "Classification for Rate Setting",
            "Classification_for_Rate_Setting",
            "classification",
        ],
    )
    expl_col = find_column(
        train_df, ["Explanation Code", "Explanation_Code", "explanation_code"]
    )

    X_train_df = pd.DataFrame(index=train_df.index)
    X_test_df = pd.DataFrame(index=test_df.index)

    # 1. NDC Hierarchical Features
    if ndc_col:
        ndc_train = extract_ndc_components(train_df[ndc_col])
        ndc_test = extract_ndc_components(test_df[ndc_col])
        X_train_df = pd.concat([X_train_df, ndc_train], axis=1)
        X_test_df = pd.concat([X_test_df, ndc_test], axis=1)

    # 2. Text Domain & Dosage Features
    if desc_col:
        text_train = extract_text_domain_features(train_df[desc_col])
        text_test = extract_text_domain_features(test_df[desc_col])
        X_train_df = pd.concat([X_train_df, text_train], axis=1)
        X_test_df = pd.concat([X_test_df, text_test], axis=1)

        # Rich TF-IDF (word + char n-grams) and SVD Embeddings
        raw_train_text = train_df[desc_col].fillna("").astype(str).tolist()
        raw_test_text = test_df[desc_col].fillna("").astype(str).tolist()

        # Word-level TF-IDF
        tfidf_word = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_word_train = tfidf_word.fit_transform(raw_train_text)
        tfidf_word_test = tfidf_word.transform(raw_test_text)

        n_svd_w = min(20, tfidf_word_train.shape[1] - 1)
        svd_w = TruncatedSVD(n_components=n_svd_w, random_state=42)
        svd_w_train = svd_w.fit_transform(tfidf_word_train)
        svd_w_test = svd_w.transform(tfidf_word_test)

        for i in range(n_svd_w):
            X_train_df[f"tfidf_word_svd_{i}"] = svd_w_train[:, i].astype(np.float32)
            X_test_df[f"tfidf_word_svd_{i}"] = svd_w_test[:, i].astype(np.float32)

        # Character-level TF-IDF
        tfidf_char = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char_wb",
            max_features=2500,
            sublinear_tf=True,
        )
        tfidf_char_train = tfidf_char.fit_transform(raw_train_text)
        tfidf_char_test = tfidf_char.transform(raw_test_text)

        n_svd_c = min(12, tfidf_char_train.shape[1] - 1)
        svd_c = TruncatedSVD(n_components=n_svd_c, random_state=42)
        svd_c_train = svd_c.fit_transform(tfidf_char_train)
        svd_c_test = svd_c.transform(tfidf_char_test)

        for i in range(n_svd_c):
            X_train_df[f"tfidf_char_svd_{i}"] = svd_c_train[:, i].astype(np.float32)
            X_test_df[f"tfidf_char_svd_{i}"] = svd_c_test[:, i].astype(np.float32)

        # OOF Cross-Validated Ridge Meta-Feature on Combined Text TF-IDF
        tfidf_full_tr = sparse_hstack([tfidf_word_train, tfidf_char_train])
        tfidf_full_te = sparse_hstack([tfidf_word_test, tfidf_char_test])
        text_kf = KFold(n_splits=10, shuffle=True, random_state=42)
        oof_text_ridge = np.zeros(len(train_df), dtype=np.float32)
        test_text_ridge = np.zeros(len(test_df), dtype=np.float32)

        for tr_idx, val_idx in text_kf.split(train_df):
            ridge_text = Ridge(alpha=1.0, random_state=42)
            ridge_text.fit(tfidf_full_tr.tocsr()[tr_idx], y_train[tr_idx])
            oof_text_ridge[val_idx] = np.maximum(0, ridge_text.predict(tfidf_full_tr.tocsr()[val_idx]))
            test_text_ridge += np.maximum(0, ridge_text.predict(tfidf_full_te.tocsr())) / 10.0

        X_train_df["tfidf_ridge_meta"] = oof_text_ridge
        X_test_df["tfidf_ridge_meta"] = test_text_ridge

    # 3. Temporal Features
    if eff_date_col:
        dt_train = extract_date_features(train_df, eff_date_col)
        dt_test = extract_date_features(test_df, eff_date_col)
        X_train_df = pd.concat([X_train_df, dt_train], axis=1)
        X_test_df = pd.concat([X_test_df, dt_test], axis=1)

    if gen_eff_date_col:
        gen_dt_train = extract_date_features(train_df, gen_eff_date_col)
        gen_dt_test = extract_date_features(test_df, gen_eff_date_col)
        X_train_df = pd.concat([X_train_df, gen_dt_train], axis=1)
        X_test_df = pd.concat([X_test_df, gen_dt_test], axis=1)

        if eff_date_col:
            d1_train = pd.to_datetime(train_df[eff_date_col], errors="coerce")
            d2_train = pd.to_datetime(train_df[gen_eff_date_col], errors="coerce")
            X_train_df["date_diff_gen_effective"] = (
                ((d1_train - d2_train).dt.days).fillna(0).astype(np.float32)
            )

            d1_test = pd.to_datetime(test_df[eff_date_col], errors="coerce")
            d2_test = pd.to_datetime(test_df[gen_eff_date_col], errors="coerce")
            X_test_df["date_diff_gen_effective"] = (
                ((d1_test - d2_test).dt.days).fillna(0).astype(np.float32)
            )

    # 4. Corresponding Generic Drug NADAC Per Unit Features
    if gen_rate_col:
        gen_r_train = (
            pd.to_numeric(train_df[gen_rate_col], errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )
        gen_r_test = (
            pd.to_numeric(test_df[gen_rate_col], errors="coerce")
            .fillna(0)
            .astype(np.float32)
        )

        X_train_df["gen_drug_nadac_rate"] = gen_r_train
        X_train_df["gen_drug_nadac_has_value"] = (gen_r_train > 0).astype(np.int8)
        X_train_df["gen_drug_nadac_log1p"] = np.log1p(np.maximum(0, gen_r_train))

        X_test_df["gen_drug_nadac_rate"] = gen_r_test
        X_test_df["gen_drug_nadac_has_value"] = (gen_r_test > 0).astype(np.int8)
        X_test_df["gen_drug_nadac_log1p"] = np.log1p(np.maximum(0, gen_r_test))

    # 5. Categorical Encodings & Interaction Terms
    cat_cols = [
        c for c in [unit_col, pharm_col, otc_col, class_col, expl_col] if c is not None
    ]

    interactions = []
    if unit_col and class_col:
        interactions.append((unit_col, class_col, "unit_x_class"))
    if unit_col and otc_col:
        interactions.append((unit_col, otc_col, "unit_x_otc"))
    if class_col and otc_col:
        interactions.append((class_col, otc_col, "class_x_otc"))
    if pharm_col and class_col:
        interactions.append((pharm_col, class_col, "pharm_x_class"))

    for col1, col2, inter_name in interactions:
        s_train = train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
        s_test = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)
        freq_map = s_train.value_counts(normalize=True).to_dict()
        X_train_df[f"freq_{inter_name}"] = (
            s_train.map(freq_map).fillna(0).astype(np.float32)
        )
        X_test_df[f"freq_{inter_name}"] = (
            s_test.map(freq_map).fillna(0).astype(np.float32)
        )

    for col in cat_cols:
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col).lower()
        s_train = train_df[col].fillna("MISSING").astype(str)
        s_test = test_df[col].fillna("MISSING").astype(str)

        # Frequency encoding
        freq = s_train.value_counts(normalize=True).to_dict()
        X_train_df[f"freq_{clean_col}"] = s_train.map(freq).fillna(0).astype(np.float32)
        X_test_df[f"freq_{clean_col}"] = s_test.map(freq).fillna(0).astype(np.float32)

        # Categorical ordinal codes
        categories = sorted(s_train.unique().tolist())
        cat_map = {cat: idx for idx, cat in enumerate(categories)}
        X_train_df[f"cat_{clean_col}"] = (
            s_train.map(cat_map).fillna(-1).astype(np.int32)
        )
        X_test_df[f"cat_{clean_col}"] = s_test.map(cat_map).fillna(-1).astype(np.int32)

    # 6. Numeric features
    raw_num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    for num_col in raw_num_cols:
        if num_col not in [target_col] and num_col not in X_train_df.columns:
            clean_name = f"raw_num_{re.sub(r'[^a-zA-Z0-9_]', '_', num_col).lower()}"
            if num_col in test_df.columns:
                X_train_df[clean_name] = train_df[num_col].fillna(0).astype(np.float32)
                X_test_df[clean_name] = test_df[num_col].fillna(0).astype(np.float32)

    # Drop string categorical columns from tabular matrix before numerical processing
    str_cols_train = [c for c in X_train_df.columns if X_train_df[c].dtype == "object"]
    for c in str_cols_train:
        X_train_df.drop(columns=[c], inplace=True)
        if c in X_test_df.columns:
            X_test_df.drop(columns=[c], inplace=True)

    # Clean feature names for LightGBM/XGBoost compatibility
    clean_feature_names = [
        re.sub(r"[ ,:\'\"\[\]\{\}\(\)<>/\\]", "_", c) for c in X_train_df.columns
    ]
    X_train_df.columns = clean_feature_names
    X_test_df.columns = clean_feature_names

    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Setup 10-Fold Partition
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(X_train_df), dtype=np.int32)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
        fold_assignments[val_idx] = fold

    X_train_df["fold"] = fold_assignments
    base_feature_cols = [c for c in clean_feature_names if c != "fold"]
    n_train = len(X_train_df)
    n_test = len(X_test_df)

    # Prepare categorical series for out-of-fold target encoding
    target_series = pd.Series(y_train)
    labeler_train = train_df[ndc_col].astype(str).str.slice(0, 5) if ndc_col else pd.Series(["UNK"] * n_train)
    labeler_test = test_df[ndc_col].astype(str).str.slice(0, 5) if ndc_col else pd.Series(["UNK"] * n_test)

    unit_s_train = train_df[unit_col].astype(str) if unit_col else pd.Series(["UNK"] * n_train)
    unit_s_test = test_df[unit_col].astype(str) if unit_col else pd.Series(["UNK"] * n_test)

    drug_stem_train = train_df[desc_col].fillna("").astype(str).apply(lambda x: x.split()[0] if x.split() else "UNK") if desc_col else pd.Series(["UNK"] * n_train)
    drug_stem_test = test_df[desc_col].fillna("").astype(str).apply(lambda x: x.split()[0] if x.split() else "UNK") if desc_col else pd.Series(["UNK"] * n_test)

    labeler_unit_train = labeler_train + "_" + unit_s_train
    labeler_unit_test = labeler_test + "_" + unit_s_test

    print(
        f"Prepared {n_train} training samples, {n_test} test samples with {len(base_feature_cols)} base features across {n_splits} folds."
    )

    # ==============================================================================
    # 3. Model Training & Evaluation Loop (Diverse Model Ensemble)
    # ==============================================================================
    model_names = [
        "LGB_raw",
        "LGB_log",
        "CAT_raw",
        "CAT_log",
        "XGB_raw",
        "XGB_log",
        "ExtraTrees",
        "Ridge_log",
    ]
    n_models = len(model_names)

    oof_preds = {name: np.zeros(n_train, dtype=np.float32) for name in model_names}
    test_preds = {name: np.zeros(n_test, dtype=np.float32) for name in model_names}

    for fold in range(n_splits):
        tr_mask = X_train_df["fold"] != fold
        va_mask = X_train_df["fold"] == fold

        # Compute strictly leakage-free Out-Of-Fold Bayesian Target Statistics
        fold_feat_tr = []
        fold_feat_va = []
        fold_feat_te = []

        enc_groups = [
            ("labeler", labeler_train, labeler_test, 10.0),
            ("unit", unit_s_train, unit_s_test, 5.0),
            ("stem", drug_stem_train, drug_stem_test, 5.0),
            ("labeler_unit", labeler_unit_train, labeler_unit_test, 5.0),
        ]

        for name, tr_s, te_s, smooth_val in enc_groups:
            enc_tr, enc_va, enc_te = compute_hierarchical_target_aggregations(
                tr_s[tr_mask], target_series[tr_mask], tr_s[va_mask], te_s, smoothing=smooth_val
            )
            fold_feat_tr.append(enc_tr)
            fold_feat_va.append(enc_va)
            fold_feat_te.append(enc_te)

        enc_tr_arr = np.hstack(fold_feat_tr)
        enc_va_arr = np.hstack(fold_feat_va)
        enc_te_arr = np.hstack(fold_feat_te)

        X_tr = np.hstack([X_train_df.loc[tr_mask, base_feature_cols].values, enc_tr_arr])
        y_tr = y_train[tr_mask]
        y_tr_log = np.log1p(np.maximum(0, y_tr))

        X_va = np.hstack([X_train_df.loc[va_mask, base_feature_cols].values, enc_va_arr])
        y_va = y_train[va_mask]
        y_va_log = np.log1p(np.maximum(0, y_va))

        X_te = np.hstack([X_test_df[base_feature_cols].values, enc_te_arr])

        # 1. LightGBM (Raw Target)
        lgb_raw = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=15,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_raw.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        )
        p_lgb_raw_va = np.maximum(0, lgb_raw.predict(X_va))
        oof_preds["LGB_raw"][va_mask] = p_lgb_raw_va
        test_preds["LGB_raw"] += np.maximum(0, lgb_raw.predict(X_te)) / n_splits

        # 2. LightGBM (Log Target)
        lgb_log = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=15,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_log.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        )
        p_lgb_log_va = np.maximum(0, np.expm1(lgb_log.predict(X_va)))
        oof_preds["LGB_log"][va_mask] = p_lgb_log_va
        test_preds["LGB_log"] += np.maximum(0, np.expm1(lgb_log.predict(X_te))) / n_splits

        # 3. CatBoost (Raw Target)
        cat_raw = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1,
        )
        cat_raw.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
        p_cat_raw_va = np.maximum(0, cat_raw.predict(X_va))
        oof_preds["CAT_raw"][va_mask] = p_cat_raw_va
        test_preds["CAT_raw"] += np.maximum(0, cat_raw.predict(X_te)) / n_splits

        # 4. CatBoost (Log Target)
        cat_log = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1,
        )
        cat_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), early_stopping_rounds=50, verbose=False)
        p_cat_log_va = np.maximum(0, np.expm1(cat_log.predict(X_va)))
        oof_preds["CAT_log"][va_mask] = p_cat_log_va
        test_preds["CAT_log"] += np.maximum(0, np.expm1(cat_log.predict(X_te))) / n_splits

        # 5. XGBoost (Raw Target)
        xgb_raw = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_raw.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb_raw_va = np.maximum(0, xgb_raw.predict(X_va))
        oof_preds["XGB_raw"][va_mask] = p_xgb_raw_va
        test_preds["XGB_raw"] += np.maximum(0, xgb_raw.predict(X_te)) / n_splits

        # 6. XGBoost (Log Target)
        xgb_log = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        xgb_log.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
        p_xgb_log_va = np.maximum(0, np.expm1(xgb_log.predict(X_va)))
        oof_preds["XGB_log"][va_mask] = p_xgb_log_va
        test_preds["XGB_log"] += np.maximum(0, np.expm1(xgb_log.predict(X_te))) / n_splits

        # 7. ExtraTrees Regressor
        et_model = ExtraTreesRegressor(
            n_estimators=350,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        p_et_va = np.maximum(0, et_model.predict(X_va))
        oof_preds["ExtraTrees"][va_mask] = p_et_va
        test_preds["ExtraTrees"] += np.maximum(0, et_model.predict(X_te)) / n_splits

        # 8. Ridge Regressor on Scaled Tabular + Text Features
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        ridge_model = Ridge(alpha=10.0, random_state=42 + fold)
        ridge_model.fit(X_tr_sc, y_tr_log)
        p_rd_va = np.maximum(0, np.expm1(ridge_model.predict(X_va_sc)))
        oof_preds["Ridge_log"][va_mask] = p_rd_va
        test_preds["Ridge_log"] += np.maximum(0, np.expm1(ridge_model.predict(X_te_sc))) / n_splits

    # Print individual model CV RMSE scores
    print("\n--- Individual Model Out-of-Fold Performance ---")
    for name in model_names:
        score = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"Model: {name:<12} | OOF RMSE: {score:.5f}")

    # ==============================================================================
    # 4. Numerically Optimized Blending Weights (SLSQP Bounded Optimization)
    # ==============================================================================
    oof_mat = np.column_stack([oof_preds[name] for name in model_names])
    test_mat = np.column_stack([test_preds[name] for name in model_names])

    def loss_func(w):
        pred = np.dot(oof_mat, w)
        return np.sqrt(mean_squared_error(y_train, pred))

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    opt_result = minimize(
        loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
    )

    optimal_weights = opt_result.x if opt_result.success else init_weights
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    print("\n--- Optimal Ensemble Weights ---")
    for name, w in zip(model_names, optimal_weights):
        print(f"  {name:<12}: {w:.4f}")

    # Compute final ensemble OOF predictions and RMSE
    blended_oof = np.maximum(0.0, np.dot(oof_mat, optimal_weights))
    final_oof_rmse = float(np.sqrt(mean_squared_error(y_train, blended_oof)))
    print(f"\nOverall OOF Ensemble RMSE: {final_oof_rmse:.6f}")

    # Generate Final Test Predictions
    final_test_preds = np.maximum(0.0, np.dot(test_mat, optimal_weights))

    submission_df = pd.DataFrame(
        {"id": np.arange(len(final_test_preds)), "NADAC Rate": final_test_preds}
    )

    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission file to {submission_path}")

    # Print Final Validation Score exactly as required
    print(f"Final Validation Score: {final_oof_rmse}")


if __name__ == "__main__":
    process_data_and_train()