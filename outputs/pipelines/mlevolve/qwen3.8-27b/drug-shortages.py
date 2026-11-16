import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TARGET = "availability"
RANDOM_STATE = 42
VAL_FRACTION = 0.15
MAX_ONEHOT_CATEGORIES = 35
TFIDF_MAX_FEATURES = 128
TFIDF_NGRAM_RANGE = (2, 4)
MAX_EPOCHS = 300
PATIENCE = 50
BATCH_SIZE = 64

ID_COL_CANDIDATES = {
    "id",
    "row_id",
    "sample_id",
    "index",
    "idx",
    "uid",
    "drug_id",
    "ndc_id",
    "package_id",
    "record_id",
    "submission_id",
}
MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "missing", "unknown", " "}
DATE_HINTS = ("date", "time", "dt", "timestamp")
CODE_HINTS = ("ndc", "code", "zip", "postal", "isbn", "sku", "upc", "gtin", "rxnorm")
DOSE_HINTS = (
    "dose",
    "dosage",
    "strength",
    "potency",
    "concentration",
    "amount",
    "mg",
    "mcg",
)
MANUFACTURER_HINTS = (
    "manufacturer",
    "manufacturer_name",
    "maker",
    "supplier",
    "company",
    "brand",
)
PRODUCT_HINTS = (
    "drug_name",
    "name",
    "product",
    "active_ingredient",
    "ingredient",
    "formulation",
    "form",
)


def clean_strings(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(
        series
    ):
        return series
    s = series.astype("string").str.strip()
    s = s.str.lower().str.strip()
    s = s.replace(
        [np.nan, "nan", "None", "null", "N/A", "NA", "Missing", "Unknown", " "], pd.NA
    )
    return s


def is_missing_token(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    s = series.astype("string").str.replace(",", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_datetime_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype("string")
    return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text)


def parse_ndc(value):
    if pd.isna(value):
        return np.nan, np.nan, np.nan, 0, False
    s = str(value).strip()
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return np.nan, np.nan, np.nan, len(digits), "-" in s
    labeler = digits[:5]
    product = digits[5:9]
    package = digits[9:10]
    try:
        return float(labeler), float(product), float(package), len(digits), "-" in s
    except Exception:
        return np.nan, np.nan, np.nan, len(digits), "-" in s


def parse_dose(text: str):
    if not text:
        return np.nan, ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zµμ%°]+)?", text)
    if not m:
        return np.nan, ""
    try:
        value = float(m.group(1))
    except Exception:
        return np.nan, ""
    unit = (m.group(2) or "").strip()
    unit = unit.replace("µ", "u").replace("μ", "u")
    return value, unit


class FeatureBuilder:
    def __init__(self):
        self.numeric_cols = []
        self.datetime_cols = []
        self.categorical_cols = []
        self.numeric_median = {}
        self.numeric_mean = {}
        self.numeric_std = {}
        self.numeric_min = {}
        self.numeric_max = {}
        self.numeric_log = {}
        self.datetime_min = {}
        self.datetime_max = {}
        self.cat_freq = {}
        self.cat_onehot_categories = {}
        self.cat_tfidf = {}
        self.dose_unit_freq = {}
        self.feature_names = []

    def _is_date_col(self, name: str, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        lname = str(name).lower()
        if any(h in lname for h in DATE_HINTS):
            parsed = parse_datetime_safe(series.head(200))
            if parsed.notna().sum() >= max(1, int(0.5 * len(parsed))):
                return True
        return False

    def _is_code_col(self, name: str) -> bool:
        lname = str(name).lower()
        return any(h in lname for h in CODE_HINTS) or lname.endswith("_id")

    def _is_numeric_col(self, name: str, series: pd.Series) -> bool:
        if self._is_code_col(name):
            return False
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return True
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            numeric = to_numeric_safe(series)
            return numeric.notna().sum() >= max(1, 0.8 * len(series))
        return False

    def fit(self, train_features: pd.DataFrame):
        self.numeric_cols = []
        self.datetime_cols = []
        self.categorical_cols = []

        for col in train_features.columns:
            series = train_features[col]
            if self._is_date_col(col, series):
                self.datetime_cols.append(col)
            elif self._is_numeric_col(col, series):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        for col in self.numeric_cols:
            s = to_numeric_safe(train_features[col])
            self.numeric_median[col] = float(s.median()) if s.notna().any() else 0.0
            self.numeric_mean[col] = float(s.mean()) if s.notna().any() else 0.0
            std = float(s.std()) if s.notna().sum() > 1 else 0.0
            self.numeric_std[col] = std if std > 1e-12 else 1.0
            self.numeric_min[col] = float(s.min()) if s.notna().any() else 0.0
            self.numeric_max[col] = float(s.max()) if s.notna().any() else 1.0
            self.numeric_log[col] = bool(
                s.dropna().ge(0).all() and self.numeric_max[col] > 1.0
            )

        for col in self.datetime_cols:
            s = parse_datetime_safe(train_features[col])
            valid = s.dropna()
            self.datetime_min[col] = valid.min().to_pydatetime() if len(valid) else None
            self.datetime_max[col] = valid.max().to_pydatetime() if len(valid) else None

        for col in self.categorical_cols:
            s = clean_strings(train_features[col]).fillna("__MISSING__").astype(str)
            freq = s.value_counts(normalize=True).to_dict()
            self.cat_freq[col] = freq
            nunique = s.nunique()
            if nunique <= MAX_ONEHOT_CATEGORIES:
                self.cat_onehot_categories[col] = sorted(s.unique().tolist())
            else:
                self.cat_onehot_categories[col] = []
            lname = str(col).lower()
            is_texty = (
                s.str.len().mean() > 6
                or s.str.contains(" ").sum() > 0
                or nunique > MAX_ONEHOT_CATEGORIES
                or any(h in lname for h in PRODUCT_HINTS + MANUFACTURER_HINTS)
            )
            if is_texty:
                docs = s.replace("__MISSING__", "").tolist()
                if any(len(d) > 0 for d in docs):
                    vec = TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=TFIDF_NGRAM_RANGE,
                        max_features=TFIDF_MAX_FEATURES,
                        sublinear_tf=True,
                        min_df=1,
                        dtype=np.float32,
                    )
                    vec.fit(docs)
                    self.cat_tfidf[col] = vec
                else:
                    self.cat_tfidf[col] = None
            else:
                self.cat_tfidf[col] = None

        # Pre-compute dose unit frequencies from training data only
        self.dose_unit_freq = {}
        for col in self.categorical_cols:
            lname = str(col).lower()
            if any(h in lname for h in DOSE_HINTS):
                s = clean_strings(train_features[col]).fillna("__MISSING__").astype(str)
                text = s.map(normalize_text)
                dose = text.map(parse_dose)
                dose_unit = dose.map(lambda r: r[1]).astype(str).replace("", "__NONE__")
                self.dose_unit_freq[col] = dose_unit.value_counts(normalize=True).to_dict()

        self.feature_names = self._build(train_features).columns.tolist()
        return self

    def _build_numeric(self, col: str, series: pd.Series) -> dict:
        out = {}
        missing = pd.isna(series).astype("int8")
        out[f"{col}_missing"] = missing
        x = to_numeric_safe(series)
        median = self.numeric_median.get(col, 0.0)
        x = x.fillna(median)
        out[f"{col}_raw"] = x.astype("float64")
        mean = self.numeric_mean.get(col, 0.0)
        std = self.numeric_std.get(col, 1.0)
        out[f"{col}_z"] = ((x - mean) / std).astype("float64")
        mn = self.numeric_min.get(col, 0.0)
        mx = self.numeric_max.get(col, 1.0)
        denom = mx - mn if mx > mn else 1.0
        out[f"{col}_minmax"] = ((x - mn) / denom).astype("float64")
        if self.numeric_log.get(col, False):
            out[f"{col}_log1p"] = np.log1p(x.clip(lower=0)).astype("float64")
        return out

    def _build_datetime(self, col: str, series: pd.Series) -> dict:
        out = {}
        out[f"{col}_missing"] = pd.isna(series).astype("int8")
        x = parse_datetime_safe(series)
        min_dt = self.datetime_min.get(col)
        max_dt = self.datetime_max.get(col)
        if min_dt is not None:
            x = x.fillna(pd.Timestamp(min_dt))
        else:
            x = x.fillna(pd.Timestamp("1970-01-01"))
        out[f"{col}_year"] = x.dt.year.astype("float64")
        out[f"{col}_month"] = x.dt.month.astype("float64")
        out[f"{col}_day"] = x.dt.day.astype("float64")
        out[f"{col}_dayofweek"] = x.dt.dayofweek.astype("float64")
        out[f"{col}_quarter"] = x.dt.quarter.astype("float64")
        out[f"{col}_dayofyear"] = x.dt.dayofyear.astype("float64")
        if min_dt is not None:
            out[f"{col}_days_since_min"] = (x - pd.Timestamp(min_dt)).dt.days.astype(
                "float64"
            )
        if max_dt is not None:
            out[f"{col}_days_until_max"] = (pd.Timestamp(max_dt) - x).dt.days.astype(
                "float64"
            )
        return out

    def _build_categorical(self, col: str, series: pd.Series) -> dict:
        out = {}
        missing = pd.isna(series).astype("int8")
        out[f"{col}_missing"] = missing
        s = clean_strings(series).fillna("__MISSING__").astype(str)
        text = s.map(normalize_text)
        out[f"{col}_len"] = s.str.len().astype("float64")
        tokens = text.map(tokenize)
        out[f"{col}_token_count"] = tokens.map(len).astype("float64")
        out[f"{col}_unique_token_count"] = tokens.map(lambda t: len(set(t))).astype(
            "float64"
        )
        out[f"{col}_digit_count"] = s.str.count(r"[0-9]").astype("float64")
        out[f"{col}_upper_count"] = (
            series.astype("string").fillna("").str.count(r"[A-Z]").astype("float64")
        )
        out[f"{col}_punct_count"] = s.str.count(r"[^a-z0-9]").astype("float64")
        freq = self.cat_freq.get(col, {})
        out[f"{col}_freq"] = s.map(freq).fillna(0.0).astype("float64")
        lname = str(col).lower()
        if any(h in lname for h in ("ndc",)):
            parsed = series.map(parse_ndc)
            out[f"{col}_labeler"] = parsed.map(lambda r: r[0]).astype("float64")
            out[f"{col}_product"] = parsed.map(lambda r: r[1]).astype("float64")
            out[f"{col}_package"] = parsed.map(lambda r: r[2]).astype("float64")
            out[f"{col}_digit_len"] = parsed.map(lambda r: r[3]).astype("float64")
            out[f"{col}_has_dash"] = parsed.map(lambda r: int(r[4])).astype("float64")
        if any(h in lname for h in DOSE_HINTS):
            dose = text.map(parse_dose)
            dose_value = dose.map(lambda r: r[0]).astype("float64")
            dose_unit = dose.map(lambda r: r[1]).astype(str).replace("", "__NONE__")
            out[f"{col}_dose_value"] = dose_value
            out[f"{col}_dose_log1p"] = np.log1p(
                dose_value.fillna(0.0).clip(lower=0)
            ).astype("float64")
            unit_freq = self.dose_unit_freq.get(col, {})
            out[f"{col}_dose_unit_freq"] = (
                dose_unit.map(unit_freq).fillna(0.0).astype("float64")
            )
            for unit in sorted(unit_freq.keys())[:15]:
                out[f"{col}_unit_{unit}"] = (dose_unit == unit).astype("float64")
        categories = self.cat_onehot_categories.get(col, [])
        for cat in categories:
            safe_cat = re.sub(r"[^a-z0-9]+", "_", cat)[:30]
            out[f"{col}_oh_{safe_cat}"] = (s == cat).astype("float64")
        tfidf = self.cat_tfidf.get(col)
        if tfidf is not None:
            docs = text.replace("__MISSING__", "").tolist()
            X = tfidf.transform(docs).toarray().astype("float32")
            feature_names = tfidf.get_feature_names_out()
            for j, fname in enumerate(feature_names):
                safe_name = re.sub(r"[^a-z0-9]+", "_", f"{col}_tf_{fname}")[:80]
                out[f"{safe_name}_{j}"] = X[:, j]
        return out

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for col in df.columns:
            if col in self.numeric_cols:
                pieces.append(
                    pd.DataFrame(self._build_numeric(col, df[col]), index=df.index)
                )
            elif col in self.datetime_cols:
                pieces.append(
                    pd.DataFrame(self._build_datetime(col, df[col]), index=df.index)
                )
            else:
                pieces.append(
                    pd.DataFrame(self._build_categorical(col, df[col]), index=df.index)
                )
        if not pieces:
            return pd.DataFrame(index=df.index)
        out = pd.concat(pieces, axis=1)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.astype("float64")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._build(df)
        if self.feature_names:
            for col in self.feature_names:
                if col not in out.columns:
                    out[col] = 0.0
            out = out[self.feature_names]
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class DualPathTabularNet(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        dense_indices,
        sparse_indices,
        dense_hidden=128,
        sparse_hidden=128,
        fusion_hidden=96,
        dropout=0.2,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.dense_indices = list(dense_indices)
        self.sparse_indices = list(sparse_indices)
        dense_dim = len(self.dense_indices)
        sparse_dim = len(self.sparse_indices)
        self.has_dense = dense_dim > 0
        self.has_sparse = sparse_dim > 0
        if self.has_dense:
            self.dense_embed = nn.Linear(dense_dim, dense_hidden)
            self.dense_norm = nn.LayerNorm(dense_hidden)
            self.dense_block = ResidualBlock(dense_hidden, dropout=dropout)
        if self.has_sparse:
            self.sparse_embed = nn.Linear(sparse_dim, sparse_hidden)
            self.sparse_norm = nn.LayerNorm(sparse_hidden)
            self.sparse_block = ResidualBlock(sparse_hidden, dropout=dropout)
        fusion_dim = 0
        if self.has_dense:
            fusion_dim += dense_hidden
        if self.has_sparse:
            fusion_dim += sparse_hidden
        if fusion_dim == 0:
            fusion_dim = 1
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_block = ResidualBlock(fusion_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, max(1, self.n_classes)),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        parts = []
        if self.has_dense:
            dx = x[:, self.dense_indices]
            h = self.dense_embed(dx)
            h = self.dense_norm(h)
            h = self.dense_block(h)
            parts.append(h)
        if self.has_sparse:
            sx = x[:, self.sparse_indices]
            h = self.sparse_embed(sx)
            h = self.sparse_norm(h)
            h = self.sparse_block(h)
            parts.append(h)
        if not parts:
            parts.append(torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype))
        h = torch.cat(parts, dim=1)
        h = self.fusion_norm(h)
        h = self.fusion_block(h)
        logits = self.head(h)
        return logits


class ClassWeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=1.0, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if class_weights is not None:
            class_weights = class_weights.detach().float()
            if class_weights.numel() == 0:
                class_weights = None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

    def forward(self, logits, targets):
        targets = targets.long()
        if logits.size(-1) == 1:
            logits = logits.view(-1)
            targets = targets.view(-1).float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = torch.exp(-bce)
            if self.class_weights is not None and self.class_weights.numel() >= 2:
                alpha = self.class_weights[targets.long().clamp(0, 1)]
            else:
                alpha = torch.ones_like(pt)
            loss = alpha * (1.0 - pt).pow(self.gamma) * bce
        else:
            log_probs = F.log_softmax(logits, dim=1)
            nll = -log_probs.gather(1, targets.view(-1, 1)).view(-1)
            pt = torch.exp(-nll)
            if self.class_weights is not None:
                alpha = self.class_weights[targets]
            else:
                alpha = torch.ones_like(pt)
            loss = alpha * (1.0 - pt).pow(self.gamma) * nll
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def infer_feature_groups(feature_names: list):
    sparse_markers = ("_oh_", "_tf_", "_unit_", "_has_dash")
    dense_indices = []
    sparse_indices = []
    for i, name in enumerate(feature_names):
        lname = str(name).lower()
        if any(marker in lname for marker in sparse_markers):
            sparse_indices.append(i)
        else:
            dense_indices.append(i)
    return dense_indices, sparse_indices


def main():
    # === DATA LOADING ===
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    # Extract target BEFORE dropping
    y_full = train[TARGET].copy()

    # Identify id columns
    test_id_col = None
    for col in test.columns:
        if str(col).lower() in ID_COL_CANDIDATES:
            test_id_col = col
            break
    if test_id_col is not None:
        test_ids = test[test_id_col].astype("int64").values
    else:
        test_ids = np.arange(len(test), dtype="int64")

    train_id_col = None
    for col in train.columns:
        if str(col).lower() in ID_COL_CANDIDATES:
            train_id_col = col
            break

    # Drop id and target from features
    drop_cols_train = [TARGET]
    if train_id_col is not None:
        drop_cols_train.append(train_id_col)
    drop_cols_test = []
    if test_id_col is not None:
        drop_cols_test.append(test_id_col)

    train = train.drop(
        columns=[c for c in drop_cols_train if c in train.columns], errors="ignore"
    )
    test = test.drop(
        columns=[c for c in drop_cols_test if c in test.columns], errors="ignore"
    )

    # Align columns
    all_feature_cols = list(dict.fromkeys(list(train.columns) + list(test.columns)))
    train = train.reindex(columns=all_feature_cols)
    test = test.reindex(columns=all_feature_cols)

    # Clean strings globally before split
    for col in train.columns:
        if pd.api.types.is_string_dtype(train[col]) or pd.api.types.is_object_dtype(
            train[col]
        ):
            train[col] = clean_strings(train[col])
    for col in test.columns:
        if pd.api.types.is_string_dtype(test[col]) or pd.api.types.is_object_dtype(
            test[col]
        ):
            test[col] = clean_strings(test[col])

    # Valid target mask
    valid_target_mask = ~y_full.isna()
    train_valid = train.loc[valid_target_mask].copy()
    y_valid = y_full.loc[valid_target_mask].copy()

    # Map target to integers using train-only labels
    unique_targets = y_valid.dropna().unique().tolist()
    target_to_int = {
        v: i for i, v in enumerate(sorted(unique_targets, key=lambda x: str(x)))
    }
    int_to_target = {i: v for v, i in target_to_int.items()}

    y_valid_int = y_valid.map(target_to_int).astype("int64")

    if len(train_valid) == 0:
        raise ValueError("No valid training rows with target.")

    # Split first, then fit feature engineering on train split only
    try:
        train_idx, val_idx = train_test_split(
            train_valid.index,
            test_size=VAL_FRACTION,
            random_state=RANDOM_STATE,
            stratify=y_valid_int,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_valid.index,
            test_size=VAL_FRACTION,
            random_state=RANDOM_STATE,
        )

    X_train_raw = train_valid.loc[train_idx, all_feature_cols].copy()
    X_val_raw = train_valid.loc[val_idx, all_feature_cols].copy()
    X_test_raw = test[all_feature_cols].copy()

    y_train = y_valid_int.loc[train_idx].astype("int64")
    y_val = y_valid_int.loc[val_idx].astype("int64")

    # === FEATURE ENGINEERING ===
    builder = FeatureBuilder().fit(X_train_raw)
    X_train = builder.transform(X_train_raw)
    X_val = builder.transform(X_val_raw)
    X_test = builder.transform(X_test_raw)

    # Ensure finite values
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_names = builder.feature_names
    n_features = len(feature_names)
    n_classes = len(int_to_target)
    n_classes = max(1, n_classes)

    # === MODEL DESIGN ===
    dense_indices, sparse_indices = infer_feature_groups(feature_names)
    if not feature_names:
        dense_indices = list(range(n_features))
        sparse_indices = []

    model = DualPathTabularNet(
        n_features=n_features,
        n_classes=n_classes,
        dense_indices=dense_indices,
        sparse_indices=sparse_indices,
        dense_hidden=128,
        sparse_hidden=128,
        fusion_hidden=96,
        dropout=0.2,
    )

    # Class weights from train target only
    counts = np.bincount(y_train.to_numpy(), minlength=n_classes).astype("float64")
    eps = 1e-6
    cw = 1.0 / (counts + eps)
    cw = cw / max(1e-12, float(cw.mean()))
    cw_tensor = torch.tensor(cw, dtype=torch.float32)

    criterion = ClassWeightedFocalLoss(
        class_weights=cw_tensor, gamma=1.0, reduction="mean"
    )

    n_params = sum(p.numel() for p in model.parameters())
    base_lr = 2.0e-3 if n_params <= 250_000 else 1.0e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(MAX_EPOCHS, 100), eta_min=1e-5
    )

    # === TENSORS AND DATALOADERS ===
    X_train_t = torch.tensor(X_train.to_numpy(dtype=np.float32), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(dtype=np.int64), dtype=torch.long)
    X_val_t = torch.tensor(X_val.to_numpy(dtype=np.float32), dtype=torch.float32)
    y_val_t = torch.tensor(y_val.to_numpy(dtype=np.int64), dtype=torch.long)
    X_test_t = torch.tensor(X_test.to_numpy(dtype=np.float32), dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_ds = TensorDataset(X_val_t, y_val_t)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    print(
        f"Device: {device} | dense={len(dense_indices)}, sparse={len(sparse_indices)}, params={n_params}, classes={n_classes}"
    )

    # === TRAINING LOOP WITH EARLY STOPPING ===
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = train_loss_sum / max(1, n_batches)

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.append(preds)
        val_preds = (
            np.concatenate(val_preds) if val_preds else np.array([], dtype=np.int64)
        )
        val_acc = (
            accuracy_score(y_val.to_numpy(), val_preds)
            if len(val_preds) == len(y_val)
            else 0.0
        )

        print(
            f"Epoch {epoch+1}/{MAX_EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch+1}. Best val_acc={best_val_acc:.4f}"
                )
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # === FINAL VALIDATION SCORE (official metric: accuracy) ===
    val_preds_final = []
    with torch.no_grad():
        for xb, _ in val_dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds_final.append(preds)
    val_preds_final = (
        np.concatenate(val_preds_final)
        if val_preds_final
        else np.zeros(len(y_val), dtype=np.int64)
    )
    score = accuracy_score(y_val.to_numpy(), val_preds_final)
    print(f"Best Validation Accuracy: {score:.6f}")

    # === TEST INFERENCE ===
    test_preds = []
    test_batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test_t), test_batch_size):
            xb = X_test_t[i : i + test_batch_size].to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.append(preds)
    test_preds = (
        np.concatenate(test_preds)
        if test_preds
        else np.zeros(len(test_ids), dtype=np.int64)
    )

    # Map integer predictions back to original target strings
    if int_to_target:
        pred_labels = np.array(
            [
                str(int_to_target.get(int(p), int_to_target.get(0, "unknown")))
                for p in test_preds
            ]
        )
    else:
        pred_labels = test_preds.astype(str)

    # === SAVE SUBMISSION ===
    submission = pd.DataFrame(
        {"id": test_ids.astype(np.int64), "availability": pred_labels}
    )
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved: {len(submission)} rows")

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()