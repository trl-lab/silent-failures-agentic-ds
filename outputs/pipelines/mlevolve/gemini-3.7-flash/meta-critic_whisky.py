import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 0. Reproducibility Setup
# ---------------------------------------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ---------------------------------------------------------
# 1. Directory Setup & Data Loading
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Meta Critic"


# ---------------------------------------------------------
# 2. Domain-Specific Cleaning & Parsing Functions
# ---------------------------------------------------------
def parse_abv(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    if match:
        num = float(match.group(1))
        if num < 1.0:
            num *= 100.0
        return num
    return np.nan


def parse_cost(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if set(val_str) <= {"$", " "}:
        return float(len(val_str.replace(" ", "")))
    match = re.search(r"(\d+(?:\.\d+)?)", val_str)
    if match:
        return float(match.group(1))
    return np.nan


def parse_age(val):
    if pd.isna(val):
        return np.nan, 1
    if isinstance(val, (int, float)):
        return float(val), 0
    val_str = str(val).lower().strip()
    if "nas" in val_str or "no age" in val_str or "non age" in val_str or val_str == "":
        return 0.0, 1
    match = re.search(r"(\d+)", val_str)
    if match:
        return float(match.group(1)), 0
    return np.nan, 1


# ---------------------------------------------------------
# 3. Column Normalization & Feature Extraction
# ---------------------------------------------------------
all_cols = list(train_df.columns)
id_col = "id" if "id" in test_df.columns else None

abv_cols = [c for c in all_cols if "abv" in c.lower() or "alcohol" in c.lower()]
cost_cols = [c for c in all_cols if "cost" in c.lower() or "price" in c.lower()]
age_cols = [
    c
    for c in all_cols
    if "age" in c.lower() and "vintage" not in c.lower() and c != target_col
]
text_cols = [
    c
    for c in all_cols
    if c
    in [
        "Whisky",
        "Name",
        "Description",
        "Review",
        "Notes",
        "whisky",
        "name",
        "description",
    ]
]


def extract_features(df):
    feat_df = pd.DataFrame(index=df.index)

    if abv_cols:
        col = abv_cols[0]
        feat_df["parsed_abv"] = df[col].apply(parse_abv)
        feat_df["is_cask_strength"] = (feat_df["parsed_abv"] >= 50.0).astype(float)
        feat_df["is_standard_40"] = (
            (feat_df["parsed_abv"] >= 39.5) & (feat_df["parsed_abv"] <= 40.5)
        ).astype(float)
        feat_df["is_standard_43"] = (
            (feat_df["parsed_abv"] >= 42.5) & (feat_df["parsed_abv"] <= 43.5)
        ).astype(float)
        feat_df["is_standard_46"] = (
            (feat_df["parsed_abv"] >= 45.5) & (feat_df["parsed_abv"] <= 46.5)
        ).astype(float)

    if cost_cols:
        col = cost_cols[0]
        feat_df["parsed_cost"] = df[col].apply(parse_cost)
        feat_df["log_cost"] = np.log1p(feat_df["parsed_cost"].clip(lower=0).fillna(0))

    if age_cols:
        col = age_cols[0]
        parsed_age_nas = df[col].apply(parse_age)
        feat_df["parsed_age"] = [p[0] for p in parsed_age_nas]
        feat_df["is_nas"] = [p[1] for p in parsed_age_nas]
        feat_df["log_age"] = np.log1p(feat_df["parsed_age"].clip(lower=0).fillna(0))

    existing_numeric = df.select_dtypes(include=[np.number]).columns.drop(
        [target_col, id_col], errors="ignore"
    )
    for col in existing_numeric:
        if col not in feat_df.columns:
            feat_df[f"num_{col}"] = df[col].astype(float)

    if "parsed_cost" in feat_df.columns and "parsed_age" in feat_df.columns:
        feat_df["price_per_age"] = feat_df["parsed_cost"] / (feat_df["parsed_age"].fillna(0) + 1.0)
        feat_df["log_cost_age_ratio"] = feat_df["log_cost"] / (feat_df["log_age"] + 1.0)
        feat_df["cost_age_mult"] = feat_df["parsed_cost"] * (feat_df["parsed_age"].fillna(0) + 1.0)

    if "parsed_cost" in feat_df.columns and "parsed_abv" in feat_df.columns:
        feat_df["cost_per_abv"] = feat_df["parsed_cost"] / (feat_df["parsed_abv"].replace(0, np.nan).fillna(40.0))

    if "parsed_age" in feat_df.columns and "parsed_abv" in feat_df.columns:
        feat_df["age_abv_interaction"] = feat_df["parsed_age"].fillna(0) * feat_df["parsed_abv"].fillna(40.0)

    num_cols_subset = [
        c
        for c in feat_df.columns
        if feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        and not c.startswith("is_")
    ]
    if len(num_cols_subset) >= 3:
        feat_df["numeric_sum"] = feat_df[num_cols_subset].sum(axis=1)
        feat_df["numeric_mean"] = feat_df[num_cols_subset].mean(axis=1)
        feat_df["numeric_std"] = feat_df[num_cols_subset].std(axis=1).fillna(0)
        feat_df["numeric_max"] = feat_df[num_cols_subset].max(axis=1)
        feat_df["numeric_min"] = feat_df[num_cols_subset].min(axis=1)

    return feat_df


train_features = extract_features(train_df)
test_features = extract_features(test_df)

# ---------------------------------------------------------
# 4. Categorical Encodings (Frequency & Interactions)
# ---------------------------------------------------------
cat_cols = (
    train_df.select_dtypes(include=["object", "category"])
    .columns.drop([c for c in text_cols if c in train_df.columns], errors="ignore")
    .drop([target_col, id_col], errors="ignore")
)

for col in cat_cols:
    train_series = train_df[col].astype(str).fillna("Missing")
    test_series = test_df[col].astype(str).fillna("Missing")

    freq_map = train_series.value_counts(normalize=True).to_dict()
    train_features[f"{col}_freq"] = train_series.map(freq_map).fillna(0).astype(float)
    test_features[f"{col}_freq"] = test_series.map(freq_map).fillna(0).astype(float)

    unique_cats = {cat: idx for idx, cat in enumerate(train_series.unique(), start=1)}
    train_features[f"{col}_cat_code"] = (
        train_series.map(unique_cats).fillna(0).astype(int)
    )
    test_features[f"{col}_cat_code"] = (
        test_series.map(unique_cats).fillna(0).astype(int)
    )

    for num_f in ["parsed_cost", "parsed_age", "parsed_abv"]:
        if num_f in train_features.columns:
            temp_df = pd.DataFrame({"cat": train_series, "val": train_features[num_f]})
            grp = temp_df.groupby("cat")["val"]
            mean_map = grp.mean().to_dict()
            std_map = grp.std().fillna(0).to_dict()
            min_map = grp.min().to_dict()
            max_map = grp.max().to_dict()

            global_median = float(train_features[num_f].median()) if not pd.isna(train_features[num_f].median()) else 0.0
            train_features[f"{col}_{num_f}_mean"] = train_series.map(mean_map).fillna(global_median).astype(float)
            test_features[f"{col}_{num_f}_mean"] = test_series.map(mean_map).fillna(global_median).astype(float)
            train_features[f"{col}_{num_f}_std"] = train_series.map(std_map).fillna(0).astype(float)
            test_features[f"{col}_{num_f}_std"] = test_series.map(std_map).fillna(0).astype(float)
            train_features[f"{col}_{num_f}_min"] = train_series.map(min_map).fillna(global_median).astype(float)
            test_features[f"{col}_{num_f}_min"] = test_series.map(min_map).fillna(global_median).astype(float)
            train_features[f"{col}_{num_f}_max"] = train_series.map(max_map).fillna(global_median).astype(float)
            test_features[f"{col}_{num_f}_max"] = test_series.map(max_map).fillna(global_median).astype(float)

# ---------------------------------------------------------
# 5. Text Feature Extraction (TF-IDF + SVD)
# ---------------------------------------------------------
for col in text_cols:
    if col in train_df.columns and col in test_df.columns:
        train_text = train_df[col].fillna("").astype(str)
        test_text = test_df[col].fillna("").astype(str)

        train_features[f"{col}_char_len"] = train_text.str.len().astype(float)
        test_features[f"{col}_char_len"] = test_text.str.len().astype(float)
        train_features[f"{col}_word_count"] = train_text.apply(
            lambda x: len(x.split())
        ).astype(float)
        test_features[f"{col}_word_count"] = test_text.apply(
            lambda x: len(x.split())
        ).astype(float)

        tfidf = TfidfVectorizer(
            max_features=250,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=2,
        )
        tfidf_train = tfidf.fit_transform(train_text)
        tfidf_test = tfidf.transform(test_text)

        n_comp = min(10, tfidf_train.shape[1] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_train = svd.fit_transform(tfidf_train)
            svd_test = svd.transform(tfidf_test)

            for comp_idx in range(n_comp):
                train_features[f"{col}_svd_{comp_idx}"] = svd_train[:, comp_idx]
                test_features[f"{col}_svd_{comp_idx}"] = svd_test[:, comp_idx]

# ---------------------------------------------------------
# 6. Leak-Free Out-Of-Fold Target Encoding & Text Ridge Meta-Feature
# ---------------------------------------------------------
y_train = train_df[target_col].values
n_splits = 10

binned_target = pd.qcut(
    y_train, q=min(10, len(np.unique(y_train))), labels=False, duplicates="drop"
)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for col in cat_cols:
    train_series = train_df[col].astype(str).fillna("Missing")
    test_series = test_df[col].astype(str).fillna("Missing")

    oof_te = np.zeros(len(train_df))
    global_mean = y_train.mean()

    for tr_idx, val_idx in skf.split(train_features, binned_target):
        tr_cat = train_series.iloc[tr_idx]
        tr_y = y_train[tr_idx]

        stats = pd.DataFrame({"cat": tr_cat, "target": tr_y}).groupby("cat")["target"]
        means = stats.mean()
        counts = stats.count()
        smooth_weight = 10.0
        smoothed = (means * counts + global_mean * smooth_weight) / (
            counts + smooth_weight
        )

        oof_te[val_idx] = (
            train_series.iloc[val_idx].map(smoothed).fillna(global_mean).values
        )

    full_stats = pd.DataFrame({"cat": train_series, "target": y_train}).groupby("cat")[
        "target"
    ]
    full_means = full_stats.mean()
    full_counts = full_stats.count()
    full_smoothed = (full_means * full_counts + global_mean * 10.0) / (
        full_counts + 10.0
    )

    train_features[f"{col}_target_enc"] = oof_te
    test_features[f"{col}_target_enc"] = (
        test_series.map(full_smoothed).fillna(global_mean).values
    )

available_text_cols = [c for c in text_cols if c in train_df.columns and c in test_df.columns]
if available_text_cols:
    train_combined_text = train_df[available_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    test_combined_text = test_df[available_text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    oof_text_ridge = np.zeros(len(train_df))
    test_text_ridge = np.zeros(len(test_df))

    for tr_idx, val_idx in skf.split(train_features, binned_target):
        tfidf_meta = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        X_tr_text = tfidf_meta.fit_transform(train_combined_text.iloc[tr_idx])
        X_val_text = tfidf_meta.transform(train_combined_text.iloc[val_idx])
        X_test_text = tfidf_meta.transform(test_combined_text)

        ridge_meta = Ridge(alpha=5.0, random_state=42)
        ridge_meta.fit(X_tr_text, y_train[tr_idx])
        oof_text_ridge[val_idx] = ridge_meta.predict(X_val_text)
        test_text_ridge += ridge_meta.predict(X_test_text) / n_splits

    train_features["oof_text_ridge_pred"] = oof_text_ridge
    test_features["oof_text_ridge_pred"] = test_text_ridge

# ---------------------------------------------------------
# 7. Imputation & Fold Assignment
# ---------------------------------------------------------
common_cols = [c for c in train_features.columns if c in test_features.columns]
X_train_df = train_features[common_cols].copy()
X_test_df = test_features[common_cols].copy()

medians = X_train_df.median(numeric_only=True)
X_train_df = X_train_df.fillna(medians).fillna(0)
X_test_df = X_test_df.fillna(medians).fillna(0)

fold_assignments = np.zeros(len(X_train_df), dtype=int)
for fold_num, (tr_idx, val_idx) in enumerate(skf.split(X_train_df, binned_target)):
    fold_assignments[val_idx] = fold_num

X_train_df["fold"] = fold_assignments
X_train_df[target_col] = y_train

X_train_df.to_parquet("./working/train_processed.parquet", index=False)
X_test_df.to_parquet("./working/test_processed.parquet", index=False)


# ---------------------------------------------------------
# 8. Model Architectures & Builders
# ---------------------------------------------------------
def get_lgbm_model(
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(
    n_estimators: int = 1500,
    learning_rate: float = 0.03,
    max_depth: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def get_catboost_model(
    iterations: int = 1500,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 42,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=False,
        thread_count=-1,
    )


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def build_torch_model_and_optimizer(
    input_dim: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.15,
    t_max: int = 80,
):
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=1e-6
    )
    return model, criterion, optimizer, scheduler


# ---------------------------------------------------------
# 9. Multi-Model K-Fold Training & Cross-Validation
# ---------------------------------------------------------
feature_cols = [c for c in X_train_df.columns if c not in [target_col, "fold", "id"]]
X = X_train_df[feature_cols].values
y = X_train_df[target_col].values
folds = X_train_df["fold"].values
X_test_raw = X_test_df[feature_cols].values

n_folds = len(np.unique(folds))
n_samples = len(X)
n_test = len(X_test_raw)
input_dim = len(feature_cols)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cb = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)
oof_ridge = np.zeros(n_samples)
oof_et = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cb = np.zeros(n_test)
test_nn = np.zeros(n_test)
test_ridge = np.zeros(n_test)
test_et = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(n_folds):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # --- A. LightGBM ---
    lgb_model = get_lgbm_model(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test_raw) / n_folds

    # --- B. XGBoost ---
    xgb_model = get_xgb_model(
        n_estimators=1500, learning_rate=0.03, max_depth=5, random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb += xgb_model.predict(X_test_raw) / n_folds

    # --- C. CatBoost ---
    cb_model = get_catboost_model(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
    )
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False
    )
    oof_cb[val_idx] = cb_model.predict(X_val)
    test_cb += cb_model.predict(X_test_raw) / n_folds

    # --- D. PyTorch TabularResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_raw)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(64, len(X_tr)),
        shuffle=True,
        drop_last=False,
    )

    nn_model, criterion, optimizer, scheduler = build_torch_model_and_optimizer(
        input_dim=input_dim,
        lr=2e-3,
        weight_decay=1e-4,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.15,
        t_max=80,
    )
    nn_model.to(device)

    best_val_rmse = float("inf")
    best_weights = None
    patience, patience_counter = 15, 0

    for epoch in range(80):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            val_preds_t = nn_model(val_tensor).cpu().numpy()
            val_rmse_epoch = root_mean_squared_error(y_val, val_preds_t)

        if val_rmse_epoch < best_val_rmse:
            best_val_rmse = val_rmse_epoch
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        oof_nn[val_idx] = nn_model(val_tensor).cpu().numpy()
        test_nn += nn_model(test_tensor).cpu().numpy() / n_folds

    # --- E. Ridge Regression ---
    ridge_model = Ridge(alpha=10.0, random_state=42 + fold)
    ridge_model.fit(X_tr_scaled, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_val_scaled)
    test_ridge += ridge_model.predict(X_test_scaled) / n_folds

    # --- F. ExtraTreesRegressor ---
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict(X_val)
    test_et += et_model.predict(X_test_raw) / n_folds

# ---------------------------------------------------------
# 10. Out-of-Fold Evaluation & Optimal Ensemble Blending
# ---------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn, oof_ridge, oof_et])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_nn, test_ridge, test_et])


def objective_rmse(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_oof = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y, blend_oof)


init_weights = [1.0 / 6.0] * 6
bounds = [(0, 1)] * 6
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective_rmse,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.array(init_weights)
)

final_oof_preds = np.dot(oof_matrix, best_weights)
final_test_preds = np.dot(test_matrix, best_weights)

final_validation_score = root_mean_squared_error(y, final_oof_preds)

# ---------------------------------------------------------
# 11. Generate Submission File & Integrity Verifications
# ---------------------------------------------------------
submission_path = "./submission/submission.csv"

if "id" in test_df.columns:
    sub_id = test_df["id"].values
else:
    sub_id = np.arange(len(test_df))

sub_df = pd.DataFrame(
    {"id": sub_id.astype(int), target_col: final_test_preds.astype(float)}
)

sub_df.to_csv(submission_path, index=False)

assert os.path.exists(
    submission_path
), f"Submission file {submission_path} was not created!"
assert sub_df.shape[0] == len(
    test_df
), f"Row count mismatch: expected {len(test_df)}, got {sub_df.shape[0]}"
assert list(sub_df.columns) == [
    "id",
    target_col,
], f"Columns mismatch: expected ['id', '{target_col}'], got {list(sub_df.columns)}"
assert not sub_df[target_col].isna().any(), "Submission target contains NaN values!"
assert not np.isinf(
    sub_df[target_col]
).any(), "Submission target contains infinite values!"

print(f"Final Validation Score: {final_validation_score}")
