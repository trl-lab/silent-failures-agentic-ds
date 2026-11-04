import os
import re
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# 0. Reproducibility & Environment Setup
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data Loading & Target Extraction
# ---------------------------------------------------------------------------
print("Loading raw parquet datasets...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "marketing_status"
if target_col not in train_df.columns:
    candidates = [c for c in train_df.columns if c.lower() == "marketing_status"]
    if candidates:
        target_col = candidates[0]
    else:
        raise ValueError(
            f"Target column '{target_col}' not found in train columns: {train_df.columns.tolist()}"
        )

id_col = "id" if "id" in train_df.columns else None

# Clean target to standard binary integers
y_raw = train_df[target_col]
if (
    y_raw.dtype == "object"
    or str(y_raw.dtype).startswith("category")
    or str(y_raw.dtype) == "bool"
):
    unique_vals = y_raw.dropna().unique()
    val_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
    y = y_raw.map(val_map).astype(int)
else:
    y = y_raw.astype(int)

train_df["target"] = y
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts(normalize=True)}")


# ---------------------------------------------------------------------------
# 2. Feature Extraction Helpers
# ---------------------------------------------------------------------------
def extract_numerical_dosage(text_series: pd.Series) -> pd.DataFrame:
    """Extract numeric strength, unit flags, and dosage counts from text."""
    s = text_series.fillna("").astype(str).str.lower()
    num_regex = re.compile(r"(\d+(?:\.\d+)?)")
    val1 = s.apply(
        lambda x: float(num_regex.search(x).group(1)) if num_regex.search(x) else np.nan
    )

    has_mg = s.str.contains(r"\bmg\b|/mg", regex=True).astype(np.int8)
    has_mcg = s.str.contains(r"\bmcg\b|\bug\b", regex=True).astype(np.int8)
    has_g = s.str.contains(r"\bg\b|\bgm\b", regex=True).astype(np.int8)
    has_ml = s.str.contains(r"\bml\b", regex=True).astype(np.int8)
    has_pct = s.str.contains(r"%", regex=True).astype(np.int8)
    has_eq = s.str.contains(r"\beq\b|equivalent", regex=True).astype(np.int8)

    return pd.DataFrame(
        {
            "dosage_val_primary": val1,
            "dosage_has_mg": has_mg,
            "dosage_has_mcg": has_mcg,
            "dosage_has_g": has_g,
            "dosage_has_ml": has_ml,
            "dosage_has_pct": has_pct,
            "dosage_has_eq": has_eq,
        }
    )


def extract_text_properties(text_series: pd.Series, prefix: str) -> pd.DataFrame:
    """Extract length, word counts, and character patterns from string columns."""
    s = text_series.fillna("").astype(str)
    return pd.DataFrame(
        {
            f"{prefix}_char_len": s.str.len().astype(np.int32),
            f"{prefix}_word_count": s.apply(lambda x: len(x.split())).astype(np.int16),
            f"{prefix}_digit_count": s.apply(
                lambda x: sum(c.isdigit() for c in x)
            ).astype(np.int16),
            f"{prefix}_has_slash": s.str.contains("/").astype(np.int8),
            f"{prefix}_has_dash": s.str.contains("-").astype(np.int8),
        }
    )


def prepare_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Extract structural, domain, and text-property features."""
    feats = pd.DataFrame(index=df.index)
    non_feature_cols = {target_col, "target", "id"}
    candidate_cols = [c for c in df.columns if c not in non_feature_cols]

    categorical_cols = []
    text_cols = []
    numeric_cols = []
    date_cols = []

    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
        else:
            sample = df[c].dropna().astype(str).head(50)
            is_date = False
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors="raise")
                    is_date = True
                except Exception:
                    is_date = False
            if is_date:
                date_cols.append(c)
            else:
                cardinality = df[c].nunique(dropna=True)
                if cardinality > 500:
                    text_cols.append(c)
                categorical_cols.append(c)

    # Numeric features
    for col in numeric_cols:
        feats[f"num_{col}"] = df[col].astype(np.float32)
        feats[f"num_{col}_isnan"] = df[col].isna().astype(np.int8)

    # Date features
    for col in date_cols:
        d = pd.to_datetime(df[col], errors="coerce")
        feats[f"date_{col}_year"] = d.dt.year.fillna(-1).astype(np.int16)
        feats[f"date_{col}_month"] = d.dt.month.fillna(-1).astype(np.int8)
        feats[f"date_{col}_day"] = d.dt.day.fillna(-1).astype(np.int8)
        feats[f"date_{col}_dayofweek"] = d.dt.dayofweek.fillna(-1).astype(np.int8)
        feats[f"date_{col}_is_na"] = d.isna().astype(np.int8)
        feats[f"date_{col}_years_since_2000"] = ((d.dt.year - 2000).fillna(-99)).astype(
            np.float32
        )

    # Categorical and string features
    for col in categorical_cols:
        s_clean = df[col].fillna("__MISSING__").astype(str).str.strip().str.lower()
        feats[f"cat_{col}_clean"] = s_clean

        prop_df = extract_text_properties(df[col], f"prop_{col}")
        for prop_c in prop_df.columns:
            feats[prop_c] = prop_df[prop_c]

        if any(
            keyword in col.lower()
            for keyword in [
                "strength",
                "dosage",
                "form",
                "active",
                "substance",
                "potency",
            ]
        ):
            dosage_df = extract_numerical_dosage(df[col])
            for d_c in dosage_df.columns:
                feats[f"{col}_{d_c}"] = dosage_df[d_c]

    # Pairwise interactions
    top_cats = [c for c in categorical_cols if df[c].nunique() < 2000][:4]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            feats[f"inter_{c1}_{c2}"] = (
                df[c1].fillna("__NA__").astype(str)
                + "___"
                + df[c2].fillna("__NA__").astype(str)
            ).astype("category")

    return feats, categorical_cols


# ---------------------------------------------------------------------------
# 3. Feature Engineering Execution
# ---------------------------------------------------------------------------
print("Extracting base features...")
train_feats, cat_cols = prepare_base_features(train_df)
test_feats, _ = prepare_base_features(test_df)

all_cols = sorted(list(set(train_feats.columns).intersection(set(test_feats.columns))))
train_feats = train_feats[all_cols].copy()
test_feats = test_feats[all_cols].copy()

# Stratified Fold Assignment for strictly isolated Out-Of-Fold Encodings
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_df), dtype=int)
train_target = train_df["target"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(train_feats, train_target)):
    fold_assignments[val_idx] = fold

train_feats["fold"] = fold_assignments

cols_to_encode = [
    c for c in train_feats.columns if c.startswith("cat_") or c.startswith("inter_")
]
global_mean = train_target.mean()
smoothing = 10.0

for col in cols_to_encode:
    freq_map = train_feats[col].value_counts(normalize=True).to_dict()
    train_feats[f"{col}_freq"] = (
        train_feats[col].map(freq_map).fillna(0.0).astype(np.float32)
    )
    test_feats[f"{col}_freq"] = (
        test_feats[col].map(freq_map).fillna(0.0).astype(np.float32)
    )

    train_te = np.zeros(len(train_feats), dtype=np.float32)
    for fold in range(n_splits):
        val_mask = fold_assignments == fold
        trn_mask = ~val_mask

        trn_sub = pd.DataFrame(
            {"cat": train_feats.loc[trn_mask, col], "target": train_target[trn_mask]}
        )
        grouped = trn_sub.groupby("cat")["target"].agg(["count", "mean"])
        smoothed = (grouped["count"] * grouped["mean"] + smoothing * global_mean) / (
            grouped["count"] + smoothing
        )
        te_dict = smoothed.to_dict()
        train_te[val_mask] = (
            train_feats.loc[val_mask, col].map(te_dict).fillna(global_mean).values
        )

    train_feats[f"{col}_te"] = train_te

    full_grouped = (
        pd.DataFrame({"cat": train_feats[col], "target": train_target})
        .groupby("cat")["target"]
        .agg(["count", "mean"])
    )
    full_smoothed = (
        full_grouped["count"] * full_grouped["mean"] + smoothing * global_mean
    ) / (full_grouped["count"] + smoothing)
    test_feats[f"{col}_te"] = (
        test_feats[col]
        .map(full_smoothed.to_dict())
        .fillna(global_mean)
        .astype(np.float32)
    )

string_cols_to_drop = [
    c
    for c in cols_to_encode
    if train_feats[c].dtype == "object" or str(train_feats[c].dtype) == "category"
]
train_feats = train_feats.drop(columns=string_cols_to_drop)
test_feats = test_feats.drop(columns=string_cols_to_drop)

# Text TF-IDF + SVD Dimensionality Reduction
print("Generating TF-IDF and SVD representations...")
text_train_series = train_df[cat_cols].fillna("").astype(str).agg(" ".join, axis=1)
text_test_series = test_df[cat_cols].fillna("").astype(str).agg(" ".join, axis=1)

tfidf = TfidfVectorizer(
    max_features=2500, ngram_range=(1, 2), min_df=3, sublinear_tf=True
)
tfidf.fit(text_train_series)

svd_dim = 16
svd = TruncatedSVD(n_components=svd_dim, random_state=42)
train_svd = svd.fit_transform(tfidf.transform(text_train_series))
test_svd = svd.transform(tfidf.transform(text_test_series))

for d in range(svd_dim):
    train_feats[f"svd_tfidf_{d}"] = train_svd[:, d].astype(np.float32)
    test_feats[f"svd_tfidf_{d}"] = test_svd[:, d].astype(np.float32)

# Imputation using training medians strictly
train_cols = [c for c in train_feats.columns if c not in ["fold", "target"]]
for col in train_cols:
    med = train_feats[col].median()
    if np.isnan(med):
        med = 0.0
    train_feats[col] = train_feats[col].fillna(med)
    test_feats[col] = test_feats[col].fillna(med)

train_feats["target"] = train_target
if id_col and id_col in train_df.columns:
    train_feats["id"] = train_df[id_col].values
    test_feats["id"] = test_df[id_col].values
else:
    train_feats["id"] = np.arange(len(train_feats))
    test_feats["id"] = np.arange(len(test_feats))

# Validation Holdout Split (Fold 0)
val_mask = train_feats["fold"] == 0
val_feats = train_feats[val_mask].copy().reset_index(drop=True)
train_split = train_feats[~val_mask].copy().reset_index(drop=True)

feature_cols = [c for c in train_split.columns if c not in ["id", "target", "fold"]]

X_train = train_split[feature_cols].copy().values.astype(np.float32)
y_train = train_split["target"].values.astype(np.float32)

X_val = val_feats[feature_cols].copy().values.astype(np.float32)
y_val = val_feats["target"].values.astype(np.float32)

X_test = test_feats[feature_cols].copy().values.astype(np.float32)
test_ids = test_feats["id"].values

print(
    f"Features: {len(feature_cols)}, Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}"
)


# ---------------------------------------------------------------------------
# 4. Model Architectures & Definitions
# ---------------------------------------------------------------------------
def get_lgbm_model(random_state: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        importance_type="gain",
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 5. Model Training & Evaluation
# ---------------------------------------------------------------------------
# 5.1 LightGBM
print("Training LightGBM model...")
lgb_clf = get_lgbm_model(random_state=42)
lgb_clf.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
val_preds_lgb = lgb_clf.predict_proba(X_val)[:, 1]
test_preds_lgb = lgb_clf.predict_proba(X_test)[:, 1]
auc_lgb = roc_auc_score(y_val, val_preds_lgb)
print(f"LightGBM Validation ROC-AUC: {auc_lgb:.5f}")

# 5.2 XGBoost
print("Training XGBoost model...")
xgb_clf = get_xgb_model(random_state=42)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
val_preds_xgb = xgb_clf.predict_proba(X_val)[:, 1]
test_preds_xgb = xgb_clf.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_val, val_preds_xgb)
print(f"XGBoost Validation ROC-AUC: {auc_xgb:.5f}")

# 5.3 CatBoost
print("Training CatBoost model...")
cat_clf = get_catboost_model(random_state=42)
cat_clf.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False,
)
val_preds_cat = cat_clf.predict_proba(X_val)[:, 1]
test_preds_cat = cat_clf.predict_proba(X_test)[:, 1]
auc_cat = roc_auc_score(y_val, val_preds_cat)
print(f"CatBoost Validation ROC-AUC: {auc_cat:.5f}")

# 5.4 PyTorch TabularResNet
print("Training PyTorch TabularResNet...")
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(
    torch.tensor(X_train_norm, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_norm, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=2
)

nn_model = TabularResNet(
    input_dim=X_train.shape[1], hidden_dim=256, num_blocks=3, dropout_rate=0.25
).to(device)
criterion = nn.BCEWithLogitsLoss()
epochs = 35
optimizer = torch.optim.AdamW(
    nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

best_nn_val_auc = 0.0
best_model_state = None

for epoch in range(1, epochs + 1):
    nn_model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(batch_y)

    scheduler.step()

    nn_model.eval()
    val_preds_epoch = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            logits = nn_model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds_epoch.extend(probs)

    epoch_val_auc = roc_auc_score(y_val, val_preds_epoch)
    if epoch_val_auc > best_nn_val_auc:
        best_nn_val_auc = epoch_val_auc
        best_model_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

if best_model_state is not None:
    nn_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

nn_model.eval()
with torch.no_grad():
    val_preds_nn = []
    for batch_x, _ in val_loader:
        logits = nn_model(batch_x.to(device))
        val_preds_nn.extend(torch.sigmoid(logits).cpu().numpy())
    val_preds_nn = np.array(val_preds_nn)

    test_logits = nn_model(test_tensor)
    test_preds_nn = torch.sigmoid(test_logits).cpu().numpy()

auc_nn = roc_auc_score(y_val, val_preds_nn)
print(f"TabularResNet Validation ROC-AUC: {auc_nn:.5f}")

# ---------------------------------------------------------------------------
# 6. Ensemble Optimization & Final Predictions
# ---------------------------------------------------------------------------
val_matrix = np.column_stack(
    [val_preds_lgb, val_preds_xgb, val_preds_cat, val_preds_nn]
)
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn]
)


def loss_func(weights):
    w = np.maximum(0, weights)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    blend = np.dot(val_matrix, w)
    return -roc_auc_score(y_val, blend)


init_weights = np.array([0.35, 0.30, 0.25, 0.10])
bounds = [(0.0, 1.0)] * 4
opt_res = minimize(
    loss_func, init_weights, method="SLSQP", bounds=bounds, options={"maxiter": 200}
)

optimal_weights = np.maximum(0, opt_res.x)
optimal_weights = optimal_weights / optimal_weights.sum()
print(
    f"Optimal Blend Weights: LGB={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, CAT={optimal_weights[2]:.3f}, NN={optimal_weights[3]:.3f}"
)

val_ensemble_preds = np.dot(val_matrix, optimal_weights)
test_ensemble_preds = np.dot(test_matrix, optimal_weights)

final_score = roc_auc_score(y_val, val_ensemble_preds)

# ---------------------------------------------------------------------------
# 7. Generate Submission CSV
# ---------------------------------------------------------------------------
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df = pd.DataFrame(
    {"id": test_ids.astype(int), "marketing_status": test_ensemble_preds}
)
sub_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(sub_df)} rows.")

print(f"Final Validation Score: {final_score}")
