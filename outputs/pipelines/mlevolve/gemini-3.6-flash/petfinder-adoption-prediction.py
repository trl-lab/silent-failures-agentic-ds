import os
import random
import numpy as np
import pandas as pd
import scipy as sp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score


# 0. Reproducibility & Environment Setup
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# 1. Load Parquet Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. Setup Stratified K-Fold Splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["AdoptionSpeed"])
):
    train_df.loc[val_idx, "fold"] = fold

# Global rescuer and breed frequency mappings from training set
rescuer_counts = train_df["RescuerID"].value_counts().to_dict()
breed1_counts = train_df["Breed1"].value_counts().to_dict()

# Group Aggregations Setup across Breed1, RescuerID, and State (fit on train set only to prevent leakage)
train_df_copy = train_df.copy()
train_df_copy["MedicalCareScore"] = (
    (train_df_copy["Vaccinated"] == 1).astype(int)
    + (train_df_copy["Dewormed"] == 1).astype(int)
    + (train_df_copy["Sterilized"] == 1).astype(int)
)

group_cols = ["Breed1", "RescuerID", "State"]
agg_num_cols = ["Age", "Fee", "PhotoAmt", "MedicalCareScore"]
agg_funcs = ["mean", "std", "min", "max"]

group_stats_dfs = {}
for g_col in group_cols:
    grouped = train_df_copy.groupby(g_col)[agg_num_cols].agg(agg_funcs)
    grouped.columns = [f"{g_col}_{col}_{func}" for col, func in grouped.columns]
    group_stats_dfs[g_col] = grouped


# 3. Domain Feature Extraction
def extract_domain_features(df):
    features = pd.DataFrame(index=df.index)

    # Metadata & Numeric
    features["Type"] = df["Type"]
    features["Age"] = df["Age"]
    features["PetAgeYears"] = df["Age"] / 12.0
    features["IsPuppyOrKitten"] = (df["Age"] <= 3).astype(int)
    features["IsSenior"] = (df["Age"] >= 84).astype(int)
    features["Breed1"] = df["Breed1"]
    features["Breed2"] = df["Breed2"]
    features["IsPurebred"] = (
        (df["Breed2"] == 0) | (df["Breed2"] == df["Breed1"])
    ).astype(int)
    features["Gender"] = df["Gender"]
    features["Color1"] = df["Color1"]
    features["Color2"] = df["Color2"]
    features["Color3"] = df["Color3"]
    features["ColorCount"] = (
        (df["Color1"] > 0).astype(int)
        + (df["Color2"] > 0).astype(int)
        + (df["Color3"] > 0).astype(int)
    )
    features["MaturitySize"] = df["MaturitySize"]

    features["FurLength"] = df["FurLength"]
    features["Vaccinated"] = df["Vaccinated"]
    features["Dewormed"] = df["Dewormed"]
    features["Sterilized"] = df["Sterilized"]
    features["Health"] = df["Health"]
    features["Quantity"] = df["Quantity"]
    features["Fee"] = df["Fee"]
    features["IsFree"] = (df["Fee"] == 0).astype(int)
    features["State"] = df["State"]
    features["VideoAmt"] = df["VideoAmt"]
    features["PhotoAmt"] = df["PhotoAmt"]

    # Composite health/care features
    features["MedicalCareScore"] = (
        (df["Vaccinated"] == 1).astype(int)
        + (df["Dewormed"] == 1).astype(int)
        + (df["Sterilized"] == 1).astype(int)
    )
    features["MedicalCareRatio"] = features["MedicalCareScore"] / 3.0
    features["MedicalUnsureCount"] = (
        (df["Vaccinated"] == 3).astype(int)
        + (df["Dewormed"] == 3).astype(int)
        + (df["Sterilized"] == 3).astype(int)
    )

    features["LogAge"] = np.log1p(df["Age"])
    features["LogFee"] = np.log1p(df["Fee"])

    # Composite breed/color combinations
    features["BreedCombo"] = df["Breed1"] * 1000 + df["Breed2"]
    features["ColorCombo"] = df["Color1"] * 100 + df["Color2"]
    features["Breed1_Color1"] = df["Breed1"] * 1000 + df["Color1"]

    features["FeePerPet"] = df["Fee"] / np.maximum(df["Quantity"], 1)
    features["PhotoPerPet"] = df["PhotoAmt"] / np.maximum(df["Quantity"], 1)

    # Merge Group Aggregations and Relative Deviations
    for g_col, g_df in group_stats_dfs.items():
        merged = df[[g_col]].merge(g_df, on=g_col, how="left").drop(columns=[g_col])
        merged.index = df.index
        for col in merged.columns:
            features[col] = merged[col].fillna(0.0)

        for num_col in agg_num_cols:
            mean_col = f"{g_col}_{num_col}_mean"
            if mean_col in features.columns:
                features[f"{num_col}_diff_{g_col}"] = features[num_col] - features[mean_col]

    # Text length & lexical richness metrics
    names = df["Name"].fillna("").astype(str)
    descs = df["Description"].fillna("").astype(str)

    features["HasName"] = (
        ~df["Name"].isna()
        & ~names.str.lower().str.contains(
            "no name|unnamed|baby|pup|kitten|cat|dog|^$", regex=True
        )
    ).astype(int)
    features["NameLen"] = names.apply(len)
    features["DescLen"] = descs.apply(len)
    features["DescWordCount"] = descs.apply(lambda x: len(x.split()))
    features["DescDigitCount"] = descs.apply(lambda x: sum(c.isdigit() for c in x))
    features["DescUppercaseRatio"] = descs.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1.0)
    )

    # Frequency Encodings
    features["RescuerFrequency"] = df["RescuerID"].map(rescuer_counts).fillna(1)
    features["Breed1Frequency"] = df["Breed1"].map(breed1_counts).fillna(1)

    return features


train_feats = extract_domain_features(train_df)
test_feats = extract_domain_features(test_df)

# 4. Out-of-Fold Target Encoding (Leak-Free)
global_mean = train_df["AdoptionSpeed"].mean()
smoothing_weight = 10.0
target_cols = ["RescuerID", "Breed1", "Breed2", "Color1", "State"]

for col in target_cols:
    train_feats[f"{col}_te"] = np.nan

for fold in range(5):
    tr_idx = train_df["fold"] != fold
    va_idx = train_df["fold"] == fold

    tr_subset = train_df[tr_idx]

    for col in target_cols:
        stats = tr_subset.groupby(col)["AdoptionSpeed"].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing_weight * global_mean) / (
            stats["count"] + smoothing_weight
        )
        mapping = smoothed.to_dict()
        train_feats.loc[va_idx, f"{col}_te"] = (
            train_df.loc[va_idx, col].map(mapping).fillna(global_mean)
        )

# Global Target Encoding for Test Set
for col in target_cols:
    stats = train_df.groupby(col)["AdoptionSpeed"].agg(["count", "mean"])
    smoothed = (stats["count"] * stats["mean"] + smoothing_weight * global_mean) / (
        stats["count"] + smoothing_weight
    )
    mapping = smoothed.to_dict()
    test_feats[f"{col}_te"] = test_df[col].map(mapping).fillna(global_mean)

# 5. Text TF-IDF + Dense Transformer Embedding Feature Extraction
train_text = (
    train_df["Name"].fillna("").astype(str)
    + " "
    + train_df["Description"].fillna("").astype(str)
).tolist()
test_text = (
    test_df["Name"].fillna("").astype(str)
    + " "
    + test_df["Description"].fillna("").astype(str)
).tolist()

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=50, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

svd_cols = [f"svd_{i}" for i in range(50)]
train_svd_df = pd.DataFrame(train_svd, columns=svd_cols, index=train_df.index)
test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_df.index)


# Transformer Embeddings Extraction with Graceful Fallback
@torch.no_grad()
def get_transformer_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    device_trans = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device_trans)
    model.eval()

    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = [t if (t and isinstance(t, str) and len(t.strip()) > 0) else "none" for t in texts[i : i + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device_trans)
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings_list.append(emb.cpu().numpy())
    return np.vstack(embeddings_list)


try:
    print("Extracting Transformer Embeddings...")
    train_dense_emb = get_transformer_embeddings(train_text)
    test_dense_emb = get_transformer_embeddings(test_text)

    emb_svd = TruncatedSVD(n_components=64, random_state=42)
    train_emb_svd = emb_svd.fit_transform(train_dense_emb)
    test_emb_svd = emb_svd.transform(test_dense_emb)

    emb_cols = [f"emb_svd_{i}" for i in range(64)]
    train_emb_df = pd.DataFrame(train_emb_svd, columns=emb_cols, index=train_df.index)
    test_emb_df = pd.DataFrame(test_emb_svd, columns=emb_cols, index=test_df.index)
except Exception as e:
    print(f"Transformer extraction fallback to expanded SVD features: {e}")
    emb_svd = TruncatedSVD(n_components=64, random_state=42)
    train_emb_svd = emb_svd.fit_transform(train_tfidf)
    test_emb_svd = emb_svd.transform(test_tfidf)
    emb_cols = [f"emb_svd_{i}" for i in range(64)]
    train_emb_df = pd.DataFrame(train_emb_svd, columns=emb_cols, index=train_df.index)
    test_emb_df = pd.DataFrame(test_emb_svd, columns=emb_cols, index=test_df.index)

# Combine Processed Feature DataFrames
X_train_full = (
    pd.concat([train_feats, train_svd_df, train_emb_df], axis=1)
    .copy()
    .fillna(0.0)
)
X_test_full = (
    pd.concat([test_feats, test_svd_df, test_emb_df], axis=1).copy().fillna(0.0)
)
y_train_full = train_df["AdoptionSpeed"].values
folds = train_df["fold"].values


# ==============================================================================
# OPTIMIZED ROUNDER FOR QUADRATIC WEIGHTED KAPPA
# ==============================================================================
class OptimizedRounder:
    """Optimizes decision thresholds for converting continuous regression outputs

    into discrete ordinal classes [0, 1, 2, 3, 4] targeting QWK directly.
    """

    def __init__(self, initial_coefficients=None):
        if initial_coefficients is None:
            self.coef_ = [0.5, 1.5, 2.5, 3.5]
        else:
            self.coef_ = list(initial_coefficients)

    def _kappa_loss(self, coef, X, y):
        sorted_coef = sorted(coef)
        X_p = np.digitize(X, sorted_coef)
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        loss_func = lambda coef: self._kappa_loss(coef, X, y)

        initial_candidates = [
            self.coef_,
            [0.5, 1.5, 2.5, 3.5],
            [0.6, 1.4, 2.4, 3.6],
            [0.4, 1.6, 2.6, 3.4],
        ]

        try:
            percentiles = np.percentile(X, [20, 40, 60, 80])
            initial_candidates.append(list(percentiles))

            probs = np.cumsum(np.bincount(y.astype(int), minlength=5) / len(y))[:-1]
            quantiles = np.quantile(X, probs)
            initial_candidates.append(list(quantiles))
        except Exception:
            pass

        best_loss = float("inf")
        best_coef = self.coef_

        for init_coef in initial_candidates:
            for method in ["Nelder-Mead", "Powell"]:
                res = sp.optimize.minimize(
                    loss_func, init_coef, method=method, options={"maxiter": 500}
                )
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_coef = sorted(res.x)

        self.coef_ = best_coef

    def predict(self, X):
        return np.digitize(X, self.coef_)


# ==============================================================================
# PYTORCH TABULAR RESNET ARCHITECTURE
# ==============================================================================
class ResidualBlock(nn.Module):
    """Residual Skip Block with LayerNorm, ReLU, and Dropout for tabular features."""

    def __init__(self, hidden_dim, dropout_rate=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet for Continuous Ordinal Feature Representations."""

    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout_rate=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


# ==============================================================================
# ENSEMBLE CROSS-VALIDATION TRAINING AND INFERENCE
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))
oof_nn = np.zeros(len(train_df))

test_lgb_preds = np.zeros(len(test_df))
test_xgb_preds = np.zeros(len(test_df))
test_cb_preds = np.zeros(len(test_df))
test_nn_preds = np.zeros(len(test_df))

num_folds = 5

for fold in range(num_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train_full.iloc[tr_idx], y_train_full[tr_idx]
    X_va, y_va = X_train_full.iloc[va_idx], y_train_full[va_idx]

    # --- 1. LightGBM Regressor ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    test_lgb_preds += lgb_model.predict(X_test_full) / num_folds

    # --- 2. XGBoost Regressor ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[va_idx] = xgb_model.predict(X_va)
    test_xgb_preds += xgb_model.predict(X_test_full) / num_folds

    # --- 3. CatBoost Regressor ---
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[va_idx] = cb_model.predict(X_va)
    test_cb_preds += cb_model.predict(X_test_full) / num_folds

    # --- 4. PyTorch Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test_full)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    test_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    nn_model = TabularResNet(input_dim=X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(25):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict(best_nn_weights)
    nn_model.to(device)
    nn_model.eval()

    val_preds_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_preds_list.extend(nn_model(bx).cpu().numpy())
    oof_nn[va_idx] = np.array(val_preds_list)

    te_preds_list = []
    with torch.no_grad():
        for bx in test_loader:
            bx = bx[0].to(device)
            te_preds_list.extend(nn_model(bx).cpu().numpy())
    test_nn_preds += np.array(te_preds_list) / num_folds

# --- Ensemble Weight Optimization & Threshold Calibration ---
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
test_matrix = np.column_stack([test_lgb_preds, test_xgb_preds, test_cb_preds, test_nn_preds])

init_rounder = OptimizedRounder()


def optimize_blend_weights(oofs, y_true, base_rounder):
    def loss_func(weights):
        w = np.array(weights)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        blend = np.dot(oofs, w)
        preds = base_rounder.predict(blend)
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    init_weights = np.array([0.35, 0.30, 0.25, 0.10])
    res = sp.optimize.minimize(
        loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 500}
    )
    best_weights = np.array(res.x)
    if best_weights.sum() != 0:
        best_weights = best_weights / best_weights.sum()
    return best_weights


optimal_weights = optimize_blend_weights(oof_matrix, y_train_full, init_rounder)
print(f"Optimized Blend Weights (LGB, XGB, CB, NN): {optimal_weights}")

oof_blend = np.dot(oof_matrix, optimal_weights)
test_blend = np.dot(test_matrix, optimal_weights)

# Refine cutoffs on the optimal blend
rounder = OptimizedRounder()
rounder.fit(oof_blend, y_train_full)

# Convert continuous out-of-fold and test predictions to discrete classes
oof_discrete = rounder.predict(oof_blend)
test_discrete = rounder.predict(test_blend)

# Compute local OOF Quadratic Weighted Kappa score
final_score = cohen_kappa_score(y_train_full, oof_discrete, weights="quadratic")

# --- Generate Submission File ---
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "AdoptionSpeed": test_discrete.astype(int)}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")