import os
import re
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import xgboost as xgb


# -----------------------------------------------------------------------------
# 0. Global Setup & Seed Initialization
# -----------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Data Loading & Modality Identification
# -----------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = "days_to_complete"
assert (
    target_col in df_train.columns
), f"Target column '{target_col}' not found in train data."

test_ids = (
    df_test["id"].to_numpy()
    if "id" in df_test.columns
    else np.arange(len(df_test), dtype=int)
)
y_train = df_train[target_col].to_numpy(dtype=np.float32)
n_train = len(df_train)
n_test = len(df_test)

feature_cols = [c for c in df_train.columns if c != target_col]

numeric_cols = []
datetime_cols = []
text_cols = []
categorical_cols = []

date_patterns = ["date", "posted", "completion", "start", "time"]

for col in feature_cols:
    col_lower = col.lower()
    is_date = False
    if pd.api.types.is_datetime64_any_dtype(df_train[col]):
        is_date = True
    elif any(p in col_lower for p in date_patterns):
        sample = df_train[col].dropna().astype(str).head(20)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise")
                is_date = True
            except Exception:
                is_date = False

    if is_date:
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_train[col]):
        numeric_cols.append(col)
    else:
        sample = df_train[col].dropna().astype(str)
        if len(sample) > 0 and sample.str.len().mean() > 40:
            text_cols.append(col)
        else:
            categorical_cols.append(col)


# -----------------------------------------------------------------------------
# 2. Stateless Base Feature Engineering
# -----------------------------------------------------------------------------
def extract_stateless_features(df_in):
    df = df_in.copy()
    fe_df = pd.DataFrame(index=df.index)

    # Numerical transformations
    for col in numeric_cols:
        val = pd.to_numeric(df[col], errors="coerce")
        fe_df[col] = val
        fe_df[f"{col}_isna"] = val.isna().astype(np.float32)
        valid_vals = val.dropna()
        if len(valid_vals) > 0 and (valid_vals >= 0).all():
            fe_df[f"{col}_log1p"] = np.log1p(val.clip(lower=0))

    # Datetime features
    baseline_date = pd.Timestamp("2020-01-01")
    for col in datetime_cols:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        fe_df[f"{col}_year"] = dt_series.dt.year.astype(np.float32)
        fe_df[f"{col}_month"] = dt_series.dt.month.astype(np.float32)
        fe_df[f"{col}_day"] = dt_series.dt.day.astype(np.float32)
        fe_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.astype(np.float32)
        fe_df[f"{col}_dayofyear"] = dt_series.dt.dayofyear.astype(np.float32)
        fe_df[f"{col}_days_from_baseline"] = (dt_series - baseline_date).dt.days.astype(
            np.float32
        )
        fe_df[f"{col}_isna"] = dt_series.isna().astype(np.float32)

    # Text structure & complexity features
    for col in text_cols:
        text_series = df[col].fillna("").astype(str)
        fe_df[f"{col}_char_len"] = text_series.str.len().astype(np.float32)
        fe_df[f"{col}_word_count"] = text_series.apply(lambda s: len(s.split())).astype(
            np.float32
        )
        fe_df[f"{col}_avg_word_len"] = fe_df[f"{col}_char_len"] / (
            fe_df[f"{col}_word_count"] + 1.0
        )
        fe_df[f"{col}_num_digits"] = text_series.apply(
            lambda s: sum(c.isdigit() for c in s)
        ).astype(np.float32)
        fe_df[f"{col}_num_punct"] = text_series.apply(
            lambda s: len(re.findall(r"[^\w\s]", s))
        ).astype(np.float32)
        fe_df[f"{col}_isna"] = (text_series.str.strip() == "").astype(np.float32)

    # Categorical entity count
    for col in categorical_cols:
        cat_series = df[col].fillna("Missing").astype(str)
        fe_df[f"{col}_entity_count"] = cat_series.apply(
            lambda s: len(re.split(r"[,|;]", s)) if s != "Missing" else 0
        ).astype(np.float32)

    # Combined corpus for SVD NLP embeddings
    if len(text_cols) > 0:
        combined_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    elif len(categorical_cols) > 0:
        combined_text = (
            df[categorical_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
    else:
        combined_text = pd.Series([""] * len(df), index=df.index)

    return fe_df, combined_text


train_stateless, train_text_corpus = extract_stateless_features(df_train)
test_stateless, test_text_corpus = extract_stateless_features(df_test)

# -----------------------------------------------------------------------------
# 3. Model Architecture Definitions
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularResBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def train_nn_fold(
    X_tr_arr: np.ndarray,
    y_tr_arr: np.ndarray,
    X_va_arr: np.ndarray,
    y_va_arr: np.ndarray,
    X_te_arr: np.ndarray,
    input_dim: int,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_arr)
    X_va_s = scaler.transform(X_va_arr)
    X_te_s = scaler.transform(X_te_arr)

    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_arr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va_arr, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout=0.2
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_losses.append(criterion(preds, by).item() * len(by))
        val_loss = sum(val_losses) / len(y_va_arr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(
        {k: v.to(device) for k, v in best_weights.items()}
        if best_weights is not None
        else model.state_dict()
    )
    model.eval()

    with torch.no_grad():
        val_preds_list = []
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_preds_list.append(model(bx).cpu().numpy())
        va_preds = np.concatenate(val_preds_list, axis=0)

        test_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    return va_preds, test_preds


# -----------------------------------------------------------------------------
# 4. Cross-Validation & Multimodal Model Training
# -----------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42

num_bins = 10
bins = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_splits = list(skf.split(df_train, bins))

oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros((N_SPLITS, n_test), dtype=np.float32)
test_preds_xgb = np.zeros((N_SPLITS, n_test), dtype=np.float32)
test_preds_nn = np.zeros((N_SPLITS, n_test), dtype=np.float32)

for fold, (train_idx, val_idx) in enumerate(fold_splits):
    # Partition stateless components
    X_tr_stateless = train_stateless.iloc[train_idx].copy()
    X_va_stateless = train_stateless.iloc[val_idx].copy()
    X_te_stateless = test_stateless.copy()

    corpus_tr = train_text_corpus.iloc[train_idx]
    corpus_va = train_text_corpus.iloc[val_idx]
    corpus_te = test_text_corpus

    cat_tr_list, cat_va_list, cat_te_list = [], [], []

    if len(categorical_cols) > 0:
        cat_data_tr = (
            df_train.iloc[train_idx][categorical_cols].fillna("Missing").astype(str)
        )
        cat_data_va = (
            df_train.iloc[val_idx][categorical_cols].fillna("Missing").astype(str)
        )
        cat_data_te = df_test[categorical_cols].fillna("Missing").astype(str)

        ohe = OneHotEncoder(
            max_categories=30, handle_unknown="ignore", sparse_output=False
        )
        ohe.fit(cat_data_tr)
        ohe_cols = [f"ohe_{c}" for c in ohe.get_feature_names_out(categorical_cols)]

        cat_tr_df = pd.DataFrame(
            ohe.transform(cat_data_tr), index=X_tr_stateless.index, columns=ohe_cols
        )
        cat_va_df = pd.DataFrame(
            ohe.transform(cat_data_va), index=X_va_stateless.index, columns=ohe_cols
        )
        cat_te_df = pd.DataFrame(
            ohe.transform(cat_data_te), index=X_te_stateless.index, columns=ohe_cols
        )

        for col in categorical_cols:
            freq_map = cat_data_tr[col].value_counts(normalize=True).to_dict()
            cat_tr_df[f"{col}_freq"] = (
                cat_data_tr[col].map(freq_map).fillna(0.0).astype(np.float32)
            )
            cat_va_df[f"{col}_freq"] = (
                cat_data_va[col].map(freq_map).fillna(0.0).astype(np.float32)
            )
            cat_te_df[f"{col}_freq"] = (
                cat_data_te[col].map(freq_map).fillna(0.0).astype(np.float32)
            )

        cat_tr_list.append(cat_tr_df)
        cat_va_list.append(cat_va_df)
        cat_te_list.append(cat_te_df)

    # TF-IDF + Latent Semantic Analysis
    tfidf = TfidfVectorizer(
        max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    tfidf_tr = tfidf.fit_transform(corpus_tr)
    tfidf_va = tfidf.transform(corpus_va)
    tfidf_te = tfidf.transform(corpus_te)

    n_components = min(32, tfidf_tr.shape[1] - 1) if tfidf_tr.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd_tr = svd.fit_transform(tfidf_tr)
    svd_va = svd.transform(tfidf_va)
    svd_te = svd.transform(tfidf_te)

    svd_cols = [f"svd_emb_{i}" for i in range(n_components)]
    svd_tr_df = pd.DataFrame(svd_tr, index=X_tr_stateless.index, columns=svd_cols)
    svd_va_df = pd.DataFrame(svd_va, index=X_va_stateless.index, columns=svd_cols)
    svd_te_df = pd.DataFrame(svd_te, index=X_te_stateless.index, columns=svd_cols)

    fold_train_df = pd.concat([X_tr_stateless] + cat_tr_list + [svd_tr_df], axis=1)
    fold_val_df = pd.concat([X_va_stateless] + cat_va_list + [svd_va_df], axis=1)
    fold_test_df = pd.concat([X_te_stateless] + cat_te_list + [svd_te_df], axis=1)

    clean_cols = [re.sub(r"[^\w]+", "_", str(c)) for c in fold_train_df.columns]
    seen_cols = {}
    unique_cols = []
    for c in clean_cols:
        if c not in seen_cols:
            seen_cols[c] = 0
            unique_cols.append(c)
        else:
            seen_cols[c] += 1
            unique_cols.append(f"{c}_{seen_cols[c]}")

    fold_train_df.columns = unique_cols
    fold_val_df.columns = unique_cols
    fold_test_df.columns = unique_cols
    feature_names = unique_cols

    imputer = SimpleImputer(strategy="median")
    imputer.fit(fold_train_df)

    fold_train_imp = pd.DataFrame(
        imputer.transform(fold_train_df),
        index=fold_train_df.index,
        columns=feature_names,
    )
    fold_val_imp = pd.DataFrame(
        imputer.transform(fold_val_df), index=fold_val_df.index, columns=feature_names
    )
    fold_test_imp = pd.DataFrame(
        imputer.transform(fold_test_df),
        index=fold_test_df.index,
        columns=feature_names,
    )

    y_tr = y_train[train_idx]
    y_va = y_train[val_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_estimators=1500,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        fold_train_imp,
        y_tr,
        eval_set=[(fold_val_imp, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(fold_val_imp)
    test_preds_lgb[fold] = lgb_model.predict(fold_test_imp)

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_estimators=1500,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(fold_train_imp, y_tr, eval_set=[(fold_val_imp, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(fold_val_imp)
    test_preds_xgb[fold] = xgb_model.predict(fold_test_imp)

    # --- TabularResNet ---
    val_pred_nn, test_pred_nn = train_nn_fold(
        X_tr_arr=fold_train_imp.values,
        y_tr_arr=y_tr,
        X_va_arr=fold_val_imp.values,
        y_va_arr=y_va,
        X_te_arr=fold_test_imp.values,
        input_dim=fold_train_imp.shape[1],
        epochs=40,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
    )
    oof_nn[val_idx] = val_pred_nn
    test_preds_nn[fold] = test_pred_nn

# -----------------------------------------------------------------------------
# 5. Out-of-Fold Optimization & Submission Generation
# -----------------------------------------------------------------------------
OOF_matrix = np.column_stack([oof_lgb, oof_xgb, oof_nn])


def ensemble_loss(weights):
    weights = weights / np.sum(weights)
    pred = np.dot(OOF_matrix, weights)
    return np.sqrt(mean_squared_error(y_train, pred))


init_weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
bounds = [(0.0, 1.0) for _ in range(3)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.array([0.45, 0.45, 0.10])
)

oof_blended = np.dot(OOF_matrix, opt_weights)
oof_blended = np.clip(oof_blended, a_min=0.0, a_max=None)
final_cv_score = float(np.sqrt(mean_squared_error(y_train, oof_blended)))

# Test blend
mean_test_lgb = np.mean(test_preds_lgb, axis=0)
mean_test_xgb = np.mean(test_preds_xgb, axis=0)
mean_test_nn = np.mean(test_preds_nn, axis=0)

test_blended = (
    opt_weights[0] * mean_test_lgb
    + opt_weights[1] * mean_test_xgb
    + opt_weights[2] * mean_test_nn
)
test_blended = np.clip(test_blended, a_min=0.0, a_max=None)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "days_to_complete": test_blended}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_cv_score}")
