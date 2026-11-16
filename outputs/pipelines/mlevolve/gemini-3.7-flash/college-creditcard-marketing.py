import copy
import os
import re
import warnings
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Data Loading & Preprocessing
# =========================================================================
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

TARGET_COL = "PAYMENTS_BY_ISSUER"
assert (
    TARGET_COL in train_df.columns
), f"Target column {TARGET_COL} not found in train data!"

# Identify ID column if present
id_col = None
for candidate in ["id", "ID", "index", "agreement_id"]:
    if candidate in test_df.columns:
        id_col = candidate
        break

y = train_df[TARGET_COL].values.copy().astype(float)
train_raw = train_df.drop(columns=[TARGET_COL])
test_raw = test_df.copy()

if id_col and id_col in train_raw.columns:
    train_raw = train_raw.drop(columns=[id_col])

test_ids = (
    test_raw[id_col].values
    if id_col and id_col in test_raw.columns
    else np.arange(len(test_raw))
)
if id_col and id_col in test_raw.columns:
    test_raw = test_raw.drop(columns=[id_col])


def clean_strings(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    return re.sub(r"\s+", " ", s)


cat_cols = []
num_cols = []
text_cols = []

for col in train_raw.columns:
    if train_raw[col].dtype == "object" or train_raw[col].dtype.name == "category":
        sample_lens = train_raw[col].dropna().astype(str).map(len)
        if len(sample_lens) > 0 and sample_lens.mean() > 40:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif np.issubdtype(train_raw[col].dtype, np.number):
        num_cols.append(col)
    else:
        cat_cols.append(col)


def engineer_base_features(df_in, numeric_columns, categorical_columns, text_columns):
    df = df_in.copy()

    for col in categorical_columns:
        df[col] = df[col].apply(clean_strings)
        df[f"{col}_len"] = df[col].astype(str).map(len)
        df[f"{col}_word_count"] = (
            df[col].astype(str).apply(lambda x: len(x.split()) if x else 0)
        )

    for col in numeric_columns:
        df[f"{col}_isna"] = df[col].isna().astype(float)
        col_min = df[col].min()
        if pd.notna(col_min) and col_min >= 0:
            df[f"{col}_log1p"] = np.log1p(df[col].fillna(0).clip(lower=0))

    for i, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[i + 1 :]:
            df[f"{col1}_mult_{col2}"] = df[col1].fillna(0) * df[col2].fillna(0)
            df[f"{col1}_div_{col2}"] = (df[col1].fillna(0) + 1.0) / (
                df[col2].fillna(0) + 1.0
            )
            df[f"{col2}_div_{col1}"] = (df[col2].fillna(0) + 1.0) / (
                df[col1].fillna(0) + 1.0
            )

    if len(categorical_columns) >= 2:
        for i, c1 in enumerate(categorical_columns[:4]):
            for c2 in categorical_columns[i + 1 : 5]:
                df[f"{c1}__x__{c2}"] = df[c1].astype(str) + "___" + df[c2].astype(str)

    return df


train_feat = engineer_base_features(train_raw, num_cols, cat_cols, text_cols)
test_feat = engineer_base_features(test_raw, num_cols, cat_cols, text_cols)

all_cat_cols = [
    c
    for c in train_feat.columns
    if train_feat[c].dtype == "object" and c not in text_cols
]
all_num_cols = [
    c for c in train_feat.columns if np.issubdtype(train_feat[c].dtype, np.number)
]

for col in all_cat_cols:
    freq_map = train_feat[col].value_counts(normalize=True).to_dict()
    train_feat[f"{col}_freq"] = train_feat[col].map(freq_map).fillna(0.0)
    test_feat[f"{col}_freq"] = test_feat[col].map(freq_map).fillna(0.0)

# Text Feature Extraction
svd_train_dfs = []
svd_test_dfs = []

for t_col in text_cols:
    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    train_text = train_feat[t_col].fillna("").astype(str).tolist()
    test_text = test_feat[t_col].fillna("").astype(str).tolist()

    tfidf_mat_train = tfidf.fit_transform(train_text)
    tfidf_mat_test = tfidf.transform(test_text)

    n_components = (
        min(10, tfidf_mat_train.shape[1] - 1) if tfidf_mat_train.shape[1] > 1 else 1
    )
    if n_components >= 1:
        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        svd_tr = svd.fit_transform(tfidf_mat_train)
        svd_te = svd.transform(tfidf_mat_test)

        svd_cols = [f"{t_col}_svd_{k}" for k in range(n_components)]
        svd_train_dfs.append(
            pd.DataFrame(svd_tr, columns=svd_cols, index=train_feat.index)
        )
        svd_test_dfs.append(
            pd.DataFrame(svd_te, columns=svd_cols, index=test_feat.index)
        )

if svd_train_dfs:
    train_feat = pd.concat([train_feat] + svd_train_dfs, axis=1)
    test_feat = pd.concat([test_feat] + svd_test_dfs, axis=1)

train_feat.drop(columns=text_cols, inplace=True, errors="ignore")
test_feat.drop(columns=text_cols, inplace=True, errors="ignore")

# Stratified K-Fold Setup
N_SPLITS = 5
num_bins = min(10, max(2, len(train_feat) // 30))
y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(train_feat, y_bins))

# Leak-Free Target Encodings
primary_cats = [c for c in all_cat_cols if c in train_raw.columns][:5]
oof_te_train = pd.DataFrame(index=train_feat.index)
te_test_folds = []

for c in primary_cats:
    oof_te_train[f"{c}_target_mean"] = 0.0
    oof_te_train[f"{c}_target_std"] = 0.0
    oof_te_train[f"{c}_target_max"] = 0.0

global_target_mean = float(np.mean(y))

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr_fold = train_feat.iloc[train_idx]
    y_tr_fold = y[train_idx]
    X_val_fold = train_feat.iloc[val_idx]

    fold_te_test = pd.DataFrame(index=test_feat.index)

    for c in primary_cats:
        df_group = pd.DataFrame({c: X_tr_fold[c], "target": y_tr_fold})
        stats = df_group.groupby(c)["target"].agg(["mean", "std", "max", "count"])

        smooth_weight = 10.0
        smoothed_mean = (
            stats["count"] * stats["mean"] + smooth_weight * global_target_mean
        ) / (stats["count"] + smooth_weight)

        val_c = X_val_fold[c]
        oof_te_train.loc[val_idx, f"{c}_target_mean"] = val_c.map(smoothed_mean).fillna(
            global_target_mean
        )
        oof_te_train.loc[val_idx, f"{c}_target_std"] = val_c.map(stats["std"]).fillna(
            0.0
        )
        oof_te_train.loc[val_idx, f"{c}_target_max"] = val_c.map(stats["max"]).fillna(
            global_target_mean
        )

        test_c = test_feat[c]
        fold_te_test[f"{c}_target_mean"] = test_c.map(smoothed_mean).fillna(
            global_target_mean
        )
        fold_te_test[f"{c}_target_std"] = test_c.map(stats["std"]).fillna(0.0)
        fold_te_test[f"{c}_target_max"] = test_c.map(stats["max"]).fillna(
            global_target_mean
        )

    te_test_folds.append(fold_te_test)

oof_te_test = pd.concat(te_test_folds).groupby(level=0).mean()
train_feat = pd.concat([train_feat, oof_te_train], axis=1)
test_feat = pd.concat([test_feat, oof_te_test], axis=1)

for col in all_cat_cols:
    if col in train_feat.columns:
        train_feat[col] = train_feat[col].fillna("missing").astype(str)
        test_feat[col] = test_feat[col].fillna("missing").astype(str)
        cats = sorted(
            list(set(train_feat[col].unique()).union(set(test_feat[col].unique())))
        )
        train_feat[col] = pd.Categorical(train_feat[col], categories=cats)
        test_feat[col] = pd.Categorical(test_feat[col], categories=cats)

for col in train_feat.columns:
    if col not in all_cat_cols and np.issubdtype(train_feat[col].dtype, np.number):
        train_feat[col] = train_feat[col].replace([np.inf, -np.inf], np.nan)
        test_feat[col] = test_feat[col].replace([np.inf, -np.inf], np.nan)

# =========================================================================
# 2. Model Architectures & Factories
# =========================================================================


def create_lgb_model(random_state: int = 42, **kwargs) -> lgb.LGBMRegressor:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def create_xgb_model(random_state: int = 42, **kwargs) -> xgb.XGBRegressor:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def create_catboost_model(random_state: int = 42, **kwargs) -> CatBoostRegressor:
    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
    }
    params.update(kwargs)
    return CatBoostRegressor(**params)


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        for block in self.blocks:
            feat = block(feat)
        out = self.head(feat)
        return out.squeeze(-1)


# =========================================================================
# 3. Cross-Validation Training & Inference Loop
# =========================================================================
cat_cols = [c for c in train_feat.columns if train_feat[c].dtype.name == "category"]
num_cols = [c for c in train_feat.columns if c not in cat_cols]

oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cb = np.zeros(len(y))
oof_nn = np.zeros(len(y))

test_preds_lgb = np.zeros(len(test_feat))
test_preds_xgb = np.zeros(len(test_feat))
test_preds_cb = np.zeros(len(test_feat))
test_preds_nn = np.zeros(len(test_feat))

train_feat_numeric = train_feat.copy()
test_feat_numeric = test_feat.copy()

for col in cat_cols:
    train_feat_numeric[col] = train_feat_numeric[col].cat.codes.astype(float)
    test_feat_numeric[col] = test_feat_numeric[col].cat.codes.astype(float)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_train_df, y_train = train_feat.iloc[train_idx], y[train_idx]
    X_val_df, y_val = train_feat.iloc[val_idx], y[val_idx]

    # 1. LightGBM Regressor
    lgb_model = create_lgb_model(random_state=SEED + fold_idx)
    lgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_df)
    test_preds_lgb += lgb_model.predict(test_feat) / len(folds)

    # 2. XGBoost Regressor
    xgb_model = create_xgb_model(
        random_state=SEED + fold_idx,
        enable_categorical=True,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val_df)
    test_preds_xgb += xgb_model.predict(test_feat) / len(folds)

    # 3. CatBoost Regressor
    cb_model = create_catboost_model(
        random_state=SEED + fold_idx,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_train_df,
        y_train,
        eval_set=(X_val_df, y_val),
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        verbose=False,
    )
    oof_cb[val_idx] = cb_model.predict(X_val_df)
    test_preds_cb += cb_model.predict(test_feat) / len(folds)

    # 4. PyTorch Tabular ResNet
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_num = train_feat_numeric.iloc[train_idx].values
    X_val_num = train_feat_numeric.iloc[val_idx].values
    X_te_num = test_feat_numeric.values

    X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr_num))
    X_val_proc = scaler.transform(imputer.transform(X_val_num))
    X_te_proc = scaler.transform(imputer.transform(X_te_num))

    y_mean, y_std = y_train.mean(), y_train.std() + 1e-7
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    train_dataset = TensorDataset(
        torch.tensor(X_tr_proc, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_proc, dtype=torch.float32),
        torch.tensor(y_val_norm, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(64, max(16, len(train_idx) // 10)),
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    nn_model = TabularResNet(
        input_dim=X_tr_proc.shape[1],
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.1,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    patience = 12
    patience_counter = 0

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = nn_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                pred = nn_model(val_x)
                val_loss += criterion(pred, val_y).item() * len(val_x)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_proc, dtype=torch.float32).to(device)
        nn_val_norm_pred = nn_model(val_tensor).cpu().numpy()
        oof_nn[val_idx] = nn_val_norm_pred * y_std + y_mean

        test_tensor = torch.tensor(X_te_proc, dtype=torch.float32).to(device)
        nn_test_norm_pred = nn_model(test_tensor).cpu().numpy()
        test_preds_nn += (nn_test_norm_pred * y_std + y_mean) / len(folds)

# =========================================================================
# 4. Out-of-Fold Evaluation & Optimal Constrained Ensemble
# =========================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn]
)


def blend_objective(weights):
    weights = np.array(weights)
    blend_pred = np.clip(oof_matrix @ weights, 0, None)
    return np.sqrt(mean_squared_error(y, blend_pred))


initial_weights = [0.35, 0.35, 0.25, 0.05]
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x if opt_res.success else np.array(initial_weights)
best_weights = best_weights / np.sum(best_weights)

final_oof = np.clip(oof_matrix @ best_weights, 0, None)
final_test = np.clip(test_matrix @ best_weights, 0, None)

final_score = np.sqrt(mean_squared_error(y, final_oof))

# =========================================================================
# 5. Generate Submission File & Verify Output
# =========================================================================
submission_df = pd.DataFrame(
    {
        "id": (
            test_ids.astype(int)
            if np.issubdtype(test_ids.dtype, np.integer)
            else np.arange(len(final_test))
        ),
        "PAYMENTS_BY_ISSUER": final_test,
    }
)

submission_df.to_csv(SUBMISSION_PATH, index=False)

assert os.path.exists(SUBMISSION_PATH), "Submission file was not created!"
assert len(submission_df) == len(
    test_feat
), f"Submission row count mismatch: {len(submission_df)} vs {len(test_feat)}"
assert not submission_df.isna().any().any(), "Submission contains NaN values!"
assert list(submission_df.columns) == [
    "id",
    "PAYMENTS_BY_ISSUER",
], f"Unexpected columns: {submission_df.columns}"

print(f"Final Validation Score: {final_score}")
