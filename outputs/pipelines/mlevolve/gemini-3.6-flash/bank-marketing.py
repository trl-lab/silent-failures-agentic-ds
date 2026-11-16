import os
import random
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# Set random seed for complete reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

# Create necessary working directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# 1. Feature Engineering & Preprocessing
# ==============================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)


def standardize_df(df):
    df = df.copy()
    col_map = {
        "V1": "age",
        "V2": "job",
        "V3": "marital",
        "V4": "education",
        "V5": "default",
        "V6": "balance",
        "V7": "housing",
        "V8": "loan",
        "V9": "contact",
        "V10": "day",
        "V11": "month",
        "V12": "duration",
        "V13": "campaign",
        "V14": "pdays",
        "V15": "previous",
        "V16": "poutcome",
        "v1": "age",
        "v2": "job",
        "v3": "marital",
        "v4": "education",
        "v5": "default",
        "v6": "balance",
        "v7": "housing",
        "v8": "loan",
        "v9": "contact",
        "v10": "day",
        "v11": "month",
        "v12": "duration",
        "v13": "campaign",
        "v14": "pdays",
        "v15": "previous",
        "v16": "poutcome",
    }
    df = df.rename(columns=col_map)
    cols_lower = {
        c: c.lower() for c in df.columns if c not in ["Class", "y", "target", "id"]
    }
    df = df.rename(columns=cols_lower)

    expected_cols = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]
    missing_expected = [c for c in expected_cols if c not in df.columns]
    if missing_expected:
        feature_cols = [
            c for c in df.columns if c not in ["Class", "y", "target", "id"]
        ]
        if len(feature_cols) == 16:
            df = df.rename(columns=dict(zip(feature_cols, expected_cols)))
    return df


train_raw = standardize_df(train_raw)
test_raw = standardize_df(test_raw)

if "Class" in train_raw.columns:
    target_col = "Class"
elif "y" in train_raw.columns:
    target_col = "y"
elif "target" in train_raw.columns:
    target_col = "target"
else:
    target_col = train_raw.columns[-1]

y_raw = train_raw[target_col]
if y_raw.dtype == object or str(y_raw.dtype) == "category":
    unique_vals = sorted(y_raw.unique())
    if set(unique_vals) == {"1", "2"}:
        y = (y_raw == "2").astype(int).values
    elif set(unique_vals) == {"no", "yes"}:
        y = (y_raw == "yes").astype(int).values
    else:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw)
else:
    unique_vals = sorted(y_raw.unique())
    if set(unique_vals) == {1, 2}:
        y = (y_raw == 2).astype(int).values
    else:
        y = y_raw.astype(int).values

train_raw["target"] = y


def engineer_features(df):
    df_feat = df.copy()

    num_cols_raw = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    for c in num_cols_raw:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0)

    cat_cols_raw = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
    ]
    for c in cat_cols_raw:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str)

    # 1. Sentinel & pdays transformations
    df_feat["pdays_contacted"] = (df_feat["pdays"] != -1).astype(int)
    df_feat["pdays_log"] = np.where(
        df_feat["pdays"] > 0, np.log1p(df_feat["pdays"]), 0.0
    )
    df_feat["pdays_recency"] = np.where(
        df_feat["pdays"] > 0, 1.0 / (df_feat["pdays"] + 1.0), 0.0
    )

    # 2. Financial Metrics & Liquidity Proxies
    df_feat["balance_is_negative"] = (df_feat["balance"] < 0).astype(int)
    df_feat["balance_is_zero"] = (df_feat["balance"] == 0).astype(int)
    df_feat["balance_signed_log"] = np.sign(df_feat["balance"]) * np.log1p(
        df_feat["balance"].abs()
    )

    df_feat["has_default"] = (df_feat["default"] == "yes").astype(int)
    df_feat["has_housing"] = (df_feat["housing"] == "yes").astype(int)
    df_feat["has_loan"] = (df_feat["loan"] == "yes").astype(int)
    df_feat["total_active_loans"] = df_feat["has_housing"] + df_feat["has_loan"]
    df_feat["financial_vulnerability"] = (
        df_feat["has_default"] * 2
        + df_feat["has_housing"]
        + df_feat["has_loan"]
        + df_feat["balance_is_negative"]
    )
    df_feat["balance_per_age"] = df_feat["balance"] / (df_feat["age"] + 1e-5)

    # 3. Campaign Intensity & Duration Dynamics
    df_feat["duration_min"] = df_feat["duration"] / 60.0
    df_feat["duration_log"] = np.log1p(np.maximum(df_feat["duration"], 0))
    df_feat["duration_per_campaign"] = df_feat["duration"] / (
        df_feat["campaign"] + 1e-5
    )
    df_feat["total_contacts"] = df_feat["campaign"] + df_feat["previous"]
    df_feat["previous_contact_ratio"] = df_feat["previous"] / (
        df_feat["total_contacts"] + 1e-5
    )
    df_feat["is_first_contact"] = (df_feat["previous"] == 0).astype(int)

    # 4. Temporal Features
    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    def parse_month(val):
        val_str = str(val).lower().strip()
        if val_str in month_map:
            return month_map[val_str]
        try:
            return int(float(val_str))
        except (ValueError, TypeError):
            return 0

    df_feat["month_num"] = df_feat["month"].apply(parse_month).astype(int)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month_num"] / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month_num"] / 12.0)
    df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["day"] / 31.0)
    df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["day"] / 31.0)
    df_feat["is_month_start"] = (df_feat["day"] <= 10).astype(int)
    df_feat["is_month_mid"] = ((df_feat["day"] > 10) & (df_feat["day"] <= 20)).astype(
        int
    )
    df_feat["is_month_end"] = (df_feat["day"] > 20).astype(int)

    # 5. Domain Interaction Strings
    df_feat["job_education"] = (
        df_feat["job"].astype(str) + "_" + df_feat["education"].astype(str)
    )
    df_feat["marital_housing"] = (
        df_feat["marital"].astype(str) + "_" + df_feat["housing"].astype(str)
    )
    df_feat["month_poutcome"] = (
        df_feat["month"].astype(str) + "_" + df_feat["poutcome"].astype(str)
    )

    return df_feat


train_eng = engineer_features(train_raw)
test_eng = engineer_features(test_raw)

cat_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "job_education",
    "marital_housing",
    "month_poutcome",
]

# Frequency Encoding
for col in cat_cols:
    freq_map = train_eng[col].value_counts(normalize=True).to_dict()
    train_eng[f"{col}_freq"] = train_eng[col].map(freq_map).fillna(0.0)
    test_eng[f"{col}_freq"] = test_eng[col].map(freq_map).fillna(0.0)

# Initialize target encoding column placeholders
for col in cat_cols:
    train_eng[f"{col}_te"] = 0.0
    test_eng[f"{col}_te"] = 0.0

# Label Encoding for tree models and embeddings
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat(
        [train_eng[col].astype(str), test_eng[col].astype(str)], axis=0
    )
    le.fit(combined)
    train_eng[col] = le.transform(train_eng[col].astype(str))
    test_eng[col] = le.transform(test_eng[col].astype(str))

# Group Aggregations
job_balance_mean = train_eng.groupby("job")["balance"].mean().to_dict()
train_eng["job_balance_diff"] = train_eng["balance"] - train_eng["job"].map(
    job_balance_mean
)
test_eng["job_balance_diff"] = test_eng["balance"] - test_eng["job"].map(
    job_balance_mean
)

job_duration_mean = train_eng.groupby("job")["duration"].mean().to_dict()
train_eng["job_duration_diff"] = train_eng["duration"] - train_eng["job"].map(
    job_duration_mean
)
test_eng["job_duration_diff"] = test_eng["duration"] - test_eng["job"].map(
    job_duration_mean
)

train_eng.to_parquet("./working/train_processed.parquet", index=False)
test_eng.to_parquet("./working/test_processed.parquet", index=False)


# ==============================================================================
# 2. PyTorch Neural Network Definition
# ==============================================================================
class TabularDataset(Dataset):

    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = (
            torch.tensor(x_cat, dtype=torch.long) if x_cat is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            if self.x_cat is not None:
                return self.x_num[idx], self.x_cat[idx], self.y[idx]
            return self.x_num[idx], self.y[idx]
        else:
            if self.x_cat is not None:
                return self.x_num[idx], self.x_cat[idx]
            return self.x_num[idx]


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.bn1(self.fc1(x))))
        out = self.drop2(self.act2(self.bn2(self.fc2(out))))
        return out + residual


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list = None,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        if cat_cardinalities:
            for num_classes in cat_cardinalities:
                emb_dim = min(32, max(4, (num_classes + 1) // 2))
                self.embeddings.append(nn.Embedding(num_classes + 1, emb_dim))
                total_emb_dim += emb_dim

        self.num_bn = (
            nn.BatchNorm1d(num_continuous) if num_continuous > 0 else nn.Identity()
        )
        in_dim = num_continuous + total_emb_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.input_act = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        x_components = []

        if x_num is not None and x_num.shape[1] > 0:
            x_num_norm = self.num_bn(x_num)
            x_components.append(x_num_norm)

        if x_cat is not None and len(self.embeddings) > 0:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(emb_outs, dim=1)
            x_components.append(x_cat_emb)

        x = torch.cat(x_components, dim=1)
        x = self.input_act(self.input_bn(self.input_proj(x)))

        for block in self.res_blocks:
            x = block(x)

        logits = self.head(x)
        return logits.squeeze(-1)


# ==============================================================================
# 3. Model Training & Evaluation (5-Fold Stratified CV)
# ==============================================================================
train_df = pd.read_parquet("./working/train_processed.parquet")
test_df = pd.read_parquet("./working/test_processed.parquet")

ignore_cols = ["target", "Class", "y", "id"]
features = [c for c in train_df.columns if c not in ignore_cols]

num_cols = [c for c in features if c not in cat_cols]

cat_cardinalities = [
    int(
        max(
            train_df[col].max() if col in train_df.columns else 0,
            test_df[col].max() if col in test_df.columns else 0,
        )
        + 1
    )
    for col in cat_cols
]

model_names = ["lgb", "xgb", "cat", "nn"]
oof_preds = {m: np.zeros(len(train_df)) for m in model_names}
test_preds = {m: np.zeros(len(test_df)) for m in model_names}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
    X_tr_df, y_tr = train_df.iloc[train_idx][features].copy(), y[train_idx]
    X_va_df, y_va = train_df.iloc[val_idx][features].copy(), y[val_idx]
    X_te_df = test_df[features].copy()

    # In-fold Target Encoding to prevent data leakage
    global_mean = y_tr.mean()
    for col in cat_cols:
        tr_col_data = train_df.iloc[train_idx][col]
        group_stats = pd.DataFrame({"cat": tr_col_data, "target": y_tr}).groupby("cat")["target"].agg(["count", "mean"])
        m = 10.0
        smoothed = (group_stats["count"] * group_stats["mean"] + m * global_mean) / (group_stats["count"] + m)
        te_map = smoothed.to_dict()

        X_tr_df[f"{col}_te"] = tr_col_data.map(te_map).fillna(global_mean)
        X_va_df[f"{col}_te"] = train_df.iloc[val_idx][col].map(te_map).fillna(global_mean)
        X_te_df[f"{col}_te"] = test_df[col].map(te_map).fillna(global_mean)

    # Model 1: LightGBM
    trn_data = lgb.Dataset(X_tr_df, label=y_tr)
    val_data = lgb.Dataset(X_va_df, label=y_va, reference=trn_data)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds["lgb"][val_idx] = lgb_model.predict(
        X_va_df, num_iteration=lgb_model.best_iteration
    )
    test_preds["lgb"] += (
        lgb_model.predict(X_te_df, num_iteration=lgb_model.best_iteration) / 5.0
    )

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    xgb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        verbose=False,
    )

    oof_preds["xgb"][val_idx] = xgb_model.predict_proba(X_va_df)[:, 1]
    test_preds["xgb"] += xgb_model.predict_proba(X_te_df)[:, 1] / 5.0

    # Model 3: CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        loss_function="Logloss",
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1,
        early_stopping_rounds=50,
    )

    cb_model.fit(
        X_tr_df,
        y_tr,
        eval_set=(X_va_df, y_va),
        use_best_model=True,
    )

    oof_preds["cat"][val_idx] = cb_model.predict_proba(X_va_df)[:, 1]
    test_preds["cat"] += cb_model.predict_proba(X_te_df)[:, 1] / 5.0

    # Model 4: PyTorch TabularResNet
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_tr_df[num_cols].values)
    X_va_num = scaler.transform(X_va_df[num_cols].values)
    X_te_num = scaler.transform(X_te_df[num_cols].values)

    X_tr_cat = X_tr_df[cat_cols].values.astype(np.int64)
    X_va_cat = X_va_df[cat_cols].values.astype(np.int64)
    X_te_cat = X_te_df[cat_cols].values.astype(np.int64)

    train_ds = TabularDataset(X_tr_num, X_tr_cat, y_tr)
    val_ds = TabularDataset(X_va_num, X_va_cat, y_va)
    test_ds = TabularDataset(X_te_num, X_te_cat)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    nn_model = TabularResNet(
        num_continuous=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=128,
        num_res_blocks=2,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    best_val_auc = -1.0
    best_nn_val_probs = None
    best_nn_test_probs = None

    for epoch in range(25):
        nn_model.train()
        for batch in train_loader:
            b_num, b_cat, b_y = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            optimizer.zero_grad()
            out = nn_model(b_num, b_cat)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                b_num, b_cat = batch[0].to(device), batch[1].to(device)
                logits = nn_model(b_num, b_cat)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.append(probs)

        val_probs = np.concatenate(val_probs_list)
        val_auc = roc_auc_score(y_va, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_nn_val_probs = val_probs

            test_probs_list = []
            with torch.no_grad():
                for batch in test_loader:
                    b_num, b_cat = batch[0].to(device), batch[1].to(device)
                    logits = nn_model(b_num, b_cat)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    test_probs_list.append(probs)
            best_nn_test_probs = np.concatenate(test_probs_list)

    oof_preds["nn"][val_idx] = best_nn_val_probs
    test_preds["nn"] += best_nn_test_probs / 5.0


# ==============================================================================
# 4. Ensemble Weight Optimization & Submission Generation
# ==============================================================================
def ensemble_loss(weights):
    weights = np.array(weights)
    weights = weights / (np.sum(weights) + 1e-12)
    blend_oof = np.zeros(len(y), dtype=np.float64)
    for i, m in enumerate(model_names):
        blend_oof += weights[i] * oof_preds[m]
    return -roc_auc_score(y, blend_oof)


init_weights = [0.25] * len(model_names)
bounds = [(0, 1)] * len(model_names)
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x / np.sum(res.x)

final_oof_preds = np.zeros(len(train_df))
final_test_preds = np.zeros(len(test_df))

for i, m in enumerate(model_names):
    final_oof_preds += best_weights[i] * oof_preds[m]
    final_test_preds += best_weights[i] * test_preds[m]

final_score = roc_auc_score(y, final_oof_preds)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "Class": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")