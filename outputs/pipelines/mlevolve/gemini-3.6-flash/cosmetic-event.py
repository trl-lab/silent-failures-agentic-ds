import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Environment & Reproducibility Setup
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Data Loading & Feature Engineering
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "outcomes"
assert target_col in train_df.columns, f"Target '{target_col}' missing in train."

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df[target_col].astype(str))
n_classes = len(label_encoder.classes_)

id_col = (
    "id" if "id" in test_df.columns else ("ID" if "ID" in test_df.columns else None)
)
if id_col is None:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df[id_col].values

train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()

if id_col and id_col in train_features.columns:
    train_features = train_features.drop(columns=[id_col])
if id_col and id_col in test_features.columns:
    test_features = test_features.drop(columns=[id_col])

n_train = len(train_df)
n_test = len(test_df)
df_all = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# Categorize Features
num_cols, cat_cols, dt_cols = [], [], []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        dt_cols.append(col)
    elif pd.api.types.is_numeric_dtype(df_all[col]):
        if df_all[col].nunique() <= 15:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notnull().sum() > 0.5 * len(df_all):
                df_all[col] = converted
                dt_cols.append(col)
            else:
                cat_cols.append(col)
        except Exception:
            cat_cols.append(col)

# Datetime Transformations
for col in dt_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_hour"] = df_all[col].dt.hour
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear
    df_all[f"{col}_is_weekend"] = (df_all[col].dt.dayofweek >= 5).astype(int)
    df_all[f"{col}_hour_sin"] = np.sin(2 * np.pi * df_all[col].dt.hour / 24.0)
    df_all[f"{col}_hour_cos"] = np.cos(2 * np.pi * df_all[col].dt.hour / 24.0)
    df_all[f"{col}_month_sin"] = np.sin(2 * np.pi * df_all[col].dt.month / 12.0)
    df_all[f"{col}_month_cos"] = np.cos(2 * np.pi * df_all[col].dt.month / 12.0)
    df_all.drop(columns=[col], inplace=True)

# Missing Indicators
for col in list(df_all.columns):
    if df_all[col].isnull().sum() > 0:
        df_all[f"{col}_isna"] = df_all[col].isnull().astype(int)

# Categorical Encodings (Leak-Free Fit on Train)
for col in cat_cols:
    df_all[col] = df_all[col].fillna("Missing").astype(str)
    freq = df_all.iloc[:n_train][col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(float)
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

# Numerical Feature Transformations
num_cols_updated = [c for c in num_cols if c in df_all.columns]
if len(num_cols_updated) > 0:
    for col in num_cols_updated:
        median_val = df_all.iloc[:n_train][col].median()
        if pd.isnull(median_val):
            median_val = 0.0
        df_all[col] = df_all[col].fillna(median_val)

        train_vals = df_all.iloc[:n_train][col]
        if (train_vals >= 0).all() and train_vals.skew() > 0.75:
            df_all[f"{col}_log1p"] = np.log1p(np.maximum(0, df_all[col]))

    df_all["num_row_mean"] = df_all[num_cols_updated].mean(axis=1)
    df_all["num_row_std"] = df_all[num_cols_updated].std(axis=1).fillna(0)
    df_all["num_row_min"] = df_all[num_cols_updated].min(axis=1)
    df_all["num_row_max"] = df_all[num_cols_updated].max(axis=1)
    df_all["num_row_sum"] = df_all[num_cols_updated].sum(axis=1)

    if len(cat_cols) > 0:
        top_cats = cat_cols[: min(3, len(cat_cols))]
        top_nums = num_cols_updated[: min(5, len(num_cols_updated))]
        train_subset = df_all.iloc[:n_train]

        for cat in top_cats:
            for num in top_nums:
                grp = train_subset.groupby(cat)[num].agg(["mean", "std"]).reset_index()
                grp.columns = [cat, f"{num}_grp_{cat}_mean", f"{num}_grp_{cat}_std"]
                df_all = df_all.merge(grp, on=cat, how="left")
                df_all[f"{num}_grp_{cat}_std"] = df_all[f"{num}_grp_{cat}_std"].fillna(
                    0
                )
                df_all[f"{num}_diff_grp_{cat}_mean"] = (
                    df_all[num] - df_all[f"{num}_grp_{cat}_mean"]
                )
                df_all[f"{num}_zscore_grp_{cat}"] = df_all[
                    f"{num}_diff_grp_{cat}_mean"
                ] / (df_all[f"{num}_grp_{cat}_std"] + 1e-5)

df_all.fillna(0, inplace=True)

X_train_df = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_df = df_all.iloc[n_train:].copy().reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = np.zeros(n_train, dtype=int)
for fold_idx, (_, val_idx) in enumerate(skf.split(X_train_df, y)):
    folds[val_idx] = fold_idx

X_train_full = X_train_df.values
y_train_full = y.astype(int)
X_test_full = X_test_df.values


# 3. PyTorch Tabular Architecture Definition
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        h = self.fc1(x_norm)
        gate, val = torch.chunk(h, 2, dim=-1)
        h = torch.sigmoid(gate) * self.act(val)
        h = self.dropout(self.fc2(h))
        return residual + h


class TabularGatedResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# 4. Base Model Factory & Training Functions
def get_lgbm_model(n_cls, seed):
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if n_cls > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_cls
    else:
        params["objective"] = "binary"
    return LGBMClassifier(**params)


def get_xgb_model(n_cls, seed):
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mlogloss" if n_cls > 2 else "logloss",
        "early_stopping_rounds": 50,
    }
    if n_cls > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_cls
    else:
        params["objective"] = "binary:logistic"
    return XGBClassifier(**params)


def get_catboost_model(n_cls, seed):
    params = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "thread_count": -1,
        "verbose": False,
    }
    if n_cls > 2:
        params["loss_function"] = "MultiClass"
    else:
        params["loss_function"] = "Logloss"
    return CatBoostClassifier(**params)


def get_extratrees_model(n_cls, seed):
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1,
    )


def train_pytorch_fold(
    X_tr_s, y_tr, X_val_s, y_val, X_te_s, in_features, n_cls, dev, epochs=25
):
    model = TabularGatedResNet(in_features=in_features, num_classes=n_cls).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    tr_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    te_ds = TensorDataset(torch.tensor(X_te_s, dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                val_loss += criterion(model(bx), by).item() * len(by)
                val_count += len(by)
        val_loss /= val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    model.eval()

    val_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(dev)
            probs = F.softmax(model(bx), dim=1).cpu().numpy()
            val_probs.append(probs)
    val_probs = np.vstack(val_probs)

    te_probs = []
    with torch.no_grad():
        for batch in te_loader:
            bx = batch[0].to(dev)
            probs = F.softmax(model(bx), dim=1).cpu().numpy()
            te_probs.append(probs)
    te_probs = np.vstack(te_probs)

    return val_probs, te_probs


# 5. Out-Of-Fold Level-1 Ensembling
num_models = 5
n_folds = 5
X_train_l1 = np.zeros((n_train, num_models * n_classes), dtype=np.float32)
X_test_l1 = np.zeros((n_test, num_models * n_classes), dtype=np.float32)

for f in range(n_folds):
    val_idx = folds == f
    train_idx = ~val_idx

    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    # Model 1: LightGBM
    lgb_model = get_lgbm_model(n_classes, SEED + f)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_val_prob = lgb_model.predict_proba(X_val)
    lgb_test_prob = lgb_model.predict_proba(X_test_full)

    # Model 2: XGBoost
    xgb_model = get_xgb_model(n_classes, SEED + f)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_prob = xgb_model.predict_proba(X_val)
    xgb_test_prob = xgb_model.predict_proba(X_test_full)

    # Model 3: CatBoost
    cb_model = get_catboost_model(n_classes, SEED + f)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_val_prob = cb_model.predict_proba(X_val)
    cb_test_prob = cb_model.predict_proba(X_test_full)

    # Model 4: ExtraTrees
    et_model = get_extratrees_model(n_classes, SEED + f)
    et_model.fit(X_tr, y_tr)
    et_val_prob = et_model.predict_proba(X_val)
    et_test_prob = et_model.predict_proba(X_test_full)

    # Model 5: PyTorch TabularGatedResNet
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test_full)

    nn_val_prob, nn_test_prob = train_pytorch_fold(
        X_tr_s,
        y_tr,
        X_val_s,
        y_val,
        X_te_s,
        in_features=X_train_full.shape[1],
        n_cls=n_classes,
        dev=device,
        epochs=25,
    )

    fold_l1_val = np.hstack(
        [lgb_val_prob, xgb_val_prob, cb_val_prob, et_val_prob, nn_val_prob]
    )
    fold_l1_test = np.hstack(
        [lgb_test_prob, xgb_test_prob, cb_test_prob, et_test_prob, nn_test_prob]
    )

    X_train_l1[val_idx] = fold_l1_val
    X_test_l1 += fold_l1_test / float(n_folds)


# 6. Level-2 Meta-Learner Stacking
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_meta_probs = np.zeros((n_train, n_classes), dtype=np.float32)
test_meta_probs = np.zeros((n_test, n_classes), dtype=np.float32)

for meta_fold, (m_train_idx, m_val_idx) in enumerate(
    skf_meta.split(X_train_l1, y_train_full)
):
    m_X_tr, m_y_tr = X_train_l1[m_train_idx], y_train_full[m_train_idx]
    m_X_val = X_train_l1[m_val_idx]

    meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED + meta_fold)
    meta_model.fit(m_X_tr, m_y_tr)

    oof_meta_probs[m_val_idx] = meta_model.predict_proba(m_X_val)
    test_meta_probs += meta_model.predict_proba(X_test_l1) / 5.0


# 7. Evaluate Score & Generate Final Submission
oof_meta_preds = np.argmax(oof_meta_probs, axis=1)
final_val_score = accuracy_score(y_train_full, oof_meta_preds)

test_meta_preds = np.argmax(test_meta_probs, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_meta_preds)

submission_df = pd.DataFrame({"id": test_ids, "outcomes": test_pred_labels})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
