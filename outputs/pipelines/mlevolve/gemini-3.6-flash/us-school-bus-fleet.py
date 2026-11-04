import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# -----------------------------------------------------------------------------
# 1. Environment & Reproducibility Setup
# -----------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Data Loading & Stratified Cross-Validation Split
# -----------------------------------------------------------------------------
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Bus age"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in training dataset.")

if "id" in train_df.columns:
    train_ids = train_df["id"].values
    test_ids = test_df["id"].values
else:
    train_ids = np.arange(len(train_df))
    test_ids = np.arange(len(test_df))
    train_df["id"] = train_ids
    test_df["id"] = test_ids

y_train = train_df[target_col].values.astype(np.float32)

num_folds = 5
num_bins = min(10, len(np.unique(y_train)))
target_bins = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, target_bins)):
    train_df.loc[val_idx, "fold"] = fold


# -----------------------------------------------------------------------------
# 3. Feature Engineering Pipeline
# -----------------------------------------------------------------------------
ignore_cols = [target_col, "id", "fold"]
candidate_cols = [c for c in train_df.columns if c not in ignore_cols]

cat_cols = []
num_cols = []

for c in candidate_cols:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        cat_cols.append(c)
    elif np.issubdtype(train_df[c].dtype, np.number):
        if train_df[c].nunique() <= 10 and train_df[c].dtype in ["int64", "int32"]:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        cat_cols.append(c)

for c in candidate_cols:
    if train_df[c].isnull().sum() > 0 or test_df[c].isnull().sum() > 0:
        train_df[f"{c}_isna"] = train_df[c].isnull().astype(np.float32)
        test_df[f"{c}_isna"] = test_df[c].isnull().astype(np.float32)

for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

for col in cat_cols:
    train_df[col] = train_df[col].astype(str).fillna("Missing")
    test_df[col] = test_df[col].astype(str).fillna("Missing")

smoothing_m = 10.0
global_mean = float(y_train.mean())

for col in cat_cols:
    train_df[f"{col}_te"] = np.nan
    test_df[f"{col}_te"] = 0.0

    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
    test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0.0).astype(np.float32)

    for fold in range(num_folds):
        tr_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        tr_data = train_df[tr_mask]
        stats = tr_data.groupby(col)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing_m * global_mean) / (
            stats["count"] + smoothing_m
        )
        smooth_map = smooth.to_dict()

        train_df.loc[val_mask, f"{col}_te"] = (
            train_df.loc[val_mask, col].map(smooth_map).fillna(global_mean)
        )

    stats_full = train_df.groupby(col)[target_col].agg(["count", "mean"])
    smooth_full = (
        stats_full["count"] * stats_full["mean"] + smoothing_m * global_mean
    ) / (stats_full["count"] + smoothing_m)
    test_df[f"{col}_te"] = (
        test_df[col].map(smooth_full.to_dict()).fillna(global_mean).astype(np.float32)
    )

for col in num_cols:
    skewness = train_df[col].skew()
    if skewness > 1.0 and (train_df[col] >= 0).all():
        train_df[f"{col}_log1p"] = np.log1p(train_df[col].values).astype(np.float32)
        test_df[f"{col}_log1p"] = np.log1p(test_df[col].values).astype(np.float32)

if len(num_cols) >= 2:
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            train_df[f"{col1}_diff_{col2}"] = (
                train_df[col1] - train_df[col2]
            ).astype(np.float32)
            test_df[f"{col1}_diff_{col2}"] = (
                test_df[col1] - test_df[col2]
            ).astype(np.float32)

            train_df[f"{col1}_ratio_{col2}"] = (
                train_df[col1] / (train_df[col2] + 1e-5)
            ).astype(np.float32)
            test_df[f"{col1}_ratio_{col2}"] = (
                test_df[col1] / (test_df[col2] + 1e-5)
            ).astype(np.float32)

            train_df[f"{col1}_prod_{col2}"] = (train_df[col1] * train_df[col2]).astype(
                np.float32
            )
            test_df[f"{col1}_prod_{col2}"] = (test_df[col1] * test_df[col2]).astype(
                np.float32
            )

if len(cat_cols) > 0 and len(num_cols) > 0:
    for c_col in cat_cols:
        for n_col in num_cols:
            for stat in ["mean", "std", "min", "max"]:
                train_df[f"{c_col}_{n_col}_grp_{stat}"] = np.nan
                test_df[f"{c_col}_{n_col}_grp_{stat}"] = np.nan

    for fold in range(num_folds):
        tr_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold
        tr_data = train_df[tr_mask]

        for c_col in cat_cols:
            for n_col in num_cols:
                grp = tr_data.groupby(c_col)[n_col].agg(["mean", "std", "min", "max"])
                grp["std"] = grp["std"].fillna(0.0)

                for stat in ["mean", "std", "min", "max"]:
                    smooth_map = grp[stat].to_dict()
                    default_val = float(tr_data[n_col].mean()) if stat != "std" else 0.0
                    train_df.loc[val_mask, f"{c_col}_{n_col}_grp_{stat}"] = (
                        train_df.loc[val_mask, c_col]
                        .map(smooth_map)
                        .fillna(default_val)
                        .astype(np.float32)
                    )

    for c_col in cat_cols:
        for n_col in num_cols:
            grp_full = train_df.groupby(c_col)[n_col].agg(["mean", "std", "min", "max"])
            grp_full["std"] = grp_full["std"].fillna(0.0)

            for stat in ["mean", "std", "min", "max"]:
                smooth_map = grp_full[stat].to_dict()
                default_val = float(train_df[n_col].mean()) if stat != "std" else 0.0
                test_df[f"{c_col}_{n_col}_grp_{stat}"] = (
                    test_df[c_col]
                    .map(smooth_map)
                    .fillna(default_val)
                    .astype(np.float32)
                )

one_hot_cats = [c for c in cat_cols if train_df[c].nunique() <= 20]
if len(one_hot_cats) > 0:
    combined = pd.concat([train_df[one_hot_cats], test_df[one_hot_cats]], axis=0)
    combined_ohe = pd.get_dummies(
        combined, columns=one_hot_cats, drop_first=True, dtype=np.float32
    )

    n_train = len(train_df)
    train_ohe = combined_ohe.iloc[:n_train].reset_index(drop=True)
    test_ohe = combined_ohe.iloc[n_train:].reset_index(drop=True)

    train_df = train_df.drop(columns=one_hot_cats)
    test_df = test_df.drop(columns=one_hot_cats)

    train_df = pd.concat([train_df, train_ohe], axis=1)
    test_df = pd.concat([test_df, test_ohe], axis=1)

remaining_objs = train_df.select_dtypes(include=["object"]).columns.tolist()
train_df = train_df.drop(columns=remaining_objs)
test_df = test_df.drop(columns=remaining_objs)

features = [c for c in train_df.columns if c not in ignore_cols]

train_df[features] = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_df[features] = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

X_test = test_df[features].values.astype(np.float32)


# -----------------------------------------------------------------------------
# 4. PyTorch Tabular Architecture Definitions
# -----------------------------------------------------------------------------
class SwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features * 2)

    def forward(self, x):
        x_proj = self.fc(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class ResNetBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.swiglu = SwiGLU(dim, dim)
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.swiglu(x)
        out = self.fc(out)
        out = self.drop(out)
        return self.norm(residual + out)


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def train_pytorch_model(
    X_tr, y_tr, X_val, y_val, X_te, input_dim, epochs=30, batch_size=256
):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout=0.1
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_y)
                val_count += len(batch_y)

        val_loss /= max(val_count, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        val_preds = model(val_tensor).cpu().numpy()

        te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        te_preds = model(te_tensor).cpu().numpy()

    return val_preds, te_preds


# -----------------------------------------------------------------------------
# 5. Model Training & Cross-Validation Loop
# -----------------------------------------------------------------------------
oof_lgb = np.zeros(len(train_df), dtype=np.float32)
oof_xgb = np.zeros(len(train_df), dtype=np.float32)
oof_et = np.zeros(len(train_df), dtype=np.float32)
oof_rf = np.zeros(len(train_df), dtype=np.float32)
oof_nn = np.zeros(len(train_df), dtype=np.float32)

test_preds_lgb = np.zeros(len(test_df), dtype=np.float32)
test_preds_xgb = np.zeros(len(test_df), dtype=np.float32)
test_preds_et = np.zeros(len(test_df), dtype=np.float32)
test_preds_rf = np.zeros(len(test_df), dtype=np.float32)
test_preds_nn = np.zeros(len(test_df), dtype=np.float32)

if HAS_CATBOOST:
    oof_cat = np.zeros(len(train_df), dtype=np.float32)
    test_preds_cat = np.zeros(len(test_df), dtype=np.float32)

for fold in range(num_folds):
    tr_idx = train_df["fold"] != fold
    val_idx = train_df["fold"] == fold

    X_tr = train_df.loc[tr_idx, features].values.astype(np.float32)
    y_tr = train_df.loc[tr_idx, target_col].values.astype(np.float32)

    X_val = train_df.loc[val_idx, features].values.astype(np.float32)
    y_val = train_df.loc[val_idx, target_col].values.astype(np.float32)

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test) / num_folds

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test) / num_folds

    # CatBoost
    if HAS_CATBOOST:
        cat_model = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx] = cat_model.predict(X_val)
        test_preds_cat += cat_model.predict(X_test) / num_folds

    # ExtraTrees
    et_model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict(X_val)
    test_preds_et += et_model.predict(X_test) / num_folds

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_model.predict(X_val)
    test_preds_rf += rf_model.predict(X_test) / num_folds

    # PyTorch Tabular ResNet
    nn_val_pred, nn_test_pred = train_pytorch_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_test,
        input_dim=len(features),
        epochs=30,
        batch_size=256,
    )
    oof_nn[val_idx] = nn_val_pred
    test_preds_nn += nn_test_pred / num_folds


# -----------------------------------------------------------------------------
# 6. Meta-Learner Stacking & Submission Output
# -----------------------------------------------------------------------------
if HAS_CATBOOST:
    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf, oof_nn])
    test_stack = np.column_stack(
        [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_rf, test_preds_nn]
    )
else:
    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_et, oof_rf, oof_nn])
    test_stack = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_et, test_preds_rf, test_preds_nn])

meta_learner = Ridge(alpha=1.0, positive=True)
meta_learner.fit(oof_stack, y_train)

oof_preds = meta_learner.predict(oof_stack)
test_preds = meta_learner.predict(test_stack)

oof_preds = np.clip(oof_preds, a_min=0.0, a_max=None)
test_preds = np.clip(test_preds, a_min=0.0, a_max=None)

sub_ids = test_ids
submission_df = pd.DataFrame({"id": sub_ids, "Bus age": test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

final_score = float(root_mean_squared_error(y_train, oof_preds))
print(f"Final Validation Score: {final_score}")
