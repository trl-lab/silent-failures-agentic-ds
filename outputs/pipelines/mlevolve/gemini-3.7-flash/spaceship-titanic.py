import json
import os
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. Directory Setup and Data Loading
# -----------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

if os.path.exists(train_path) and os.path.exists(test_path):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
else:
    df_train = pd.read_csv("./input/original/train.csv")
    df_test = pd.read_csv("./input/original/test.csv")

df_train["is_train"] = 1
df_test["is_train"] = 0
target_col = "Transported"

if target_col in df_train.columns:
    df_train[target_col] = df_train[target_col].astype(bool)

df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# -----------------------------------------------------------------------------
# 2. Relational & Entity Parsing (Group, Cabin, Name)
# -----------------------------------------------------------------------------
passenger_split = df_all["PassengerId"].str.split("_", expand=True)
df_all["GroupId"] = passenger_split[0]
df_all["GroupMemberId"] = passenger_split[1].astype(int)

group_sizes = df_all["GroupId"].value_counts()
df_all["GroupSize"] = df_all["GroupId"].map(group_sizes)
df_all["IsAlone"] = (df_all["GroupSize"] == 1).astype(int)

# Parse Cabin (Deck/Num/Side)
cabin_split = df_all["Cabin"].str.split("/", expand=True)
df_all["CabinDeck"] = cabin_split[0]
df_all["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
df_all["CabinSide"] = cabin_split[2]

# Parse Name into Surname
df_all["Surname"] = df_all["Name"].str.split().str[-1]
surname_counts = df_all["Surname"].value_counts()
df_all["FamilySize"] = df_all["Surname"].map(surname_counts).fillna(1).astype(int)

# Cabin occupancy
cabin_counts = df_all["Cabin"].value_counts()
df_all["CabinOccupancy"] = df_all["Cabin"].map(cabin_counts).fillna(0).astype(int)

# -----------------------------------------------------------------------------
# 3. Domain Heuristics & Deterministic Rule Imputations
# -----------------------------------------------------------------------------
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Rule 0: Minors (Age <= 12) spend 0 across all amenities and cannot have VIP status
df_all.loc[df_all["Age"] <= 12, spend_cols] = 0.0
df_all.loc[df_all["Age"] <= 12, "VIP"] = False

# Rule 1: If CryoSleep == True, passenger spends 0
for col in spend_cols:
    df_all.loc[df_all["CryoSleep"] == True, col] = 0.0

# Rule 2: If TotalSpend > 0, CryoSleep must be False
known_spend = df_all[spend_cols].sum(axis=1)
df_all.loc[(df_all["CryoSleep"].isna()) & (known_spend > 0), "CryoSleep"] = False

# Rule 3: Group members frequently share HomePlanet and Cabin features
group_planet_map = (
    df_all.dropna(subset=["HomePlanet"]).groupby("GroupId")["HomePlanet"].first()
)
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(
    df_all["GroupId"].map(group_planet_map)
)

deck_planet_rules = {
    "A": "Europa",
    "B": "Europa",
    "C": "Europa",
    "T": "Europa",
    "G": "Earth",
}
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(
    df_all["CabinDeck"].map(deck_planet_rules)
)

surname_planet_map = (
    df_all.dropna(subset=["HomePlanet"]).groupby("Surname")["HomePlanet"].first()
)
df_all["HomePlanet"] = df_all["HomePlanet"].fillna(
    df_all["Surname"].map(surname_planet_map)
)

group_surname_map = (
    df_all.dropna(subset=["Surname"]).groupby("GroupId")["Surname"].first()
)
df_all["Surname"] = df_all["Surname"].fillna(df_all["GroupId"].map(group_surname_map))

group_deck_map = (
    df_all.dropna(subset=["CabinDeck"]).groupby("GroupId")["CabinDeck"].first()
)
df_all["CabinDeck"] = df_all["CabinDeck"].fillna(df_all["GroupId"].map(group_deck_map))

group_side_map = (
    df_all.dropna(subset=["CabinSide"]).groupby("GroupId")["CabinSide"].first()
)
df_all["CabinSide"] = df_all["CabinSide"].fillna(df_all["GroupId"].map(group_side_map))

group_dest_map = (
    df_all.dropna(subset=["Destination"]).groupby("GroupId")["Destination"].first()
)
df_all["Destination"] = df_all["Destination"].fillna(
    df_all["GroupId"].map(group_dest_map)
)

# Rule 4: Earth passengers never have VIP status
df_all.loc[df_all["HomePlanet"] == "Earth", "VIP"] = df_all.loc[
    df_all["HomePlanet"] == "Earth", "VIP"
].fillna(False)

# -----------------------------------------------------------------------------
# 4. Expenditure & Interaction Feature Engineering
# -----------------------------------------------------------------------------
df_all["AmenityCount"] = (df_all[spend_cols] > 0).sum(axis=1).astype(int)

for col in spend_cols:
    df_all[col] = df_all[col].fillna(0.0)

df_all["TotalSpend"] = df_all[spend_cols].sum(axis=1)
df_all["ZeroSpend"] = (df_all["TotalSpend"] == 0).astype(int)
df_all["LuxurySpend"] = df_all["Spa"] + df_all["VRDeck"] + df_all["RoomService"]
df_all["ServiceSpend"] = df_all["FoodCourt"] + df_all["ShoppingMall"]
df_all["LuxuryToServiceRatio"] = (df_all["LuxurySpend"] + 1.0) / (
    df_all["ServiceSpend"] + 1.0
)
df_all["Luxury_Prop"] = (df_all["LuxurySpend"] / (df_all["TotalSpend"] + 1.0)).astype(
    np.float32
)
df_all["Service_Prop"] = (df_all["ServiceSpend"] / (df_all["TotalSpend"] + 1.0)).astype(
    np.float32
)

for col in spend_cols + ["TotalSpend", "LuxurySpend", "ServiceSpend"]:
    df_all[f"Log_{col}"] = np.log1p(np.maximum(df_all[col], 0))

for col in spend_cols:
    df_all[f"{col}_Ratio"] = (df_all[col] / (df_all["TotalSpend"] + 1.0)).astype(
        np.float32
    )

df_all["Age"] = df_all["Age"].fillna(df_all["Age"].median())
df_all["IsChild"] = (df_all["Age"] <= 12).astype(int)
df_all["IsElderly"] = (df_all["Age"] >= 60).astype(int)
df_all["AgeGroup"] = pd.cut(
    df_all["Age"],
    bins=[-1, 12, 18, 30, 50, 120],
    labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
).astype(str)

df_all["CabinNum"] = df_all["CabinNum"].fillna(df_all["CabinNum"].median())
df_all["CabinRegion"] = (df_all["CabinNum"] // 300).astype(int)
df_all["CabinBatch"] = (df_all["CabinNum"] // 100).astype(int)

# Normalized spatial deck coordinate: CabinNum / Max(CabinNum) per Deck
deck_max = df_all.groupby("CabinDeck")["CabinNum"].transform("max")
df_all["CabinNum_Deck_Ratio"] = (df_all["CabinNum"] / deck_max.replace(0, 1)).fillna(0.0)

df_all["Deck_Side"] = (
    df_all["CabinDeck"].fillna("Unknown") + "_" + df_all["CabinSide"].fillna("Unknown")
)
df_all["Deck_Region"] = (
    df_all["CabinDeck"].fillna("Unknown") + "_" + df_all["CabinRegion"].astype(str)
)
df_all["Deck_Side_Region"] = (
    df_all["CabinDeck"].fillna("Unknown")
    + "_"
    + df_all["CabinSide"].fillna("Unknown")
    + "_"
    + df_all["CabinRegion"].astype(str)
)
df_all["Planet_Deck"] = (
    df_all["HomePlanet"].fillna("Unknown") + "_" + df_all["CabinDeck"].fillna("Unknown")
)

group_stats = (
    df_all.groupby("GroupId")
    .agg(
        Group_Mean_Spend=("TotalSpend", "mean"),
        Group_Max_Spend=("TotalSpend", "max"),
        Group_Cryo_Count=("CryoSleep", lambda x: (x == True).sum()),
        Group_Child_Count=("IsChild", "sum"),
    )
    .reset_index()
)

df_all = df_all.merge(group_stats, on="GroupId", how="left")
df_all["Group_Total_Spend"] = df_all["Group_Mean_Spend"] * df_all["GroupSize"]
df_all["Group_PerCapita_Spend"] = df_all["Group_Mean_Spend"] / df_all["GroupSize"]
df_all["Spend_Diff_GroupMean"] = df_all["TotalSpend"] - df_all["Group_Mean_Spend"]

cat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "Deck_Side",
    "Deck_Region",
    "Deck_Side_Region",
    "Planet_Deck",
    "AgeGroup",
]
for col in cat_cols:
    df_all[col] = df_all[col].fillna("Unknown").astype(str)

# -----------------------------------------------------------------------------
# 5. Encoding & Feature Preprocessing
# -----------------------------------------------------------------------------
encoded_cat_cols = []
for col in cat_cols:
    le = LabelEncoder()
    df_all[f"{col}_encoded"] = le.fit_transform(df_all[col])
    encoded_cat_cols.append(f"{col}_encoded")

num_cols = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "LuxurySpend",
    "ServiceSpend",
    "LuxuryToServiceRatio",
    "Luxury_Prop",
    "Service_Prop",
    "AmenityCount",
    "Log_RoomService",
    "Log_FoodCourt",
    "Log_ShoppingMall",
    "Log_Spa",
    "Log_VRDeck",
    "Log_TotalSpend",
    "Log_LuxurySpend",
    "Log_ServiceSpend",
    "RoomService_Ratio",
    "FoodCourt_Ratio",
    "ShoppingMall_Ratio",
    "Spa_Ratio",
    "VRDeck_Ratio",
    "GroupSize",
    "IsAlone",
    "FamilySize",
    "CabinOccupancy",
    "CabinRegion",
    "CabinBatch",
    "IsChild",
    "IsElderly",
    "CabinNum",
    "CabinNum_Deck_Ratio",
    "Group_Mean_Spend",
    "Group_Max_Spend",
    "Group_Total_Spend",
    "Group_PerCapita_Spend",
    "Group_Cryo_Count",
    "Group_Child_Count",
    "Spend_Diff_GroupMean",
]

# -----------------------------------------------------------------------------
# 6. Split Back to Train/Test & Generate Stratified Folds
# -----------------------------------------------------------------------------
df_train_proc = df_all[df_all["is_train"] == 1].copy().reset_index(drop=True)
df_test_proc = df_all[df_all["is_train"] == 0].copy().reset_index(drop=True)

df_train_proc["Target"] = df_train_proc[target_col].astype(int)

# Initialize fellow target statistics columns
for col in ["Fellow_Group_Rate", "Fellow_Group_Count", "Fellow_Cabin_Rate", "Fellow_Cabin_Count"]:
    df_train_proc[col] = 0.5 if "Rate" in col else 0.0
    df_test_proc[col] = 0.5 if "Rate" in col else 0.0

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
df_train_proc["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(df_train_proc, df_train_proc["Target"])
):
    df_train_proc.loc[val_idx, "fold"] = fold

feature_names = encoded_cat_cols + num_cols
cb_feature_names = cat_cols + num_cols


# -----------------------------------------------------------------------------
# 7. Model Architectures & Parameter Configurations
# -----------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_continuous_features: int,
        embedding_dims: List[int] = None,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0

        for i, card in enumerate(cat_cardinalities):
            if embedding_dims and i < len(embedding_dims):
                emb_dim = embedding_dims[i]
            else:
                emb_dim = max(4, min(50, (card + 1) // 2 + 2))
            self.embeddings.append(nn.Embedding(card, emb_dim))
            total_emb_dim += emb_dim

        self.num_bn = (
            nn.BatchNorm1d(num_continuous_features)
            if num_continuous_features > 0
            else nn.Identity()
        )

        in_dim = total_emb_dim + num_continuous_features
        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embedded_parts = []
        if len(self.embeddings) > 0 and x_cat is not None:
            for i, emb_layer in enumerate(self.embeddings):
                embedded_parts.append(emb_layer(x_cat[:, i]))
            cat_rep = torch.cat(embedded_parts, dim=1)
        else:
            cat_rep = torch.empty(x_num.size(0), 0, device=x_num.device)

        if x_num is not None and x_num.size(1) > 0:
            num_rep = self.num_bn(x_num)
            x = torch.cat([cat_rep, num_rep], dim=1) if cat_rep.size(1) > 0 else num_rep
        else:
            x = cat_rep

        h = self.input_projection(x)
        for block in self.res_blocks:
            h = block(h)

        logits = self.head(h).squeeze(-1)
        return logits


def build_neural_model_and_optimizer(
    cat_cardinalities: List[int],
    num_continuous_features: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = TabularResNet(
        cat_cardinalities=cat_cardinalities,
        num_continuous_features=num_continuous_features,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.25,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    return model, criterion, optimizer, scheduler


def get_lgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": seed,
        "n_estimators": 2000,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "min_child_weight": 2,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.85,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# -----------------------------------------------------------------------------
# 8. 5-Fold Cross-Validation Training & Out-of-Fold Inference
# -----------------------------------------------------------------------------
n_train = len(df_train_proc)
n_test = len(df_test_proc)
n_folds = df_train_proc["fold"].nunique()
device = "cuda" if torch.cuda.is_available() else "cpu"

cat_cardinalities = [
    max(int(df_train_proc[c].max()), int(df_test_proc[c].max())) + 1
    for c in encoded_cat_cols
]

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cat = np.zeros(n_train, dtype=np.float64)
oof_nn = np.zeros(n_train, dtype=np.float64)

test_lgb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)
test_cat = np.zeros(n_test, dtype=np.float64)
test_nn = np.zeros(n_test, dtype=np.float64)

y_all = df_train_proc["Target"].values

print(
    f"Starting 5-Fold CV Training on {n_train} samples with {len(feature_names)} features..."
)

for fold in range(n_folds):
    train_mask = df_train_proc["fold"] != fold
    val_mask = df_train_proc["fold"] == fold

    X_train_df = df_train_proc.loc[train_mask]
    X_val_df = df_train_proc.loc[val_mask]

    y_train = X_train_df["Target"].values
    y_val = X_val_df["Target"].values
    val_indices = np.where(val_mask)[0]

    # Tree-based Data
    X_tr_trees = X_train_df[feature_names].values
    X_va_trees = X_val_df[feature_names].values
    X_te_trees = df_test_proc[feature_names].values

    # LightGBM
    lgb_params = get_lgb_params(seed=42 + fold)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr_trees,
        y_train,
        eval_set=[(X_va_trees, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_indices] = lgb_model.predict_proba(X_va_trees)[:, 1]
    test_lgb += lgb_model.predict_proba(X_te_trees)[:, 1] / n_folds

    # XGBoost
    xgb_params = get_xgb_params(seed=42 + fold)
    xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(
        X_tr_trees,
        y_train,
        eval_set=[(X_va_trees, y_val)],
        verbose=False,
    )
    oof_xgb[val_indices] = xgb_model.predict_proba(X_va_trees)[:, 1]
    test_xgb += xgb_model.predict_proba(X_te_trees)[:, 1] / n_folds

    # CatBoost
    cb_params = get_catboost_params(seed=42 + fold)
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_tr_trees,
        y_train,
        eval_set=(X_va_trees, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cat[val_indices] = cb_model.predict_proba(X_va_trees)[:, 1]
    test_cat += cb_model.predict_proba(X_te_trees)[:, 1] / n_folds

    # Neural Network Data (Fit scaler strictly on training fold)
    scaler = StandardScaler()
    X_tr_num_scaled = scaler.fit_transform(X_train_df[num_cols].values)
    X_va_num_scaled = scaler.transform(X_val_df[num_cols].values)
    X_te_num_scaled = scaler.transform(df_test_proc[num_cols].values)

    X_tr_cat = torch.tensor(X_train_df[encoded_cat_cols].values, dtype=torch.long)
    X_va_cat = torch.tensor(X_val_df[encoded_cat_cols].values, dtype=torch.long)
    X_te_cat = torch.tensor(df_test_proc[encoded_cat_cols].values, dtype=torch.long)

    X_tr_num = torch.tensor(X_tr_num_scaled, dtype=torch.float32)
    X_va_num = torch.tensor(X_va_num_scaled, dtype=torch.float32)
    X_te_num = torch.tensor(X_te_num_scaled, dtype=torch.float32)

    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    y_va_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_cat, X_tr_num, y_tr_t)
    val_ds = TensorDataset(X_va_cat, X_va_num, y_va_t)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    nn_epochs = 35
    nn_model, criterion, optimizer, scheduler = build_neural_model_and_optimizer(
        cat_cardinalities=cat_cardinalities,
        num_continuous_features=len(num_cols),
        lr=1e-3,
        weight_decay=1e-4,
        epochs=nn_epochs,
        device=device,
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(nn_epochs):
        nn_model.train()
        for b_cat, b_num, b_y in train_loader:
            b_cat = b_cat.to(device)
            b_num = b_num.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            preds = nn_model(b_cat, b_num)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_losses = []
        with torch.no_grad():
            for b_cat, b_num, b_y in val_loader:
                b_cat = b_cat.to(device)
                b_num = b_num.to(device)
                b_y = b_y.to(device)
                p = nn_model(b_cat, b_num)
                val_losses.append(criterion(p, b_y).item())

        mean_val_loss = np.mean(val_losses)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    nn_model.eval()

    with torch.no_grad():
        va_preds = torch.sigmoid(
            nn_model(X_va_cat.to(device), X_va_num.to(device))
        ).cpu()
        oof_nn[val_indices] = va_preds.numpy()

        te_preds = torch.sigmoid(
            nn_model(X_te_cat.to(device), X_te_num.to(device))
        ).cpu()
        test_nn += te_preds.numpy() / n_folds

    fold_acc_lgb = accuracy_score(y_val, (oof_lgb[val_indices] >= 0.5).astype(int))
    fold_acc_xgb = accuracy_score(y_val, (oof_xgb[val_indices] >= 0.5).astype(int))
    fold_acc_cat = accuracy_score(y_val, (oof_cat[val_indices] >= 0.5).astype(int))
    fold_acc_nn = accuracy_score(y_val, (oof_nn[val_indices] >= 0.5).astype(int))
    print(
        f"Fold {fold+1}/{n_folds} Accuracy | LGB: {fold_acc_lgb:.4f} | XGB: {fold_acc_xgb:.4f} | CAT: {fold_acc_cat:.4f} | NN: {fold_acc_nn:.4f}"
    )

# -----------------------------------------------------------------------------
# 9. Ensemble Optimization & Threshold Tuning
# -----------------------------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def ensemble_loss(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = np.clip(np.dot(oof_matrix, w), 1e-7, 1.0 - 1e-7)
    return log_loss(y_all, pred)


init_weights = [0.30, 0.30, 0.25, 0.15]
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.array(init_weights)
)
print(
    f"Optimal Model Ensemble Weights: LGB={opt_weights[0]:.3f}, XGB={opt_weights[1]:.3f}, CAT={opt_weights[2]:.3f}, NN={opt_weights[3]:.3f}"
)

oof_blend = np.dot(oof_matrix, opt_weights)
test_blend = np.dot(test_matrix, opt_weights)

best_threshold = 0.50
best_accuracy = 0.0
for th in np.linspace(0.40, 0.60, 201):
    acc = accuracy_score(y_all, (oof_blend >= th).astype(int))
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = th

print(f"Optimal Decision Threshold: {best_threshold:.4f}")
print(f"Out-of-Fold Overall Accuracy: {best_accuracy:.5f}")

# -----------------------------------------------------------------------------
# 10. Generate Test Predictions & Save Submission
# -----------------------------------------------------------------------------
final_test_predictions = (test_blend >= best_threshold).astype(bool)

submission_df = pd.DataFrame(
    {"id": np.arange(len(df_test_proc)), "Transported": final_test_predictions}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission with shape {submission_df.shape} to {submission_path}")

print(f"Final Validation Score: {best_accuracy}")