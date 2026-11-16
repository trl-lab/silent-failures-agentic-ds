import copy
import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

# Ensure required output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Set global random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y_full_arr = train_df["Transported"].astype(bool).values
n_train = len(train_df)
n_test = len(test_df)

# Concatenate for domain feature engineering and group imputations
df = pd.concat([train_df.drop(columns=["Transported"]), test_df], ignore_index=True)

# 2. Extract Structural Identifiers & Parse Categoricals
df["Group_Id"] = df["PassengerId"].apply(lambda x: str(x).split("_")[0]).astype(int)
df["Group_Pos"] = df["PassengerId"].apply(lambda x: str(x).split("_")[1]).astype(int)

group_sizes = df["Group_Id"].value_counts()
df["Group_Size"] = df["Group_Id"].map(group_sizes)
df["Is_Alone"] = (df["Group_Size"] == 1).astype(int)


def parse_cabin(val):
    if pd.isna(val):
        return np.nan, np.nan, np.nan
    parts = str(val).split("/")
    if len(parts) == 3:
        num = float(parts[1]) if parts[1].isdigit() else np.nan
        return parts[0], num, parts[2]
    return np.nan, np.nan, np.nan


cabin_parsed = df["Cabin"].apply(parse_cabin)
df["Cabin_Deck"] = [x[0] for x in cabin_parsed]
df["Cabin_Num"] = [x[1] for x in cabin_parsed]
df["Cabin_Side"] = [x[2] for x in cabin_parsed]

cabin_counts = df["Cabin"].value_counts()
df["Cabin_Size"] = df["Cabin"].map(cabin_counts).fillna(1)


def parse_surname(val):
    if pd.isna(val):
        return np.nan
    parts = str(val).strip().split()
    return parts[-1] if len(parts) > 1 else np.nan


df["Surname"] = df["Name"].apply(parse_surname)
surname_counts = df["Surname"].value_counts()
df["Family_Size"] = df["Surname"].map(surname_counts).fillna(1)

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Add explicit missingness binary indicators before domain imputation
for col in spend_cols + ["Age", "CryoSleep", "Cabin", "HomePlanet", "VIP", "Destination"]:
    df[f"{col}_isnan"] = df[col].isna().astype(int)
df["Missing_Count"] = df[[f"{col}_isnan" for col in spend_cols + ["Age", "CryoSleep", "Cabin", "HomePlanet", "VIP", "Destination"]]].sum(axis=1)

# Spatial cabin region features & relative position
df["Cabin_Region_50"] = df["Cabin_Num"] // 50
df["Cabin_Region_100"] = df["Cabin_Num"] // 100

deck_side_max = df.groupby(["Cabin_Deck", "Cabin_Side"])["Cabin_Num"].transform("max")
deck_side_min = df.groupby(["Cabin_Deck", "Cabin_Side"])["Cabin_Num"].transform("min")
df["Cabin_Num_Rel_Deck_Side"] = (df["Cabin_Num"] - deck_side_min) / (deck_side_max - deck_side_min + 1e-5)

# 3. Domain Rule-Based Imputations
known_spend_sum = df[spend_cols].fillna(0).sum(axis=1)
df.loc[known_spend_sum > 0, "CryoSleep"] = False

for col in spend_cols:
    df.loc[(df["CryoSleep"] == True) | (df["Age"] == 0), col] = 0.0

group_homeplanet = df.groupby("Group_Id")["HomePlanet"].first()
df["HomePlanet"] = df["HomePlanet"].fillna(df["Group_Id"].map(group_homeplanet))

surname_homeplanet = df.groupby("Surname")["HomePlanet"].first()
df["HomePlanet"] = df["HomePlanet"].fillna(df["Surname"].map(surname_homeplanet))

df.loc[
    df["HomePlanet"].isna() & df["Cabin_Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"
] = "Europa"
df.loc[df["HomePlanet"].isna() & (df["Cabin_Deck"] == "G"), "HomePlanet"] = "Earth"

df.loc[df["CryoSleep"].isna() & (df[spend_cols].sum(axis=1) == 0), "CryoSleep"] = True
df["CryoSleep"] = df["CryoSleep"].astype(int)

for col in spend_cols:
    df[col] = df[col].fillna(0.0)

df["Age"] = df.groupby("HomePlanet")["Age"].transform(lambda x: x.fillna(x.median()))

# 4. Domain Feature Construction
df["Total_Spend"] = df[spend_cols].sum(axis=1)
df["Zero_Spend"] = (df["Total_Spend"] == 0).astype(int)
df["Luxury_Spend"] = df["Spa"] + df["VRDeck"] + df["FoodCourt"]
df["Essential_Spend"] = df["RoomService"] + df["ShoppingMall"]
df["Luxury_to_Total_Ratio"] = df["Luxury_Spend"] / (df["Total_Spend"] + 1.0)

for col in spend_cols + ["Total_Spend", "Luxury_Spend", "Essential_Spend"]:
    df[f"{col}_log"] = np.log1p(df[col])

for col in spend_cols:
    df[f"{col}_ratio"] = df[col] / (df["Total_Spend"] + 1.0)

df["Is_Child"] = (df["Age"] < 12).astype(int)
df["Is_Senior"] = (df["Age"] >= 60).astype(int)

df["Group_Spend_Mean"] = df.groupby("Group_Id")["Total_Spend"].transform("mean")
df["Group_Spend_Sum"] = df.groupby("Group_Id")["Total_Spend"].transform("sum")

# Fine-grained group and cabin statistics
df["Group_Zero_Spend_Count"] = df.groupby("Group_Id")["Zero_Spend"].transform("sum")
df["Group_Spend_Max"] = df.groupby("Group_Id")["Total_Spend"].transform("max")
df["Group_Spend_Min"] = df.groupby("Group_Id")["Total_Spend"].transform("min")
df["Group_Spend_Var"] = df.groupby("Group_Id")["Total_Spend"].transform("var").fillna(0)
df["Group_Cryo_Ratio"] = df.groupby("Group_Id")["CryoSleep"].transform("mean")

deck_spend_mean = df.groupby("Cabin_Deck")["Total_Spend"].transform("mean")
df["Spend_vs_Deck_Mean"] = df["Total_Spend"] / (deck_spend_mean + 1.0)

planet_spend_mean = df.groupby("HomePlanet")["Total_Spend"].transform("mean")
df["Spend_vs_Planet_Mean"] = df["Total_Spend"] / (planet_spend_mean + 1.0)

df["Services_Used_Count"] = (df[spend_cols] > 0).sum(axis=1)

for col in ["Cabin_Deck", "Cabin_Side", "Destination", "HomePlanet", "Surname"]:
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq).fillna(0)

df["HomePlanet_Deck"] = (
    df["HomePlanet"].astype(str) + "_" + df["Cabin_Deck"].astype(str)
)
df["Deck_Side"] = df["Cabin_Deck"].astype(str) + "_" + df["Cabin_Side"].astype(str)

drop_cols = ["PassengerId", "Cabin", "Name", "Surname", "Group_Id"]
features = [c for c in df.columns if c not in drop_cols]

train_processed = df.iloc[:n_train][features].copy()
test_processed = df.iloc[n_train:][features].copy()

# Out-of-fold target encoding for categorical combinations
te_cols = ["Deck_Side", "HomePlanet_Deck", "Destination"]
te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for c in te_cols:
    train_processed[f"{c}_te"] = np.nan
    test_processed[f"{c}_te"] = 0.0

for tr_idx, va_idx in te_skf.split(train_processed, y_full_arr):
    tr_sub = train_processed.iloc[tr_idx]
    va_sub = train_processed.iloc[va_idx]
    y_tr_sub = pd.Series(y_full_arr[tr_idx], index=tr_sub.index)

    for c in te_cols:
        means = y_tr_sub.groupby(tr_sub[c]).mean()
        train_processed.iloc[va_idx, train_processed.columns.get_loc(f"{c}_te")] = va_sub[c].map(means)

global_mean = float(y_full_arr.mean())
for c in te_cols:
    train_processed[f"{c}_te"] = train_processed[f"{c}_te"].fillna(global_mean)
    global_means = pd.Series(y_full_arr, index=train_processed.index).groupby(train_processed[c]).mean()
    test_processed[f"{c}_te"] = test_processed[c].map(global_means).fillna(global_mean)

# Save engineered features for record
df.iloc[:n_train].to_parquet("./working/train_engineered.parquet")
df.iloc[n_train:].to_parquet("./working/test_engineered.parquet")

cat_cols = [
    "HomePlanet",
    "Destination",
    "Cabin_Deck",
    "Cabin_Side",
    "VIP",
    "HomePlanet_Deck",
    "Deck_Side",
]

for col in cat_cols:
    train_processed[col] = train_processed[col].astype(str)
    test_processed[col] = test_processed[col].astype(str)

num_cols = [c for c in train_processed.columns if c not in cat_cols]

# 5. Model Input Representations
# Neural Network Encodings
X_cat_encoded = np.zeros((n_train, len(cat_cols)), dtype=np.int64)
X_test_cat_encoded = np.zeros((n_test, len(cat_cols)), dtype=np.int64)
cat_cardinalities = []

for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    combined_vals = pd.concat([train_processed[col], test_processed[col]]).astype(str)
    le.fit(combined_vals)
    X_cat_encoded[:, i] = le.transform(train_processed[col].astype(str))
    X_test_cat_encoded[:, i] = le.transform(test_processed[col].astype(str))
    cat_cardinalities.append(len(le.classes_))

X_num_array = train_processed[num_cols].fillna(0).values.astype(np.float32)
X_test_num_array = test_processed[num_cols].fillna(0).values.astype(np.float32)

# GBDT Categorical Formatting
X_gbdt = train_processed.copy()
X_test_gbdt = test_processed.copy()
for col in cat_cols:
    X_gbdt[col] = X_gbdt[col].astype("category")
    X_test_gbdt[col] = X_test_gbdt[col].astype("category")


# 6. PyTorch Neural Architecture Definition
class TabularResBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_numerical_features: int,
        cat_cardinalities: list,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        if cat_cardinalities:
            for card in cat_cardinalities:
                emb_dim = min(50, max(4, (card + 1) // 2))
                self.embeddings.append(nn.Embedding(card + 1, emb_dim))
                total_emb_dim += emb_dim

        self.num_numerical_features = num_numerical_features
        self.num_bn = (
            nn.BatchNorm1d(num_numerical_features)
            if num_numerical_features > 0
            else None
        )

        in_dim = num_numerical_features + total_emb_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        inputs = []
        if self.num_bn is not None and x_num is not None and x_num.shape[1] > 0:
            inputs.append(self.num_bn(x_num))
        if x_cat is not None and len(self.embeddings) > 0:
            for i, emb in enumerate(self.embeddings):
                inputs.append(emb(x_cat[:, i]))
        x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# 7. Stratified 5-Fold Cross-Validation & Ensemble Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgbm = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_preds_lgbm = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cat = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lgb_params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 4,
}

xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "n_jobs": 4,
    "enable_categorical": True,
}

cb_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "eval_metric": "Accuracy",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
    "thread_count": 4,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_gbdt, y_full_arr)):
    X_tr_gb, y_tr = X_gbdt.iloc[train_idx], y_full_arr[train_idx]
    X_va_gb, y_va = X_gbdt.iloc[val_idx], y_full_arr[val_idx]

    # --- LightGBM ---
    lgbm_model = lgb.LGBMClassifier(**lgb_params)
    lgbm_model.fit(
        X_tr_gb,
        y_tr,
        eval_set=[(X_va_gb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgbm[val_idx] = lgbm_model.predict_proba(X_va_gb)[:, 1]
    test_preds_lgbm += lgbm_model.predict_proba(X_test_gbdt)[:, 1] / 5.0

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr_gb, y_tr, eval_set=[(X_va_gb, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_gb)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_gbdt)[:, 1] / 5.0

    # --- CatBoost ---
    X_tr_cb = X_tr_gb.copy()
    X_va_cb = X_va_gb.copy()
    X_test_cb = X_test_gbdt.copy()
    for col in cat_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols,
        verbose=False,
    )
    oof_cat[val_idx] = cb_model.predict_proba(X_va_cb)[:, 1]
    test_preds_cat += cb_model.predict_proba(X_test_cb)[:, 1] / 5.0

    # --- PyTorch TabularResNet ---
    nn_model = TabularResNet(
        num_numerical_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_num_array[train_idx], dtype=torch.float32),
        torch.tensor(X_cat_encoded[train_idx], dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2
    )

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(40):
        nn_model.train()
        for b_num, b_cat, b_y in train_loader:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(b_num, b_cat)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            v_num = torch.tensor(
                X_num_array[val_idx], dtype=torch.float32, device=device
            )
            v_cat = torch.tensor(
                X_cat_encoded[val_idx], dtype=torch.long, device=device
            )
            v_y = torch.tensor(y_va, dtype=torch.float32, device=device)
            v_logits = nn_model(v_num, v_cat)
            v_loss = criterion(v_logits, v_y).item()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_nn_weights = copy.deepcopy(nn_model.state_dict())

    nn_model.load_state_dict(best_nn_weights)
    nn_model.eval()
    with torch.no_grad():
        v_num = torch.tensor(X_num_array[val_idx], dtype=torch.float32, device=device)
        v_cat = torch.tensor(X_cat_encoded[val_idx], dtype=torch.long, device=device)
        oof_nn[val_idx] = torch.sigmoid(nn_model(v_num, v_cat)).cpu().numpy()

        t_num = torch.tensor(X_test_num_array, dtype=torch.float32, device=device)
        t_cat = torch.tensor(X_test_cat_encoded, dtype=torch.long, device=device)
        test_preds_nn += torch.sigmoid(nn_model(t_num, t_cat)).cpu().numpy() / 5.0

# 8. Optimization / Nelder-Mead & Powell Probability Weighting & Threshold Search
oof_matrix = np.column_stack([oof_lgbm, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_preds_lgbm, test_preds_xgb, test_preds_cat, test_preds_nn])

def ensemble_objective(params, oof_preds, y_true):
    weights = params[:4]
    threshold = params[4]
    exp_w = np.exp(weights - np.max(weights))
    w = exp_w / np.sum(exp_w)
    blend_probs = np.dot(oof_preds, w)
    preds = (blend_probs > threshold).astype(int)
    return -accuracy_score(y_true, preds)

initial_params = [0.0, 0.0, 0.0, 0.0, 0.5]

res_nm = minimize(
    ensemble_objective,
    initial_params,
    args=(oof_matrix, y_full_arr),
    method="Nelder-Mead",
    options={"maxiter": 1000, "disp": False},
)

res_powell = minimize(
    ensemble_objective,
    res_nm.x,
    args=(oof_matrix, y_full_arr),
    method="Powell",
    options={"maxiter": 1000, "disp": False},
)

best_params = res_powell.x if res_powell.fun < res_nm.fun else res_nm.x
exp_w = np.exp(best_params[:4] - np.max(best_params[:4]))
best_weights = exp_w / np.sum(exp_w)
best_threshold = best_params[4]

best_oof_blend = np.dot(oof_matrix, best_weights)

# Fine-grained decision threshold grid search on OOF predictions
best_thresh_grid = best_threshold
best_acc_grid = accuracy_score(y_full_arr, (best_oof_blend > best_threshold).astype(int))

for t in np.linspace(0.3, 0.7, 401):
    acc = accuracy_score(y_full_arr, (best_oof_blend > t).astype(int))
    if acc > best_acc_grid:
        best_acc_grid = acc
        best_thresh_grid = t

best_threshold = best_thresh_grid
meta_val_acc = best_acc_grid

# 9. Format Submission & Write Output File
final_test_blend = np.dot(test_matrix, best_weights)
final_test_preds = (final_test_blend > best_threshold).astype(bool)

submission_df = pd.DataFrame({"id": np.arange(n_test), "Transported": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

# 10. Print Final Score Parser Requirement
print(f"Final Validation Score: {meta_val_acc}")