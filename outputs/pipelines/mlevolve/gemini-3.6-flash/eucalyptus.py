import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Load Data & Setup Output Paths
# -----------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

target_col = "Utility"
y_raw = train_df[target_col].astype(str).values

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)

train_df = train_df.drop(columns=[target_col], errors="ignore")
train_len = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

if "id" in combined_df.columns:
    combined_df = combined_df.drop(columns=["id"])

# -----------------------------------------------------------------------------
# 2. Domain Feature Engineering & Preprocessing
# -----------------------------------------------------------------------------
num_cols = [
    "Altitude",
    "Rainfall",
    "Frosts",
    "Year",
    "PMCno",
    "DBH",
    "Ht",
    "Surv",
    "Vig",
    "Ins_res",
    "Stem_Fm",
    "Crown_Fm",
    "Brnch_Fm",
]
cat_cols = ["Abbrev", "Rep", "Locality", "Map_Ref", "Latitude", "Sp"]

for c in num_cols:
    if c in combined_df.columns:
        combined_df[c] = pd.to_numeric(combined_df[c], errors="coerce")

for c in cat_cols:
    if c in combined_df.columns:
        combined_df[c] = combined_df[c].astype(str)

missing_vital_cols = ["Vig", "Ins_res", "Stem_Fm", "Crown_Fm", "Brnch_Fm", "DBH", "Ht"]
for c in missing_vital_cols:
    if c in combined_df.columns:
        combined_df[f"{c}_isna"] = combined_df[c].isna().astype(int)

combined_df["num_missing_vitals"] = combined_df[
    [f"{c}_isna" for c in missing_vital_cols]
].sum(axis=1)

med_dict = {c: combined_df.iloc[:train_len][c].median() for c in num_cols if c in combined_df.columns}
for c, med in med_dict.items():
    combined_df[f"{c}_imputed"] = combined_df[c].fillna(med)

dbh = combined_df["DBH_imputed"]
ht = combined_df["Ht_imputed"]
surv = combined_df["Surv_imputed"]
vig = combined_df["Vig_imputed"]
ins = combined_df["Ins_res_imputed"]
stem = combined_df["Stem_Fm_imputed"]
crown = combined_df["Crown_Fm_imputed"]
brnch = combined_df["Brnch_Fm_imputed"]
frost = combined_df["Frosts_imputed"]
rain = combined_df["Rainfall_imputed"]
year = combined_df["Year_imputed"]

combined_df["volume_index"] = (dbh**2) * ht
combined_df["slenderness_ratio"] = ht / (dbh + 1e-3)
combined_df["form_sum"] = stem + crown + brnch
combined_df["form_mean"] = combined_df["form_sum"] / 3.0
combined_df["vitality_index"] = vig * ins * surv
combined_df["overall_quality_index"] = (
    combined_df["vitality_index"] * combined_df["form_sum"]
)
combined_df["stress_index"] = frost / (rain + 1.0)
combined_df["trial_age_proxy"] = 1992.0 - year
combined_df["annual_ht_growth"] = ht / (combined_df["trial_age_proxy"].clip(lower=1.0))
combined_df["annual_dbh_growth"] = dbh / (
    combined_df["trial_age_proxy"].clip(lower=1.0)
)

train_part = combined_df.iloc[:train_len]

# Group-level aggregations (mean, std, deviation, z-score) for species, locality, and year
vital_params = ["Ht", "DBH", "Vig", "Stem_Fm", "Crown_Fm", "Brnch_Fm", "Surv"]
group_cols = ["Sp", "Locality", "Year"]

for grp in group_cols:
    if grp in combined_df.columns:
        for vp in vital_params:
            vp_col = f"{vp}_imputed"
            if vp_col in combined_df.columns:
                grp_mean = train_part.groupby(grp)[vp_col].mean()
                grp_std = train_part.groupby(grp)[vp_col].std().fillna(1.0)

                mean_mapped = combined_df[grp].map(grp_mean).fillna(train_part[vp_col].mean())
                std_mapped = combined_df[grp].map(grp_std).fillna(1.0).replace(0, 1.0)

                combined_df[f"{vp}_mean_by_{grp}"] = mean_mapped
                combined_df[f"{vp}_std_by_{grp}"] = std_mapped
                combined_df[f"{vp}_dev_by_{grp}"] = combined_df[vp_col] - mean_mapped
                combined_df[f"{vp}_zscore_by_{grp}"] = (combined_df[vp_col] - mean_mapped) / (std_mapped + 1e-5)

for sp_col in ["Sp", "Locality"]:
    if sp_col in combined_df.columns:
        sp_ht_map = train_part.groupby(sp_col)["Ht_imputed"].median()
        sp_dbh_map = train_part.groupby(sp_col)["DBH_imputed"].median()
        sp_ht_med = combined_df[sp_col].map(sp_ht_map).fillna(train_part["Ht_imputed"].median())
        sp_dbh_med = combined_df[sp_col].map(sp_dbh_map).fillna(train_part["DBH_imputed"].median())
        combined_df[f"ht_rel_{sp_col}"] = ht / (sp_ht_med + 1e-3)
        combined_df[f"dbh_rel_{sp_col}"] = dbh / (sp_dbh_med + 1e-3)

for c in cat_cols:
    if c in combined_df.columns:
        freq = train_part[c].value_counts()
        combined_df[f"{c}_freq"] = combined_df[c].map(freq).fillna(0)
        combined_df[c] = LabelEncoder().fit_transform(combined_df[c])

X_all = combined_df.iloc[:train_len].copy().reset_index(drop=True)
X_test = combined_df.iloc[train_len:].copy().reset_index(drop=True)

feature_cols = [c for c in X_all.columns if c not in ["Utility_imputed"]]
X_train_df = X_all[feature_cols].copy()
X_test_df = X_test[feature_cols].copy()

cat_cols_in_features = [c for c in cat_cols if c in feature_cols]
te_cols = [c for c in ["Abbrev", "Locality", "Map_Ref", "Latitude", "Sp"] if c in feature_cols]


# -----------------------------------------------------------------------------
# 3. PyTorch Deep Tabular ResNet Definition
# -----------------------------------------------------------------------------
class SwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.norm1(x)
        gate_val = self.fc1(x_norm)
        gate, val = gate_val.chunk(2, dim=-1)
        h = self.act(gate) * val
        h = self.dropout(h)
        out = self.fc2(h)
        out = self.dropout(out)
        return res + out


class TabularResNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(num_features)
        self.input_proj = nn.Sequential(nn.Linear(num_features, d_model), nn.SiLU())
        self.blocks = nn.ModuleList(
            [SwiGLUBlock(d_model=d_model, dropout=dropout) for _ in range(n_blocks)]
        )
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) > 1:
            x = self.bn_in(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# 4. Stratified 5-Fold Training & Cross Validation
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_nn = np.zeros((train_len, num_classes))
oof_lgb = np.zeros((train_len, num_classes))
oof_xgb = np.zeros((train_len, num_classes))
oof_hgb = np.zeros((train_len, num_classes))
oof_cat = np.zeros((train_len, num_classes))
oof_rf = np.zeros((train_len, num_classes))
oof_et = np.zeros((train_len, num_classes))

test_nn = np.zeros((len(X_test_df), num_classes))
test_lgb = np.zeros((len(X_test_df), num_classes))
test_xgb = np.zeros((len(X_test_df), num_classes))
test_hgb = np.zeros((len(X_test_df), num_classes))
test_cat = np.zeros((len(X_test_df), num_classes))
test_rf = np.zeros((len(X_test_df), num_classes))
test_et = np.zeros((len(X_test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_encoded)):
    y_tr, y_va = y_encoded[train_idx], y_encoded[val_idx]
    X_tr_df = X_train_df.iloc[train_idx].copy()
    X_va_df = X_train_df.iloc[val_idx].copy()
    X_te_df = X_test_df.copy()

    # Out-Of-Fold Multi-Class Target Encoding
    smoothing = 10.0
    for col in te_cols:
        if col in X_tr_df.columns:
            for k in range(num_classes):
                y_k = (y_tr == k).astype(float)
                global_prior = y_k.mean()
                cat_stats = (
                    pd.DataFrame({col: X_tr_df[col], "target": y_k})
                    .groupby(col)["target"]
                    .agg(["count", "mean"])
                )
                smooth_val = (
                    cat_stats["count"] * cat_stats["mean"] + smoothing * global_prior
                ) / (cat_stats["count"] + smoothing)
                col_name = f"{col}_te_{k}"
                X_tr_df[col_name] = X_tr_df[col].map(smooth_val).fillna(global_prior).astype(float)
                X_va_df[col_name] = X_va_df[col].map(smooth_val).fillna(global_prior).astype(float)
                X_te_df[col_name] = X_te_df[col].map(smooth_val).fillna(global_prior).astype(float)

    X_tr = X_tr_df.values
    X_va = X_va_df.values
    X_te = X_te_df.values

    # Imputer & Scaler for Neural Network
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_va_imp = imputer.transform(X_va)
    X_te_imp = imputer.transform(X_te)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_va_s = scaler.transform(X_va_imp)
    X_te_s = scaler.transform(X_te_imp)

    # PyTorch Data Loaders
    tr_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)
    )

    tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_dataset, batch_size=64, shuffle=False)

    model = TabularResNet(num_features=X_tr.shape[1], num_classes=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-5)

    best_val_loss = float("inf")
    best_nn_val_probs = None

    for epoch in range(70):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_probs_list = []
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits, by).item() * len(by)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_probs_list.append(probs)

        val_loss /= len(va_dataset)
        val_probs = np.vstack(val_probs_list)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_val_probs = val_probs
            torch.save(model.state_dict(), f"./working/resnet_fold_{fold}.pt")

    oof_nn[val_idx] = best_nn_val_probs

    # NN Test Inference
    model.load_state_dict(torch.load(f"./working/resnet_fold_{fold}.pt"))
    model.eval()
    with torch.no_grad():
        te_tensor = torch.tensor(X_te_s, dtype=torch.float32).to(device)
        te_logits = model(te_tensor)
        te_probs = torch.softmax(te_logits, dim=-1).cpu().numpy()
        test_nn += te_probs / n_splits

    # Model 2: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=24,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
    test_lgb += lgb_model.predict_proba(X_te) / n_splits

    # Model 3: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
    test_xgb += xgb_model.predict_proba(X_te) / n_splits

    # Model 4: HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.03, random_state=42 + fold
    )
    hgb_model.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_model.predict_proba(X_va)
    test_hgb += hgb_model.predict_proba(X_te) / n_splits

    # Model 5: CatBoost
    cat_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        early_stopping_rounds=30,
        random_seed=42 + fold,
        verbose=0,
    )
    cat_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_cat[val_idx] = cat_model.predict_proba(X_va)
    test_cat += cat_model.predict_proba(X_te) / n_splits

    # Model 6: RandomForest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42 + fold, n_jobs=-1
    )
    rf_model.fit(X_tr_imp, y_tr)
    oof_rf[val_idx] = rf_model.predict_proba(X_va_imp)
    test_rf += rf_model.predict_proba(X_te_imp) / n_splits

    # Model 7: ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=300, max_depth=12, random_state=42 + fold, n_jobs=-1
    )
    et_model.fit(X_tr_imp, y_tr)
    oof_et[val_idx] = et_model.predict_proba(X_va_imp)
    test_et += et_model.predict_proba(X_te_imp) / n_splits


# -----------------------------------------------------------------------------
# 5. Ensemble Optimization & Final Predictions
# -----------------------------------------------------------------------------
all_oof = [oof_nn, oof_lgb, oof_xgb, oof_hgb, oof_cat, oof_rf, oof_et]
all_test = [test_nn, test_lgb, test_xgb, test_hgb, test_cat, test_rf, test_et]
n_models = len(all_oof)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_blend_probs(params, probs_list):
    weights = softmax(params[:n_models])
    temp = np.clip(params[n_models], 0.1, 5.0)

    blend = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list):
        scaled_p = np.exp(np.log(np.clip(p, 1e-12, 1.0)) / temp)
        scaled_p /= np.sum(scaled_p, axis=1, keepdims=True)
        blend += weights[i] * scaled_p
    return blend


def ensemble_loss(params):
    blend_probs = get_blend_probs(params, all_oof)
    preds = np.argmax(blend_probs, axis=1)
    return -accuracy_score(y_encoded, preds)


best_loss = float("inf")
best_params = None

np.random.seed(42)
starts = [
    np.append(np.ones(n_models), 1.0),
    np.append(np.zeros(n_models), 1.0),
]
for _ in range(15):
    rand_w = np.random.randn(n_models)
    rand_t = np.random.uniform(0.5, 2.0)
    starts.append(np.append(rand_w, rand_t))

for init_p in starts:
    res = opt.minimize(
        ensemble_loss,
        init_p,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    if res.fun < best_loss:
        best_loss = res.fun
        best_params = res.x

oof_final_probs = get_blend_probs(best_params, all_oof)
oof_final_preds = np.argmax(oof_final_probs, axis=1)
score = accuracy_score(y_encoded, oof_final_preds)

test_final_probs = get_blend_probs(best_params, all_test)
test_final_preds = np.argmax(test_final_probs, axis=1)
test_final_labels = le.inverse_transform(test_final_preds)

# -----------------------------------------------------------------------------
# 6. Output Submission File & Validation Score
# -----------------------------------------------------------------------------
sub_df = pd.DataFrame({"id": test_ids, "Utility": test_final_labels})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")