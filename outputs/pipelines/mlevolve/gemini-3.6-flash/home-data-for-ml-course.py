import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


# 1. Global Reproducibility Setup
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Data Loading & Comprehensive Feature Engineering
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

if "GrLivArea" in train_df.columns and "SalePrice" in train_df.columns:
    train_df = train_df[~((train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000))].reset_index(drop=True)

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

target_col = "SalePrice"
y_train_raw = train_df[target_col].values
y_train_log = np.log1p(y_train_raw)

test_ids = test_df["id"].values

train_len = len(train_df)
combined_df = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Impute categoricals where NA represents structural absence
none_cols = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]
for col in none_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna("None")

# Impute numericals where NA represents 0 amount
zero_cols = [
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
    "MasVnrArea",
]
for col in zero_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna(0)

# Impute LotFrontage with Neighborhood median from train
if "LotFrontage" in combined_df.columns and "Neighborhood" in combined_df.columns:
    nb_frontage_med = train_df.groupby("Neighborhood")["LotFrontage"].median().to_dict()
    global_frontage_med = train_df["LotFrontage"].median()
    combined_df["LotFrontage"] = combined_df.apply(
        lambda row: (
            nb_frontage_med.get(row["Neighborhood"], global_frontage_med)
            if pd.isna(row["LotFrontage"])
            else row["LotFrontage"]
        ),
        axis=1,
    )

# Impute GarageYrBlt with YearBuilt
if "GarageYrBlt" in combined_df.columns:
    combined_df["GarageYrBlt"] = combined_df["GarageYrBlt"].fillna(
        combined_df["YearBuilt"]
    )

# Remaining categoricals: mode imputation from train
cat_cols_remaining = combined_df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols_remaining:
    mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else "Missing"
    combined_df[col] = combined_df[col].fillna(mode_val)

# Remaining numericals: median imputation from train
num_cols_remaining = combined_df.select_dtypes(include=[np.number]).columns
for col in num_cols_remaining:
    if col != "id":
        med_val = train_df[col].median()
        combined_df[col] = combined_df[col].fillna(med_val)

# Ordinal Encoding for Quality/Condition Features
qual_dict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
qual_cols = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]
for col in qual_cols:
    if col in combined_df.columns:
        combined_df[col + "_Num"] = (
            combined_df[col].map(qual_dict).fillna(0).astype(int)
        )

exposure_dict = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
if "BsmtExposure" in combined_df.columns:
    combined_df["BsmtExposure_Num"] = (
        combined_df["BsmtExposure"].map(exposure_dict).fillna(0).astype(int)
    )

bsmt_fin_dict = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    if col in combined_df.columns:
        combined_df[col + "_Num"] = (
            combined_df[col].map(bsmt_fin_dict).fillna(0).astype(int)
        )

garage_fin_dict = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
if "GarageFinish" in combined_df.columns:
    combined_df["GarageFinish_Num"] = (
        combined_df["GarageFinish"].map(garage_fin_dict).fillna(0).astype(int)
    )

functional_dict = {
    "Typ": 7,
    "Min1": 6,
    "Min2": 5,
    "Mod": 4,
    "Maj1": 3,
    "Maj2": 2,
    "Sev": 1,
    "Sal": 0,
}
if "Functional" in combined_df.columns:
    combined_df["Functional_Num"] = (
        combined_df["Functional"].map(functional_dict).fillna(7).astype(int)
    )

# Domain Feature Constructions
combined_df["TotalSF"] = (
    combined_df["TotalBsmtSF"] + combined_df["1stFlrSF"] + combined_df["2ndFlrSF"]
)
combined_df["TotalBath"] = (
    combined_df["FullBath"]
    + 0.5 * combined_df["HalfBath"]
    + combined_df["BsmtFullBath"]
    + 0.5 * combined_df["BsmtHalfBath"]
)
combined_df["TotalPorchSF"] = (
    combined_df["OpenPorchSF"]
    + combined_df["EnclosedPorch"]
    + combined_df["3SsnPorch"]
    + combined_df["ScreenPorch"]
    + combined_df["WoodDeckSF"]
)

combined_df["HouseAge"] = (combined_df["YrSold"] - combined_df["YearBuilt"]).clip(
    lower=0
)
combined_df["RemodAge"] = (combined_df["YrSold"] - combined_df["YearRemodAdd"]).clip(
    lower=0
)
combined_df["GarageAge"] = (combined_df["YrSold"] - combined_df["GarageYrBlt"]).clip(
    lower=0
)
combined_df["IsRemodeled"] = (
    combined_df["YearRemodAdd"] != combined_df["YearBuilt"]
).astype(int)

combined_df["OverallQual_GrLivArea"] = (
    combined_df["OverallQual"] * combined_df["GrLivArea"]
)
combined_df["OverallQual_TotalSF"] = combined_df["OverallQual"] * combined_df["TotalSF"]
combined_df["OverallQual_Cond"] = (
    combined_df["OverallQual"] * combined_df["OverallCond"]
)

combined_df["HasPool"] = (combined_df["PoolArea"] > 0).astype(int)
combined_df["Has2ndFloor"] = (combined_df["2ndFlrSF"] > 0).astype(int)
combined_df["HasGarage"] = (combined_df["GarageArea"] > 0).astype(int)
combined_df["HasBsmt"] = (combined_df["TotalBsmtSF"] > 0).astype(int)
combined_df["HasFireplace"] = (combined_df["Fireplaces"] > 0).astype(int)
combined_df["HasPorch"] = (combined_df["TotalPorchSF"] > 0).astype(int)

# Composite Quality Metrics and Spatial Ratios
combined_df["TotalQual"] = (
    combined_df["OverallQual"]
    + combined_df["BsmtQual_Num"]
    + combined_df["KitchenQual_Num"]
    + combined_df["ExterQual_Num"]
    + combined_df["HeatingQC_Num"]
)
combined_df["GrLivArea_TotalSF"] = combined_df["GrLivArea"] / (
    combined_df["TotalSF"] + 1e-5
)
combined_df["LotArea_TotalSF"] = combined_df["LotArea"] / (
    combined_df["TotalSF"] + 1e-5
)
combined_df["TotalBath_TotalSF"] = combined_df["TotalBath"] / (
    combined_df["TotalSF"] + 1e-5
)

# Neighborhood-Relative Standard Features & Spatial Ratios
if "Neighborhood" in combined_df.columns:
    nb_totalsf_med = combined_df.groupby("Neighborhood")["TotalSF"].transform("median")
    combined_df["TotalSF_Nb_Ratio"] = combined_df["TotalSF"] / (nb_totalsf_med + 1e-5)

    nb_qual_mean = combined_df.groupby("Neighborhood")["OverallQual"].transform("mean")
    combined_df["OverallQual_Nb_Diff"] = combined_df["OverallQual"] - nb_qual_mean

if "TotRmsAbvGrd" in combined_df.columns:
    combined_df["GrLivArea_TotRms_Ratio"] = combined_df["GrLivArea"] / (
        combined_df["TotRmsAbvGrd"] + 1e-5
    )

# Frequency Encoding for Categorical Features
cat_cols_for_freq = combined_df.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols_for_freq:
    freq_map = combined_df[c].value_counts(normalize=True).to_dict()
    combined_df[f"{c}_Freq"] = combined_df[c].map(freq_map).fillna(0)

# Apply log1p transformation to highly skewed continuous numerical features (skewness > 0.75)
num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

skew_series = combined_df.iloc[:train_len][num_cols].skew()
skewed_cols = skew_series[skew_series > 0.75].index.tolist()
skewed_cols = [c for c in skewed_cols if combined_df[c].nunique() > 2]

for c in skewed_cols:
    combined_df[c] = np.log1p(np.maximum(0, combined_df[c]))

combined_df = combined_df.copy()

X_train_full = combined_df.iloc[:train_len].copy().reset_index(drop=True)
X_test_full = combined_df.iloc[train_len:].copy().reset_index(drop=True)

if "id" in X_train_full.columns:
    X_train_full = X_train_full.drop(columns=["id"])
if "id" in X_test_full.columns:
    X_test_full = X_test_full.drop(columns=["id"])

# Out-Of-Fold Target Encoding for Neighborhood and MSSubClass
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
target_enc_cols = ["Neighborhood", "MSSubClass"]

for col in target_enc_cols:
    if col in X_train_full.columns:
        X_train_full[f"{col}_TargetEnc"] = np.nan
        X_test_full[f"{col}_TargetEnc"] = np.nan

        global_mean = y_train_log.mean()

        for tr_i, va_i in kf_te.split(X_train_full):
            fold_tr_col = X_train_full.iloc[tr_i][col]
            fold_tr_y = pd.Series(y_train_log[tr_i], index=fold_tr_col.index)
            enc_map = fold_tr_y.groupby(fold_tr_col).mean()
            X_train_full.loc[va_i, f"{col}_TargetEnc"] = (
                X_train_full.iloc[va_i][col].map(enc_map).fillna(global_mean)
            )

        full_enc_map = (
            pd.Series(y_train_log, index=X_train_full.index)
            .groupby(X_train_full[col])
            .mean()
        )
        X_test_full[f"{col}_TargetEnc"] = (
            X_test_full[col].map(full_enc_map).fillna(global_mean)
        )


# Prepare dense unscaled features for linear models and PyTorch TabularResNet
def build_dense_features(X_tr, X_te):
    comb = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    cat_cols = comb.select_dtypes(include=["object", "category"]).columns.tolist()
    comb_enc = pd.get_dummies(comb, columns=cat_cols, drop_first=True)
    comb_enc = comb_enc.fillna(0)

    tr_len = len(X_tr)
    X_tr_enc = comb_enc.iloc[:tr_len].values.astype(np.float32)
    X_te_enc = comb_enc.iloc[tr_len:].values.astype(np.float32)
    return X_tr_enc, X_te_enc


X_tr_dense, X_te_dense = build_dense_features(X_train_full, X_test_full)

# Prepare tree features with explicit categorical dtypes
X_train_tree = X_train_full.copy()
X_test_tree = X_test_full.copy()
cat_cols_tree = X_train_tree.select_dtypes(
    include=["object", "category"]
).columns.tolist()
for c in cat_cols_tree:
    X_train_tree[c] = X_train_tree[c].astype("category")
    X_test_tree[c] = X_test_tree[c].astype("category")


# 3. PyTorch Architecture Definition
class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pytorch_resnet(
    X_tr, y_tr, X_va, y_va, X_te, epochs=120, batch_size=64, lr=1e-3
):
    input_dim = X_tr.shape[1]
    model = TabularResNet(input_dim=input_dim).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(X_va, dtype=torch.float32).to(device)
            y_v = torch.tensor(y_va, dtype=torch.float32).to(device).unsqueeze(1)
            val_loss = criterion(model(x_v), y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    with torch.no_grad():
        oof_pred = (
            model(torch.tensor(X_va, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .ravel()
        )
        test_pred = (
            model(torch.tensor(X_te, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .ravel()
        )

    return oof_pred, test_pred


# 4. Multi-Seed 10-Fold Training Loop
SEEDS = [42, 100, 2024]
N_SPLITS = 10
model_names = [
    "LightGBM",
    "XGBoost",
    "CatBoost_ET",
    "Ridge",
    "ElasticNet",
    "TabularResNet",
    "SVR",
    "HistGBR",
]
num_models = len(model_names)

oof_matrix = np.zeros((train_len, num_models))
test_matrix = np.zeros((len(test_df), num_models))

for seed in SEEDS:
    seed_everything(seed)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_seed = np.zeros((train_len, num_models))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_tree)):
        X_tr_tree, y_tr = X_train_tree.iloc[tr_idx], y_train_log[tr_idx]
        X_va_tree, y_va = X_train_tree.iloc[va_idx], y_train_log[va_idx]

        scaler = StandardScaler()
        X_tr_d = scaler.fit_transform(X_tr_dense[tr_idx])
        X_va_d = scaler.transform(X_tr_dense[va_idx])
        X_te_d = scaler.transform(X_te_dense)

        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            verbose=-1,
            n_estimators=1500,
        )
        lgb_model.fit(
            X_tr_tree,
            y_tr,
            eval_set=[(X_va_tree, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_seed[va_idx, 0] = lgb_model.predict(X_va_tree)
        test_matrix[:, 0] += lgb_model.predict(X_test_tree) / (N_SPLITS * len(SEEDS))

        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            enable_categorical=True,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], verbose=False)
        oof_seed[va_idx, 1] = xgb_model.predict(X_va_tree)
        test_matrix[:, 1] += xgb_model.predict(X_test_tree) / (N_SPLITS * len(SEEDS))

        # 3. CatBoost or ExtraTrees
        if HAS_CATBOOST:
            cb_model = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.02,
                depth=5,
                l2_leaf_reg=3.0,
                random_seed=seed + fold,
                verbose=0,
                early_stopping_rounds=50,
            )
            cb_model.fit(
                X_tr_tree, y_tr, eval_set=(X_va_tree, y_va), cat_features=cat_cols_tree
            )
            oof_seed[va_idx, 2] = cb_model.predict(X_va_tree)
            test_matrix[:, 2] += cb_model.predict(X_test_tree) / (N_SPLITS * len(SEEDS))
        else:
            et_model = ExtraTreesRegressor(
                n_estimators=300, max_depth=12, random_state=seed + fold, n_jobs=-1
            )
            et_model.fit(X_tr_d, y_tr)
            oof_seed[va_idx, 2] = et_model.predict(X_va_d)
            test_matrix[:, 2] += et_model.predict(X_te_d) / (N_SPLITS * len(SEEDS))

        # 4. Ridge Regression
        ridge = Ridge(alpha=10.0, random_state=seed + fold)
        ridge.fit(X_tr_d, y_tr)
        oof_seed[va_idx, 3] = ridge.predict(X_va_d)
        test_matrix[:, 3] += ridge.predict(X_te_d) / (N_SPLITS * len(SEEDS))

        # 5. ElasticNet Regression
        enet = ElasticNet(
            alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=seed + fold
        )
        enet.fit(X_tr_d, y_tr)
        oof_seed[va_idx, 4] = enet.predict(X_va_d)
        test_matrix[:, 4] += enet.predict(X_te_d) / (N_SPLITS * len(SEEDS))

        # 6. PyTorch TabularResNet
        res_oof, res_test = train_pytorch_resnet(X_tr_d, y_tr, X_va_d, y_va, X_te_d)
        oof_seed[va_idx, 5] = res_oof
        test_matrix[:, 5] += res_test / (N_SPLITS * len(SEEDS))

        # 7. Support Vector Regressor (SVR with RBF kernel)
        svr_model = SVR(kernel="rbf", C=10.0, epsilon=0.01)
        svr_model.fit(X_tr_d, y_tr)
        oof_seed[va_idx, 6] = svr_model.predict(X_va_d)
        test_matrix[:, 6] += svr_model.predict(X_te_d) / (N_SPLITS * len(SEEDS))

        # 8. HistGradientBoostingRegressor
        hgb_model = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.03, random_state=seed + fold
        )
        hgb_model.fit(X_tr_d, y_tr)
        oof_seed[va_idx, 7] = hgb_model.predict(X_va_d)
        test_matrix[:, 7] += hgb_model.predict(X_te_d) / (N_SPLITS * len(SEEDS))

    oof_matrix += oof_seed / len(SEEDS)


# 5. Cross-Validated Ridge Meta-Learner (Level-2 Stacking)
meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10)
meta_learner.fit(oof_matrix, y_train_log)

oof_blended = meta_learner.predict(oof_matrix)
test_blended = meta_learner.predict(test_matrix)

final_cv_rmse = root_mean_squared_error(y_train_log, oof_blended)

# 6. Generate Official Output Submission File
final_sale_price_preds = np.expm1(test_blended)
submission_df = pd.DataFrame({"id": test_ids, "SalePrice": final_sale_price_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_cv_rmse:.5f}")