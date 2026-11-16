import os
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from catboost import CatBoostRegressor

# =============================================================================
# 0. Setup Directories and Seeds
# =============================================================================
output_dir = "./working"
models_dir = os.path.join(output_dir, "models")
submission_dir = "./submission"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 1. Load Data & Outlier Removal
# =============================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Known domain outliers: GrLivArea > 4000 with SalePrice < 300000
if "SalePrice" in train_df.columns and "GrLivArea" in train_df.columns:
    outlier_mask = (train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)
    train_df = train_df[~outlier_mask].reset_index(drop=True)

y_train_raw = train_df["SalePrice"].values
y_train_log = np.log1p(y_train_raw)

X_train_raw = train_df.drop(columns=["SalePrice"])
X_test_raw = test_df.copy()

n_train = len(X_train_raw)

# Combine datasets for uniform feature processing
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# =============================================================================
# 2. Imputation & Ordinal Encoding
# =============================================================================
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
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna("None")

train_lot_frontage_medians = (
    df_all.iloc[:n_train].groupby("Neighborhood")["LotFrontage"].median()
)
global_lot_frontage_median = df_all.iloc[:n_train]["LotFrontage"].median()


def fill_lot_frontage(row):
    if pd.isna(row["LotFrontage"]):
        return train_lot_frontage_medians.get(
            row["Neighborhood"], global_lot_frontage_median
        )
    return row["LotFrontage"]


df_all["LotFrontage"] = df_all.apply(fill_lot_frontage, axis=1)

zero_cols = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]
for col in zero_cols:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna(0)

cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    train_mode = df_all.iloc[:n_train][col].mode()
    mode_val = train_mode[0] if len(train_mode) > 0 else "None"
    df_all[col] = df_all[col].fillna(mode_val)

qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
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
    if col in df_all.columns:
        df_all[col] = df_all[col].map(qual_map).fillna(0).astype(int)

exposure_map = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
if "BsmtExposure" in df_all.columns:
    df_all["BsmtExposure"] = (
        df_all["BsmtExposure"].map(exposure_map).fillna(0).astype(int)
    )

bsmt_fin_map = {
    "None": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}
for col in ["BsmtFinType1", "BsmtFinType2"]:
    if col in df_all.columns:
        df_all[col] = df_all[col].map(bsmt_fin_map).fillna(0).astype(int)

garage_finish_map = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
if "GarageFinish" in df_all.columns:
    df_all["GarageFinish"] = (
        df_all["GarageFinish"].map(garage_finish_map).fillna(0).astype(int)
    )

functional_map = {
    "Sal": 0,
    "Sev": 1,
    "Maj2": 2,
    "Maj1": 3,
    "Mod": 4,
    "Min2": 5,
    "Min1": 6,
    "Typ": 7,
}
if "Functional" in df_all.columns:
    df_all["Functional"] = (
        df_all["Functional"].map(functional_map).fillna(7).astype(int)
    )

# =============================================================================
# 3. Domain Feature Engineering
# =============================================================================
df_all["TotalSF"] = df_all["TotalBsmtSF"] + df_all["1stFlrSF"] + df_all["2ndFlrSF"]
df_all["TotalPorchSF"] = (
    df_all["OpenPorchSF"]
    + df_all["EnclosedPorch"]
    + df_all["3SsnPorch"]
    + df_all["ScreenPorch"]
    + df_all["WoodDeckSF"]
)
df_all["TotalBathrooms"] = (
    df_all["FullBath"]
    + (0.5 * df_all["HalfBath"])
    + df_all["BsmtFullBath"]
    + (0.5 * df_all["BsmtHalfBath"])
)

df_all["HouseAge"] = df_all["YrSold"] - df_all["YearBuilt"]
df_all["RemodAge"] = df_all["YrSold"] - df_all["YearRemodAdd"]
df_all["IsRemodeled"] = (df_all["YearRemodAdd"] != df_all["YearBuilt"]).astype(int)

garage_yr = np.where(
    df_all["GarageYrBlt"] == 0, df_all["YearBuilt"], df_all["GarageYrBlt"]
)
df_all["GarageAge"] = df_all["YrSold"] - garage_yr

df_all["OverallQual_TotalSF"] = df_all["OverallQual"] * df_all["TotalSF"]
df_all["OverallQual_GrLivArea"] = df_all["OverallQual"] * df_all["GrLivArea"]
df_all["ExterQual_KitchenQual"] = df_all["ExterQual"] * df_all["KitchenQual"]

# Domain ratio features
df_all["GrLivArea_Per_Room"] = np.where(
    df_all["TotRmsAbvGrd"] > 0, df_all["GrLivArea"] / df_all["TotRmsAbvGrd"], 0.0
)
df_all["TotalSF_To_LotArea"] = np.where(
    df_all["LotArea"] > 0, df_all["TotalSF"] / df_all["LotArea"], 0.0
)
df_all["1stFlrSF_To_Bsmt"] = np.where(
    df_all["TotalBsmtSF"] > 0, df_all["1stFlrSF"] / df_all["TotalBsmtSF"], 0.0
)
df_all["OverallQual_OverallCond"] = df_all["OverallQual"] * df_all["OverallCond"]

# Neighborhood group summary statistics computed strictly from training split
neigh_grliv_median = (
    df_all.iloc[:n_train].groupby("Neighborhood")["GrLivArea"].median()
)
neigh_lot_median = df_all.iloc[:n_train].groupby("Neighborhood")["LotArea"].median()
df_all["Neigh_Med_GrLivArea"] = (
    df_all["Neighborhood"]
    .map(neigh_grliv_median)
    .fillna(df_all.iloc[:n_train]["GrLivArea"].median())
)
df_all["Neigh_Med_LotArea"] = (
    df_all["Neighborhood"]
    .map(neigh_lot_median)
    .fillna(df_all.iloc[:n_train]["LotArea"].median())
)

df_all["HasPool"] = (df_all["PoolArea"] > 0).astype(int)
df_all["Has2ndFloor"] = (df_all["2ndFlrSF"] > 0).astype(int)
df_all["HasGarage"] = (df_all["GarageArea"] > 0).astype(int)
df_all["HasBsmt"] = (df_all["TotalBsmtSF"] > 0).astype(int)
df_all["HasFireplace"] = (df_all["Fireplaces"] > 0).astype(int)

if "Id" in df_all.columns:
    df_all = df_all.drop(columns=["Id"])

X_train_df = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_df = df_all.iloc[n_train:].copy().reset_index(drop=True)

# =============================================================================
# 4. One-Hot Encoding & Preprocessing
# =============================================================================
target_enc_cols = [
    "Neighborhood",
    "MSSubClass",
    "Exterior1st",
    "Exterior2nd",
    "GarageType",
    "SaleCondition",
]

remaining_cat_cols = X_train_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
df_combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
df_combined_encoded = pd.get_dummies(
    df_combined, columns=remaining_cat_cols, drop_first=True
)

X_train_encoded = df_combined_encoded.iloc[:n_train].copy().reset_index(drop=True)
X_test_encoded = df_combined_encoded.iloc[n_train:].copy().reset_index(drop=True)

num_cols = X_train_encoded.select_dtypes(include=[np.number]).columns.tolist()
skewness = (
    X_train_encoded[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
)
skewed_cols = skewness[skewness > 0.75].index.tolist()

for col in skewed_cols:
    X_train_encoded[col] = np.log1p(np.maximum(0, X_train_encoded[col]))
    X_test_encoded[col] = np.log1p(np.maximum(0, X_test_encoded[col]))

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_encoded), columns=X_test_encoded.columns
)

kf = KFold(n_splits=10, shuffle=True, random_state=seed)
cv_folds = list(kf.split(X_train_scaled))

# Save processed objects for reproducibility
X_train_scaled.to_parquet(os.path.join(output_dir, "X_train_full.parquet"))
X_test_scaled.to_parquet(os.path.join(output_dir, "X_test.parquet"))
pd.DataFrame({"target": y_train_log}).to_parquet(
    os.path.join(output_dir, "y_train_full.parquet")
)


# =============================================================================
# 5. Model Architecture & Factory Definitions
# =============================================================================
class HousePriceMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super(HousePriceMLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_proj = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.out_layer = nn.Linear(hidden_dims[2], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        res = self.res_proj(x)
        x = self.block1(x) + res
        x = self.block2(x)
        out = self.out_layer(x)
        return out.squeeze(-1)


def get_pytorch_model_and_optimizer(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-2
):
    model = HousePriceMLP(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.25
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def get_lgbm_model(random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_xgb_model(random_state: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
    )


def get_ridge_model(alpha: float = 15.0, random_state: int = 42) -> Ridge:
    return Ridge(alpha=alpha, random_state=random_state)


def get_lasso_model(alpha: float = 0.0005, random_state: int = 42) -> Lasso:
    return Lasso(alpha=alpha, max_iter=10000, random_state=random_state)


def get_elasticnet_model(
    alpha: float = 0.0005, l1_ratio: float = 0.5, random_state: int = 42
) -> ElasticNet:
    return ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state
    )


# =============================================================================
# 6. PyTorch Fold Training Function
# =============================================================================
def train_pytorch_fold(
    X_tr, y_tr, X_va, y_va, X_te, input_dim, fold_idx, epochs=250, batch_size=32
):
    tr_ds = TensorDataset(
        torch.tensor(X_tr.values, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va.values, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    te_tensor = torch.tensor(X_te.values, dtype=torch.float32).to(device)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model, criterion, optimizer = get_pytorch_model_and_optimizer(
        input_dim=input_dim, lr=1e-3, weight_decay=1e-2
    )
    model.to(device)

    best_loss = float("inf")
    best_weights = None
    patience = 35
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(va_ds)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), os.path.join(models_dir, f"mlp_fold_{fold_idx}.pt"))

    model.eval()
    with torch.no_grad():
        va_tensor = torch.tensor(X_va.values, dtype=torch.float32).to(device)
        oof_pred = model(va_tensor).cpu().numpy()
        test_pred = model(te_tensor).cpu().numpy()

    return oof_pred, test_pred


# =============================================================================
# 7. Cross-Validation Loop & Out-Of-Fold Predictions
# =============================================================================
n_samples = len(X_train_scaled)
n_test = len(X_test_scaled)
n_folds = len(cv_folds)
input_dim = X_train_scaled.shape[1]

model_names = ["lgb", "xgb", "cat", "ridge", "lasso", "elasticnet", "mlp"]
n_models = len(model_names)

oof_preds = {m: np.zeros(n_samples) for m in model_names}
test_preds = {m: np.zeros(n_test) for m in model_names}

for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
    X_tr, y_tr = X_train_scaled.iloc[tr_idx].copy(), y_train_log[tr_idx]
    X_va, y_va = X_train_scaled.iloc[va_idx].copy(), y_train_log[va_idx]
    X_te_fold = X_test_scaled.copy()

    # Re-compute target encoding inside CV loop to avoid leakage
    for col in target_enc_cols:
        te_col = f"{col}_TE"
        if col in X_train_df.columns and te_col in X_tr.columns:
            tr_cats = X_train_df[col].iloc[tr_idx].astype(str)
            va_cats = X_train_df[col].iloc[va_idx].astype(str)
            te_cats = X_test_df[col].astype(str)

            tr_mean = y_tr.mean()
            df_tr_cat = pd.DataFrame({"cat": tr_cats.values, "y": y_tr})
            means = df_tr_cat.groupby("cat")["y"].mean()

            X_tr[te_col] = tr_cats.map(means).fillna(tr_mean).values
            X_va[te_col] = va_cats.map(means).fillna(tr_mean).values
            X_te_fold[te_col] = te_cats.map(means).fillna(tr_mean).values

    # Model 1: LightGBM
    lgb_model = get_lgbm_model(random_state=seed + fold_idx)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_preds["lgb"][va_idx] = lgb_model.predict(X_va)
    test_preds["lgb"] += lgb_model.predict(X_te_fold) / n_folds

    # Model 2: XGBoost
    xgb_model = get_xgb_model(random_state=seed + fold_idx)
    xgb_model.set_params(early_stopping_rounds=50)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds["xgb"][va_idx] = xgb_model.predict(X_va)
    test_preds["xgb"] += xgb_model.predict(X_te_fold) / n_folds

    # Model 3: CatBoost
    cat_model = get_catboost_model(random_state=seed + fold_idx)
    cat_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0
    )
    oof_preds["cat"][va_idx] = cat_model.predict(X_va)
    test_preds["cat"] += cat_model.predict(X_te_fold) / n_folds

    # Model 4: Ridge Regression
    ridge_model = get_ridge_model(alpha=15.0, random_state=seed + fold_idx)
    ridge_model.fit(X_tr, y_tr)
    oof_preds["ridge"][va_idx] = ridge_model.predict(X_va)
    test_preds["ridge"] += ridge_model.predict(X_te_fold) / n_folds

    # Model 5: Lasso Regression
    lasso_model = get_lasso_model(alpha=0.0005, random_state=seed + fold_idx)
    lasso_model.fit(X_tr, y_tr)
    oof_preds["lasso"][va_idx] = lasso_model.predict(X_va)
    test_preds["lasso"] += lasso_model.predict(X_te_fold) / n_folds

    # Model 6: ElasticNet Regression
    elastic_model = get_elasticnet_model(
        alpha=0.0005, l1_ratio=0.5, random_state=seed + fold_idx
    )
    elastic_model.fit(X_tr, y_tr)
    oof_preds["elasticnet"][va_idx] = elastic_model.predict(X_va)
    test_preds["elasticnet"] += elastic_model.predict(X_te_fold) / n_folds

    # Model 7: PyTorch Deep ResMLP
    mlp_oof, mlp_test = train_pytorch_fold(
        X_tr, y_tr, X_va, y_va, X_te_fold, input_dim, fold_idx
    )
    oof_preds["mlp"][va_idx] = mlp_oof
    test_preds["mlp"] += mlp_test / n_folds

# =============================================================================
# 8. Ensemble Optimization via SLSQP
# =============================================================================
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])


def blend_objective(weights):
    weights = weights / np.sum(weights)
    pred = oof_matrix @ weights
    return np.sqrt(np.mean((pred - y_train_log) ** 2))


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    options={"maxiter": 1000},
)

best_weights = opt_res.x / np.sum(opt_res.x)

# =============================================================================
# 9. Compute Validation Score & Generate Submission File
# =============================================================================
oof_final_log = oof_matrix @ best_weights
final_rmsle = np.sqrt(np.mean((oof_final_log - y_train_log) ** 2))

test_final_log = test_matrix @ best_weights
test_final_price = np.expm1(np.maximum(0, test_final_log))

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_final_price)), "SalePrice": test_final_price}
)

submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_rmsle}")