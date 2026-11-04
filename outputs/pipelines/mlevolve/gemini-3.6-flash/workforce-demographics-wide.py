import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.stats import skew
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Environment setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Parquet Datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Identify target column
target_col = "Female (%)"
if target_col not in train_df.columns:
    possible_targets = [
        c for c in train_df.columns if "female" in c.lower() or "target" in c.lower()
    ]
    target_col = possible_targets[0] if possible_targets else train_df.columns[-1]

y = train_df[target_col].values.astype(np.float32)

# ID Column Handling
if "id" in test_df.columns:
    test_ids = test_df["id"].values
elif "Id" in test_df.columns:
    test_ids = test_df["Id"].values
else:
    test_ids = np.arange(len(test_df))

drop_cols = [c for c in [target_col, "id", "Id", "ID"] if c in train_df.columns]
X_train_raw = train_df.drop(columns=drop_cols)
drop_cols_test = [c for c in [target_col, "id", "Id", "ID"] if c in test_df.columns]
X_test_raw = test_df.drop(columns=drop_cols_test)

common_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
X_train_raw = X_train_raw[common_cols].copy()
X_test_raw = X_test_raw[common_cols].copy()


# 2. Extract Row Statistics and Categorical Features
def extract_row_stats(df, num_cols):
    if len(num_cols) == 0:
        return pd.DataFrame(index=df.index)
    num_arr = df[num_cols].to_numpy(dtype=np.float32)
    stats = pd.DataFrame(index=df.index)
    stats["row_mean"] = np.nanmean(num_arr, axis=1)
    stats["row_std"] = np.nanstd(num_arr, axis=1)
    stats["row_median"] = np.nanmedian(num_arr, axis=1)
    q75 = np.nanpercentile(num_arr, 75, axis=1)
    q25 = np.nanpercentile(num_arr, 25, axis=1)
    stats["row_iqr"] = q75 - q25
    stats["row_min"] = np.nanmin(num_arr, axis=1)
    stats["row_max"] = np.nanmax(num_arr, axis=1)
    stats["row_sum"] = np.nansum(num_arr, axis=1)
    stats["row_nans"] = np.isnan(num_arr).sum(axis=1)
    stats["row_zeros"] = (num_arr == 0).sum(axis=1)
    stats["row_nonzero_ratio"] = (num_arr != 0).sum(axis=1) / float(num_arr.shape[1])

    sk = skew(num_arr, axis=1, nan_policy="omit")
    stats["row_skew"] = np.nan_to_num(sk, nan=0.0)

    pct_ranks = df[num_cols].rank(pct=True, axis=1)
    pct_ranks.columns = [f"{col}_pctrank" for col in num_cols]

    return pd.concat([stats, pct_ranks], axis=1)


num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train_raw.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

train_row_stats = extract_row_stats(X_train_raw, num_cols)
test_row_stats = extract_row_stats(X_test_raw, num_cols)

for col in cat_cols:
    freq = X_train_raw[col].value_counts(normalize=True).to_dict()
    X_train_raw[f"{col}_freq"] = X_train_raw[col].map(freq).fillna(0)
    X_test_raw[f"{col}_freq"] = X_test_raw[col].map(freq).fillna(0)

# Extract PCA & SVD embeddings and ratio interactions
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="median")
    X_tr_num_imp = num_imputer.fit_transform(X_train_raw[num_cols])
    X_te_num_imp = num_imputer.transform(X_test_raw[num_cols])

    scaler_dim = StandardScaler()
    X_tr_num_scaled = scaler_dim.fit_transform(X_tr_num_imp)
    X_te_num_scaled = scaler_dim.transform(X_te_num_imp)

    n_comp = min(5, len(num_cols))
    pca = PCA(n_components=n_comp, random_state=42)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)

    pca_tr = pca.fit_transform(X_tr_num_scaled)
    pca_te = pca.transform(X_te_num_scaled)
    svd_tr = svd.fit_transform(X_tr_num_scaled)
    svd_te = svd.transform(X_te_num_scaled)

    pca_cols = [f"pca_{i}" for i in range(n_comp)]
    svd_cols = [f"svd_{i}" for i in range(n_comp)]

    pca_svd_train = pd.DataFrame(np.hstack([pca_tr, svd_tr]), columns=pca_cols + svd_cols, index=X_train_raw.index)
    pca_svd_test = pd.DataFrame(np.hstack([pca_te, svd_te]), columns=pca_cols + svd_cols, index=X_test_raw.index)

    # Non-linear ratio interactions among top numeric columns
    stds = np.std(X_tr_num_imp, axis=0)
    top_idx = np.argsort(stds)[::-1][:min(5, len(num_cols))]
    top_num_cols = [num_cols[i] for i in top_idx]

    ratio_train_dict = {}
    ratio_test_dict = {}
    for i in range(len(top_num_cols)):
        for j in range(i + 1, len(top_num_cols)):
            c1, c2 = top_num_cols[i], top_num_cols[j]
            c1_tr = X_tr_num_imp[:, num_cols.index(c1)]
            c2_tr = X_tr_num_imp[:, num_cols.index(c2)]
            c1_te = X_te_num_imp[:, num_cols.index(c1)]
            c2_te = X_te_num_imp[:, num_cols.index(c2)]

            ratio_train_dict[f"ratio_{c1}_{c2}"] = c1_tr / (np.abs(c2_tr) + 1e-5)
            ratio_train_dict[f"ratio_{c2}_{c1}"] = c2_tr / (np.abs(c1_tr) + 1e-5)
            ratio_test_dict[f"ratio_{c1}_{c2}"] = c1_te / (np.abs(c2_te) + 1e-5)
            ratio_test_dict[f"ratio_{c2}_{c1}"] = c2_te / (np.abs(c1_te) + 1e-5)

    ratios_train = pd.DataFrame(ratio_train_dict, index=X_train_raw.index)
    ratios_test = pd.DataFrame(ratio_test_dict, index=X_test_raw.index)

    X_train = pd.concat([X_train_raw, train_row_stats, pca_svd_train, ratios_train], axis=1)
    X_test = pd.concat([X_test_raw, test_row_stats, pca_svd_test, ratios_test], axis=1)
else:
    X_train = pd.concat([X_train_raw, train_row_stats], axis=1)
    X_test = pd.concat([X_test_raw, test_row_stats], axis=1)


# 3. Model Architecture: Gated Tabular ResNet
class GatedResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        g = self.gate(out)
        out = out * g
        return residual + out


class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=128, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# 4. Out-of-Fold Cross Validation Setup
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
oof_et = np.zeros(len(X_train))
oof_hist = np.zeros(len(X_train))
oof_nn = np.zeros(len(X_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))
test_et = np.zeros(len(X_test))
test_hist = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

torch.manual_seed(42)
np.random.seed(42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr = X_train.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    # Out-of-fold Target Encoding inside fold loop
    for col in cat_cols:
        target_map = (
            pd.DataFrame({"cat": X_tr[col], "target": y_tr})
            .groupby("cat")["target"]
            .mean()
            .to_dict()
        )
        global_mean = float(y_tr.mean())
        X_tr[f"{col}_te"] = X_tr[col].map(target_map).fillna(global_mean)
        X_va[f"{col}_te"] = X_va[col].map(target_map).fillna(global_mean)
        X_te[f"{col}_te"] = X_te[col].map(target_map).fillna(global_mean)

    for col in cat_cols:
        if col in X_tr.columns:
            X_tr.drop(columns=[col], inplace=True)
            X_va.drop(columns=[col], inplace=True)
            X_te.drop(columns=[col], inplace=True)

    feature_cols = X_tr.columns.tolist()

    # Fold preprocessing state isolation & fold-specific PCA/SVD
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_imp = imputer.fit_transform(X_tr[feature_cols])
    X_va_imp = imputer.transform(X_va[feature_cols])
    X_te_imp = imputer.transform(X_te[feature_cols])

    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_va_scaled = scaler.transform(X_va_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    if len(num_cols) > 0:
        n_comp = min(5, len(num_cols))
        pca = PCA(n_components=n_comp, random_state=42)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        pca_tr = pca.fit_transform(X_tr_scaled)
        pca_va = pca.transform(X_va_scaled)
        pca_te = pca.transform(X_te_scaled)
        svd_tr = svd.fit_transform(X_tr_scaled)
        svd_va = svd.transform(X_va_scaled)
        svd_te = svd.transform(X_te_scaled)
        X_tr_imp = np.hstack([X_tr_imp, pca_tr, svd_tr])
        X_va_imp = np.hstack([X_va_imp, pca_va, svd_va])
        X_te_imp = np.hstack([X_te_imp, pca_te, svd_te])
        X_tr_scaled = np.hstack([X_tr_scaled, pca_tr, svd_tr])
        X_va_scaled = np.hstack([X_va_scaled, pca_va, svd_va])
        X_te_scaled = np.hstack([X_te_scaled, pca_te, svd_te])

    # Base Model 1: LightGBM Regressor
    lgb_train = lgb.Dataset(X_tr_imp, label=y_tr)
    lgb_val = lgb.Dataset(X_va_imp, label=y_va, reference=lgb_train)
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42 + fold,
        "verbose": -1,
        "n_jobs": -1,
    }
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va_imp)
    test_lgb += lgb_model.predict(X_te_imp) / n_splits

    # Base Model 2: XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb.fit(X_tr_imp, y_tr, eval_set=[(X_va_imp, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb.predict(X_va_imp)
    test_xgb += xgb.predict(X_te_imp) / n_splits

    # Base Model 3: CatBoost Regressor
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=42 + fold,
        verbose=0,
    )
    cb.fit(
        X_tr_imp,
        y_tr,
        eval_set=(X_va_imp, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = cb.predict(X_va_imp)
    test_cb += cb.predict(X_te_imp) / n_splits

    # Base Model 4: ExtraTrees Regressor
    et = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et.fit(X_tr_imp, y_tr)
    oof_et[val_idx] = et.predict(X_va_imp)
    test_et += et.predict(X_te_imp) / n_splits

    # Base Model 5: HistGradientBoosting Regressor
    hist = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_depth=6,
        l2_regularization=1.0,
        random_state=42 + fold,
    )
    hist.fit(X_tr_imp, y_tr)
    oof_hist[val_idx] = hist.predict(X_va_imp)
    test_hist += hist.predict(X_te_imp) / n_splits

    # Base Model 6: PyTorch Gated Tabular ResNet
    input_dim = X_tr_scaled.shape[1]
    net = TabularResNet(
        in_features=input_dim, hidden_dim=128, num_blocks=3, dropout=0.15
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32).unsqueeze(1),
    )

    batch_size = min(128, max(32, len(train_dataset) // 20))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_net_state = None

    epochs = 40
    for epoch in range(epochs):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = net(bx)
                val_loss += criterion(out, by).item() * len(by)
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import copy
            best_net_state = copy.deepcopy(net.state_dict())

    if best_net_state is not None:
        net.load_state_dict(best_net_state)

    net.eval()
    with torch.no_grad():
        oof_nn[val_idx] = (
            net(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
        )
        test_nn += (
            net(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
            .flatten()
            / n_splits
        )

    fold_blend = (
        0.20 * oof_lgb[val_idx]
        + 0.20 * oof_xgb[val_idx]
        + 0.20 * oof_cb[val_idx]
        + 0.15 * oof_et[val_idx]
        + 0.15 * oof_hist[val_idx]
        + 0.10 * oof_nn[val_idx]
    )
    fold_rmse = np.sqrt(np.mean((y_va - fold_blend) ** 2))
    print(f"Fold {fold + 1} Blend RMSE: {fold_rmse:.5f}")


# 5. Ridge + Random Forest Meta-Learner Stacking with internal OOF
OOF_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_et, oof_hist, oof_nn])
TEST_matrix = np.column_stack([test_lgb, test_xgb, test_cb, test_et, test_hist, test_nn])

oof_ridge = np.zeros(len(y))
oof_rf = np.zeros(len(y))
test_ridge = np.zeros(len(TEST_matrix))
test_rf = np.zeros(len(TEST_matrix))

meta_kf = KFold(n_splits=10, shuffle=True, random_state=42)
for m_tr_idx, m_va_idx in meta_kf.split(OOF_matrix):
    m_X_tr, m_y_tr = OOF_matrix[m_tr_idx], y[m_tr_idx]
    m_X_va = OOF_matrix[m_va_idx]

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(m_X_tr, m_y_tr)
    oof_ridge[m_va_idx] = ridge.predict(m_X_va)
    test_ridge += ridge.predict(TEST_matrix) / 10.0

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(m_X_tr, m_y_tr)
    oof_rf[m_va_idx] = rf.predict(m_X_va)
    test_rf += rf.predict(TEST_matrix) / 10.0

oof_final = 0.5 * oof_ridge + 0.5 * oof_rf
test_final = 0.5 * test_ridge + 0.5 * test_rf

# Domain Boundary Clipping
y_min, y_max = float(y.min()), float(y.max())
oof_final = np.clip(oof_final, y_min, y_max)
test_final = np.clip(test_final, y_min, y_max)

final_score = np.sqrt(np.mean((y - oof_final) ** 2))

# Write Official Submission File
submission = pd.DataFrame({"id": test_ids, "Female (%)": test_final})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")