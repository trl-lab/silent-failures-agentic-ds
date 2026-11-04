import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# 1. Reproducibility & Setup
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# 2. Data Loading & Feature Engineering
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "DEPSUM"
id_col = "id"

if id_col not in test_df.columns:
    test_df[id_col] = np.arange(len(test_df))
if id_col not in train_df.columns:
    train_df[id_col] = np.arange(len(train_df))

y_train = train_df[target_col].values.astype(np.float32)
test_ids = test_df[id_col].values

feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

datetime_cols = []
numeric_cols = []
categorical_cols = []

for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(X_train[c]):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(X_train[c]):
        numeric_cols.append(c)
    else:
        try:
            converted = pd.to_datetime(X_train[c], errors="coerce")
            if converted.notna().sum() > 0.5 * len(X_train):
                X_train[c] = converted
                X_test[c] = pd.to_datetime(X_test[c], errors="coerce")
                datetime_cols.append(c)
                continue
        except Exception:
            pass
        categorical_cols.append(c)

for c in datetime_cols:
    for df in [X_train, X_test]:
        df[f"{c}_year"] = df[c].dt.year
        df[f"{c}_month"] = df[c].dt.month
        df[f"{c}_day"] = df[c].dt.day
        df[f"{c}_dayofweek"] = df[c].dt.dayofweek
        df[f"{c}_dayofyear"] = df[c].dt.dayofyear
        df[f"{c}_quarter"] = df[c].dt.quarter
        df[f"{c}_is_weekend"] = df[c].dt.dayofweek.isin([5, 6]).astype(int)
        df[f"{c}_sin_month"] = np.sin(2 * np.pi * df[c].dt.month / 12.0)
        df[f"{c}_cos_month"] = np.cos(2 * np.pi * df[c].dt.month / 12.0)
        df[f"{c}_sin_dow"] = np.sin(2 * np.pi * df[c].dt.dayofweek / 7.0)
        df[f"{c}_cos_dow"] = np.cos(2 * np.pi * df[c].dt.dayofweek / 7.0)
    X_train.drop(columns=[c], inplace=True)
    X_test.drop(columns=[c], inplace=True)

numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

for c in numeric_cols:
    train_nulls = X_train[c].isna().sum()
    if train_nulls > 0:
        X_train[f"{c}_isna"] = X_train[c].isna().astype(int)
        X_test[f"{c}_isna"] = X_test[c].isna().astype(int)
    median_val = X_train[c].median()
    if pd.isna(median_val):
        median_val = 0.0
    X_train[c] = X_train[c].fillna(median_val)
    X_test[c] = X_test[c].fillna(median_val)

# Apply log1p transformation to highly skewed numeric features (skew > 1.5)
for c in numeric_cols:
    if X_train[c].skew() > 1.5:
        X_train[c] = np.log1p(np.maximum(0, X_train[c]))
        X_test[c] = np.log1p(np.maximum(0, X_test[c]))

for c in categorical_cols:
    X_train[c] = X_train[c].astype(str).fillna("missing")
    X_test[c] = X_test[c].astype(str).fillna("missing")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for c in categorical_cols:
    freq_map = X_train[c].value_counts(normalize=True).to_dict()
    X_train[f"{c}_freq"] = X_train[c].map(freq_map).fillna(0)
    X_test[f"{c}_freq"] = X_test[c].map(freq_map).fillna(0)

    X_train[f"{c}_te"] = np.nan
    for tr_idx, va_idx in kf.split(X_train):
        tr_df_cat = pd.DataFrame(
            {"cat": X_train.iloc[tr_idx][c], "target": y_train[tr_idx]}
        )
        te_map = tr_df_cat.groupby("cat")["target"].mean().to_dict()
        X_train.iloc[va_idx, X_train.columns.get_loc(f"{c}_te")] = X_train.iloc[va_idx][
            c
        ].map(te_map).values

    global_mean = float(y_train.mean())
    X_train[f"{c}_te"] = X_train[f"{c}_te"].fillna(global_mean)

    full_df_cat = pd.DataFrame({"cat": X_train[c], "target": y_train})
    full_te_map = full_df_cat.groupby("cat")["target"].mean().to_dict()
    X_test[f"{c}_te"] = X_test[c].map(full_te_map).fillna(global_mean)

    categories = sorted(list(set(X_train[c].unique()).union(set(X_test[c].unique()))))
    cat_to_code = {cat: i for i, cat in enumerate(categories)}
    X_train[c] = X_train[c].map(cat_to_code).astype(int)
    X_test[c] = X_test[c].map(cat_to_code).astype(int)

num_cols_updated = [
    c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
]
if len(num_cols_updated) >= 3:
    for df in [X_train, X_test]:
        df["num_row_mean"] = df[num_cols_updated].mean(axis=1)
        df["num_row_std"] = df[num_cols_updated].std(axis=1).fillna(0)
        df["num_row_min"] = df[num_cols_updated].min(axis=1)
        df["num_row_max"] = df[num_cols_updated].max(axis=1)

# Multi-statistic group aggregations (mean, std, min, max, median) across top categorical variables
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    top_cats = categorical_cols[: min(3, len(categorical_cols))]
    top_num_cols = numeric_cols[: min(5, len(numeric_cols))]
    for cat_c in top_cats:
        for num_c in top_num_cols:
            grp_stats = (
                X_train.groupby(cat_c)[num_c]
                .agg(["mean", "std", "min", "max", "median"])
                .reset_index()
            )
            grp_stats.columns = [cat_c] + [
                f"{cat_c}_{num_c}_grp_{stat}"
                for stat in ["mean", "std", "min", "max", "median"]
            ]

            X_train = X_train.merge(grp_stats, on=cat_c, how="left")
            X_test = X_test.merge(grp_stats, on=cat_c, how="left")

            for stat in ["mean", "std", "min", "max", "median"]:
                col_name = f"{cat_c}_{num_c}_grp_{stat}"
                fill_val = X_train[num_c].mean() if stat != "std" else 0.0
                X_train[col_name] = X_train[col_name].fillna(fill_val)
                X_test[col_name] = X_test[col_name].fillna(fill_val)

# Select numeric features for PCA transformation inside CV
num_cols_pca = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

const_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
if const_cols:
    X_train.drop(columns=const_cols, inplace=True)
    X_test.drop(columns=const_cols, inplace=True)

X_train = X_train.copy()
X_test = X_test.copy()

X_mat = X_train.values.astype(np.float32)
X_test_mat = X_test.values.astype(np.float32)
y_mat = y_train.astype(np.float32)


# 3. Neural Network Model Architecture
class ResBlock(nn.Module):
    def __init__(self, features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.bn1 = nn.BatchNorm1d(features)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(features, features)
        self.bn2 = nn.BatchNorm1d(features)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.act2(out + residual)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(1)


def train_nn_fold(X_tr, y_tr, X_va, y_va, X_te, device, input_dim):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-8
    y_tr_s = (y_tr - y_mean) / y_std
    y_va_s = (y_va - y_mean) / y_std

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_s, dtype=torch.float32),
    )
    va_dataset = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va_s, dtype=torch.float32),
    )
    te_dataset = TensorDataset(torch.tensor(X_te_s, dtype=torch.float32))

    batch_size = min(256, max(32, len(X_tr_s) // 20))
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_dataset, batch_size=512, shuffle=False)

    model = TabularResNet(
        input_dim=input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    best_weights = None
    patience = 12
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                va_loss += criterion(out, by).item() * len(by)
        va_loss /= len(va_dataset)
        scheduler.step(va_loss)

        if va_loss < best_loss:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()

    val_preds = []
    with torch.no_grad():
        for bx, _ in va_loader:
            bx = bx.to(device)
            pred = model(bx).cpu().numpy()
            val_preds.append(pred)
    val_preds = np.concatenate(val_preds) * y_std + y_mean

    test_preds = []
    with torch.no_grad():
        for (bx,) in te_loader:
            bx = bx.to(device)
            pred = model(bx).cpu().numpy()
            test_preds.append(pred)
    test_preds = np.concatenate(test_preds) * y_std + y_mean

    return val_preds, test_preds


# 4. Model Training & Cross-Validation
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

cb_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}

cv_splits = list(kf.split(X_mat, y_mat))
n_samples = len(y_mat)
n_test = len(X_test_mat)
n_folds = len(cv_splits)

oof_lgb = np.zeros(n_samples)
oof_lgb_log = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_cat_log = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_lgb_log = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_cat_log = np.zeros(n_test)
test_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_mat.shape[1]

for fold, (tr_idx, va_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_mat[tr_idx].copy(), y_mat[tr_idx]
    X_va, y_va = X_mat[va_idx].copy(), y_mat[va_idx]
    X_te_curr = X_test_mat.copy()

    if len(num_cols_pca) >= 5:
        pca_scaler = StandardScaler()
        X_tr_pca_sc = pca_scaler.fit_transform(X_tr[:, :len(num_cols_pca)])
        X_va_pca_sc = pca_scaler.transform(X_va[:, :len(num_cols_pca)])
        X_te_pca_sc = pca_scaler.transform(X_te_curr[:, :len(num_cols_pca)])

        pca = PCA(n_components=5, random_state=42)
        pca_tr = pca.fit_transform(X_tr_pca_sc)
        pca_va = pca.transform(X_va_pca_sc)
        pca_te = pca.transform(X_te_pca_sc)

        X_tr = np.hstack([X_tr, pca_tr.astype(np.float32)])
        X_va = np.hstack([X_va, pca_va.astype(np.float32)])
        X_te_curr = np.hstack([X_te_curr, pca_te.astype(np.float32)])

    y_tr_log = np.log1p(np.maximum(0, y_tr))
    y_va_log = np.log1p(np.maximum(0, y_va))

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[va_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_te_curr) / n_folds

    lgb_log_model = lgb.LGBMRegressor(**lgb_params)
    lgb_log_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb_log[va_idx] = np.maximum(0, np.expm1(lgb_log_model.predict(X_va)))
    test_lgb_log += np.maximum(0, np.expm1(lgb_log_model.predict(X_te_curr))) / n_folds

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_te_curr) / n_folds

    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )
    oof_cat[va_idx] = cb_model.predict(X_va)
    test_cat += cb_model.predict(X_te_curr) / n_folds

    cb_log_model = cb.CatBoostRegressor(**cb_params)
    cb_log_model.fit(
        X_tr,
        y_tr_log,
        eval_set=(X_va, y_va_log),
        verbose=False,
    )
    oof_cat_log[va_idx] = np.maximum(0, np.expm1(cb_log_model.predict(X_va)))
    test_cat_log += np.maximum(0, np.expm1(cb_log_model.predict(X_te_curr))) / n_folds

    val_nn, te_nn = train_nn_fold(X_tr, y_tr, X_va, y_va, X_te_curr, device, X_tr.shape[1])
    oof_nn[va_idx] = val_nn
    test_nn += te_nn / n_folds


# 5. Out-of-Fold Ensemble Optimization & Final Prediction
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


oof_matrix = np.column_stack([oof_lgb, oof_lgb_log, oof_xgb, oof_cat, oof_cat_log, oof_nn])
test_matrix = np.column_stack([test_lgb, test_lgb_log, test_xgb, test_cat, test_cat_log, test_nn])


def loss_func(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    return rmse(y_mat, pred)


init_w = np.ones(6) / 6.0
bounds = [(0, 1) for _ in range(6)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

res = minimize(
    loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints
)
if res.success and np.sum(res.x) > 0:
    best_w = res.x / np.sum(res.x)
else:
    best_w = init_w

final_oof_pred = np.dot(oof_matrix, best_w)
final_test_pred = np.dot(test_matrix, best_w)

if np.min(y_mat) >= 0:
    final_oof_pred = np.maximum(0, final_oof_pred)
    final_test_pred = np.maximum(0, final_test_pred)

final_score = rmse(y_mat, final_oof_pred)

# 6. Export Submission CSV
submission = pd.DataFrame({id_col: test_ids, target_col: final_test_pred})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")