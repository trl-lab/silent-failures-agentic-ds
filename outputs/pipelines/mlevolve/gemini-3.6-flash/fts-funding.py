import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PyTorch Neural Network Architecture & Gating Modules
# ---------------------------------------------------------
class FeatureGating(nn.Module):
    """Squeeze-and-Excitation style feature gating mechanism for tabular inputs."""

    def __init__(self, input_dim: int, reduction_ratio: int = 4):
        super(FeatureGating, self).__init__()
        reduced_dim = max(4, input_dim // reduction_ratio)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class TabularResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, Feature Gating, and Dropout."""

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(TabularResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gating = FeatureGating(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.gating(out)
        return residual + out


class TabularGatedResNet(nn.Module):
    """Deep Gated Residual Network for Tabular Regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super(TabularGatedResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1)


# ---------------------------------------------------------
# Training helper for PyTorch Model
# ---------------------------------------------------------
def train_nn_model(
    X_tr,
    y_tr_log,
    X_va,
    y_va_log,
    input_dim,
    device,
    epochs=25,
    batch_size=256,
):
    model = TabularGatedResNet(input_dim=input_dim).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    va_tensor_x = torch.tensor(X_va, dtype=torch.float32).to(device)
    va_tensor_y = torch.tensor(y_va_log, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            total_samples += bx.size(0)

        scheduler.step()
        train_loss = running_loss / total_samples

        model.eval()
        with torch.no_grad():
            va_preds = model(va_tensor_x)
            val_loss = criterion(va_preds, va_tensor_y).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"  NN Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    model.load_state_dict(best_weights)
    return model


def predict_nn(model, X_data, device, batch_size=512):
    model.eval()
    dataset = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            p = model(bx)
            preds_list.append(p.cpu().numpy())
    return np.concatenate(preds_list, axis=0)


# ---------------------------------------------------------
# Data Preprocessing Pipeline
# ---------------------------------------------------------
def load_and_preprocess_data():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "amountUSD"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    y_raw = train_df[target_col].values
    y_log = np.log1p(np.clip(y_raw, 0, None))
    train_df = train_df.drop(columns=[target_col])

    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))
    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))

    test_ids = test_df["id"].values

    all_cols = [c for c in train_df.columns if c != "id"]

    date_cols, text_cols, cat_cols, num_cols = [], [], [], []

    for col in all_cols:
        col_type = train_df[col].dtype
        if (
            pd.api.types.is_datetime64_any_dtype(col_type)
            or "date" in col.lower()
            or "created" in col.lower()
            or "time" in col.lower()
        ):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_type):
            num_cols.append(col)
        elif pd.api.types.is_string_dtype(col_type) or col_type == object:
            sample_vals = train_df[col].dropna().astype(str).head(100)
            avg_len = sample_vals.str.len().mean() if len(sample_vals) > 0 else 0
            if avg_len > 60:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            cat_cols.append(col)

    # Log1p transformation for skewed numerical inputs
    for col in num_cols:
        tr_vals = train_df[col].dropna()
        if len(tr_vals) > 0 and tr_vals.skew() > 1.0:
            min_val = train_df[col].min()
            if min_val >= 0:
                train_df[col] = np.log1p(train_df[col])
                test_df[col] = np.log1p(test_df[col])
            else:
                shift = abs(min_val)
                train_df[col] = np.log1p(train_df[col] + shift)
                test_df[col] = np.log1p(test_df[col] + shift)

    # Quantile rank transformations for numerical variables
    for col in list(num_cols):
        train_df[f"{col}_qrank"] = train_df[col].rank(pct=True)
        test_df[f"{col}_qrank"] = test_df[col].rank(pct=True)

    # Pairwise numerical ratio and difference features
    num_list = list(num_cols)
    for i in range(len(num_list)):
        for j in range(i + 1, len(num_list)):
            c1, c2 = num_list[i], num_list[j]
            train_df[f"{c1}_minus_{c2}"] = train_df[c1] - train_df[c2]
            test_df[f"{c1}_minus_{c2}"] = test_df[c1] - test_df[c2]
            eps = 1e-5
            train_df[f"{c1}_ratio_{c2}"] = (train_df[c1] + eps) / (train_df[c2] + eps)
            test_df[f"{c1}_ratio_{c2}"] = (test_df[c1] + eps) / (test_df[c2] + eps)

    # Composite categorical combinations before target encoding
    orig_cat_cols = list(cat_cols)
    for i in range(len(orig_cat_cols)):
        for j in range(i + 1, len(orig_cat_cols)):
            c1, c2 = orig_cat_cols[i], orig_cat_cols[j]
            combo_col = f"{c1}_{c2}_combo"
            train_df[combo_col] = train_df[c1].astype(str) + "_" + train_df[c2].astype(str)
            test_df[combo_col] = test_df[c1].astype(str) + "_" + test_df[c2].astype(str)
            cat_cols.append(combo_col)

    # Categorical Group Aggregation Statistics
    for cat in cat_cols:
        for num in num_cols:
            grp = train_df.groupby(cat)[num]
            mean_map = grp.mean().to_dict()
            std_map = grp.std().to_dict()
            min_map = grp.min().to_dict()
            max_map = grp.max().to_dict()

            train_df[f"{cat}_{num}_grp_mean"] = train_df[cat].map(mean_map)
            train_df[f"{cat}_{num}_grp_std"] = train_df[cat].map(std_map)
            train_df[f"{cat}_{num}_grp_min"] = train_df[cat].map(min_map)
            train_df[f"{cat}_{num}_grp_max"] = train_df[cat].map(max_map)

            test_df[f"{cat}_{num}_grp_mean"] = test_df[cat].map(mean_map)
            test_df[f"{cat}_{num}_grp_std"] = test_df[cat].map(std_map)
            test_df[f"{cat}_{num}_grp_min"] = test_df[cat].map(min_map)
            test_df[f"{cat}_{num}_grp_max"] = test_df[cat].map(max_map)

    # Process Date Columns
    for col in date_cols:
        train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
        test_df[col] = pd.to_datetime(test_df[col], errors="coerce")

        for df in [train_df, test_df]:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_timestamp"] = pd.to_numeric(df[col], errors="coerce") / 1e9

        train_df = train_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])

    # Process Text Columns via TF-IDF + SVD
    for col in text_cols:
        train_text = train_df[col].fillna("").astype(str)
        test_text = test_df[col].fillna("").astype(str)

        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        train_tfidf = tfidf.fit_transform(train_text)
        test_tfidf = tfidf.transform(test_text)

        n_comp = min(12, train_tfidf.shape[1] - 1) if train_tfidf.shape[1] > 1 else 1
        if n_comp > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            train_svd = svd.fit_transform(train_tfidf)
            test_svd = svd.transform(test_tfidf)

            for i in range(n_comp):
                train_df[f"{col}_svd_{i}"] = train_svd[:, i]
                test_df[f"{col}_svd_{i}"] = test_svd[:, i]

        train_df = train_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])

    # Categorical Frequency & Out-of-Fold Target Encoding
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y_log.mean()

    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
        test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

        train_df[f"{col}_te"] = np.nan
        for tr_idx, val_idx in kf.split(train_df):
            fold_tr_cats = train_df.iloc[tr_idx][col]
            fold_tr_y = pd.Series(y_log[tr_idx], index=train_df.index[tr_idx])
            mapping = fold_tr_y.groupby(fold_tr_cats).mean().to_dict()
            train_df.iloc[val_idx, train_df.columns.get_loc(f"{col}_te")] = (
                train_df.iloc[val_idx][col].map(mapping)
            )

        train_df[f"{col}_te"] = train_df[f"{col}_te"].fillna(global_mean)
        full_mapping = (
            pd.Series(y_log, index=train_df.index)
            .groupby(train_df[col])
            .mean()
            .to_dict()
        )
        test_df[f"{col}_te"] = test_df[col].map(full_mapping).fillna(global_mean)

        train_df[col] = train_df[col].astype("category").cat.codes
        test_df[col] = test_df[col].astype("category").cat.codes

    feature_cols = [c for c in train_df.columns if c != "id"]

    for col in feature_cols:
        median_val = train_df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    return X_train, y_raw, X_test, test_ids, feature_cols


# ---------------------------------------------------------
# Main Cross-Validation Ensemble Pipeline
# ---------------------------------------------------------
def main():
    X_train, y_raw, X_test, test_ids, feature_cols = load_and_preprocess_data()
    y_log = np.log1p(np.clip(y_raw, 0, None))

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_models = np.zeros((len(X_train), 4), dtype=np.float32)
    test_models = np.zeros((len(X_test), 4), dtype=np.float32)

    input_dim = X_train.shape[1]

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_idx])
        X_va = scaler.transform(X_train[val_idx])
        X_te = scaler.transform(X_test)
        y_tr_log = y_log[tr_idx]
        y_va_log = y_log[val_idx]

        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2500,
            learning_rate=0.015,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.015,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=100,
        )
        xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)

        # 3. CatBoost
        cat_model = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.015,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=100,
        )
        cat_model.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)

        # 4. TabularGatedResNet
        nn_model = train_nn_model(
            X_tr,
            y_tr_log,
            X_va,
            y_va_log,
            input_dim=input_dim,
            device=device,
            epochs=25,
            batch_size=256,
        )

        # Fold Predictions (Validation)
        pred_lgb_va = lgb_model.predict(X_va)
        pred_xgb_va = xgb_model.predict(X_va)
        pred_cat_va = cat_model.predict(X_va)
        pred_nn_va = predict_nn(nn_model, X_va, device=device)

        oof_models[val_idx, 0] = pred_lgb_va
        oof_models[val_idx, 1] = pred_xgb_va
        oof_models[val_idx, 2] = pred_cat_va
        oof_models[val_idx, 3] = pred_nn_va

        # Fold Predictions (Test)
        pred_lgb_te = lgb_model.predict(X_te)
        pred_xgb_te = xgb_model.predict(X_te)
        pred_cat_te = cat_model.predict(X_te)
        pred_nn_te = predict_nn(nn_model, X_te, device=device)

        test_models[:, 0] += pred_lgb_te / n_splits
        test_models[:, 1] += pred_xgb_te / n_splits
        test_models[:, 2] += pred_cat_te / n_splits
        test_models[:, 3] += pred_nn_te / n_splits

    # Convert OOF and Test predictions to raw USD scale via expm1
    oof_models_raw = np.expm1(np.clip(oof_models, 0, None))
    test_models_raw = np.expm1(np.clip(test_models, 0, None))

    # Solve for optimal blend weights directly on raw scale minimizing true RMSE
    def objective(weights):
        pred_raw = np.dot(oof_models_raw, weights)
        return root_mean_squared_error(y_raw, pred_raw)

    n_models = oof_models_raw.shape[1]
    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x
    best_weights = np.maximum(0, best_weights)
    if np.sum(best_weights) > 0:
        best_weights = best_weights / np.sum(best_weights)

    oof_preds_raw = np.dot(oof_models_raw, best_weights)
    score = root_mean_squared_error(y_raw, oof_preds_raw)

    test_preds_raw = np.dot(test_models_raw, best_weights)
    test_preds_raw = np.nan_to_num(test_preds_raw, nan=0.0)

    # Generate Submission
    sub_df = pd.DataFrame({"id": test_ids, "amountUSD": test_preds_raw})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()