import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ==========================================
# 1. PyTorch Neural Architecture
# ==========================================
class TabularResNetBlock(nn.Module):
    """Residual block with Batch Normalization, Dropout, and SiLU activations."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)

        if in_features != hidden_features:
            self.shortcut = nn.Linear(in_features, hidden_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.act(out + residual)
        return out


class DeepTabularModel(nn.Module):
    """Multi-layer Tabular ResNet architecture with dropout regularization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(TabularResNetBlock(prev_dim, h_dim, dropout=dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


# ==========================================
# 2. Main Execution Pipeline
# ==========================================
def main():
    # Setup directories
    input_dir = "./input"
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # --------------------------------------
    # Data Loading
    # --------------------------------------
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    # Identify target column
    target_col = "Amount"
    if target_col not in train_df.columns:
        possible = [c for c in train_df.columns if c.lower() == "amount"]
        target_col = possible[0] if possible else train_df.columns[-1]

    # Handle ID column
    id_col = None
    for col in ["id", "ID", "Id", "index", "Index"]:
        if col in test_df.columns:
            id_col = col
            break

    if id_col is not None:
        test_ids = test_df[id_col].values
    else:
        test_ids = np.arange(len(test_df))

    # Extract raw target and compute log1p transformed target
    y_raw = train_df[target_col].values.astype(float)
    y_log = np.log1p(np.maximum(0, y_raw))

    # Identify feature columns
    exclude_cols = {target_col}
    if id_col:
        exclude_cols.add(id_col)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Classify feature types
    date_cols = []
    text_cols = []
    num_cols = []

    for c in feature_cols:
        col_type = train_df[c].dtype
        col_name_lower = c.lower()

        if np.issubdtype(col_type, np.number):
            num_cols.append(c)
        elif (
            np.issubdtype(col_type, np.datetime64)
            or "date" in col_name_lower
            or "year" in col_name_lower
            or "period" in col_name_lower
        ):
            date_cols.append(c)
        else:
            text_cols.append(c)

    # --------------------------------------
    # Date / Time Feature Engineering
    # --------------------------------------
    train_date_feats = pd.DataFrame(index=train_df.index)
    test_date_feats = pd.DataFrame(index=test_df.index)

    tr_dt_dict = {}
    te_dt_dict = {}

    for col in date_cols:
        tr_dt = pd.to_datetime(train_df[col], errors="coerce")
        te_dt = pd.to_datetime(test_df[col], errors="coerce")

        tr_dt = tr_dt.mask((tr_dt.dt.year < 1900) | (tr_dt.dt.year > 2100))
        te_dt = te_dt.mask((te_dt.dt.year < 1900) | (te_dt.dt.year > 2100))

        tr_dt_dict[col] = tr_dt
        te_dt_dict[col] = te_dt

        prefix = f"dt_{col}_"
        train_date_feats[prefix + "year"] = tr_dt.dt.year.fillna(-1)
        test_date_feats[prefix + "year"] = te_dt.dt.year.fillna(-1)

        train_date_feats[prefix + "month"] = tr_dt.dt.month.fillna(-1)
        test_date_feats[prefix + "month"] = te_dt.dt.month.fillna(-1)

        train_date_feats[prefix + "quarter"] = tr_dt.dt.quarter.fillna(-1)
        test_date_feats[prefix + "quarter"] = te_dt.dt.quarter.fillna(-1)

        train_date_feats[prefix + "dayofweek"] = tr_dt.dt.dayofweek.fillna(-1)
        test_date_feats[prefix + "dayofweek"] = te_dt.dt.dayofweek.fillna(-1)

        epoch = pd.Timestamp("2020-01-01")
        train_date_feats[prefix + "days_epoch"] = (tr_dt - epoch).dt.days.fillna(-999)
        test_date_feats[prefix + "days_epoch"] = (te_dt - epoch).dt.days.fillna(-999)

    if len(date_cols) >= 2:
        for i in range(len(date_cols)):
            for j in range(i + 1, len(date_cols)):
                c1, c2 = date_cols[i], date_cols[j]
                dt1_tr = tr_dt_dict[c1]
                dt2_tr = tr_dt_dict[c2]
                dt1_te = te_dt_dict[c1]
                dt2_te = te_dt_dict[c2]

                diff_name = f"dt_diff_{c1}_vs_{c2}"
                train_date_feats[diff_name] = (dt1_tr - dt2_tr).dt.days.fillna(-999)
                test_date_feats[diff_name] = (dt1_te - dt2_te).dt.days.fillna(-999)

    # --------------------------------------
    # Text Embeddings (TF-IDF + SVD)
    # --------------------------------------
    if text_cols:
        tr_text_concat = (
            train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
        te_text_concat = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        tfidf = TfidfVectorizer(
            max_features=5000,
            sublinear_tf=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        tr_tfidf = tfidf.fit_transform(tr_text_concat)
        te_tfidf = tfidf.transform(te_text_concat)

        svd_comp = min(32, max(2, tr_tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=svd_comp, random_state=42)
        tr_svd = svd.fit_transform(tr_tfidf)
        te_svd = svd.transform(te_tfidf)

        svd_cols = [f"tfidf_svd_{i}" for i in range(svd_comp)]
        train_text_feats = pd.DataFrame(tr_svd, columns=svd_cols, index=train_df.index)
        test_text_feats = pd.DataFrame(te_svd, columns=svd_cols, index=test_df.index)
    else:
        train_text_feats = pd.DataFrame(index=train_df.index)
        test_text_feats = pd.DataFrame(index=test_df.index)

    # --------------------------------------
    # Frequency Encoding
    # --------------------------------------
    train_freq_feats = pd.DataFrame(index=train_df.index)
    test_freq_feats = pd.DataFrame(index=test_df.index)

    for col in text_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_freq_feats[f"freq_{col}"] = train_df[col].map(freq_map).fillna(0.0)
        test_freq_feats[f"freq_{col}"] = test_df[col].map(freq_map).fillna(0.0)

    # --------------------------------------
    # Leak-Free Out-of-Fold Target Encoding
    # --------------------------------------
    train_te_feats = pd.DataFrame(index=train_df.index)
    test_te_feats = pd.DataFrame(index=test_df.index)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_target_mean = y_log.mean()
    m_smoothing = 10.0

    for col in text_cols:
        col_te_tr = pd.Series(index=train_df.index, dtype=float)

        for tr_idx, val_idx in kf.split(train_df):
            tr_cat = train_df[col].iloc[tr_idx]
            tr_y = y_log[tr_idx]

            grouped = (
                pd.DataFrame({"cat": tr_cat, "target": tr_y})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            smooth = (
                grouped["count"] * grouped["mean"] + m_smoothing * global_target_mean
            ) / (grouped["count"] + m_smoothing)

            val_cat = train_df[col].iloc[val_idx]
            col_te_tr.iloc[val_idx] = val_cat.map(smooth).fillna(global_target_mean)

        train_te_feats[f"te_{col}"] = col_te_tr

        grouped_full = (
            pd.DataFrame({"cat": train_df[col], "target": y_log})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smooth_full = (
            grouped_full["count"] * grouped_full["mean"]
            + m_smoothing * global_target_mean
        ) / (grouped_full["count"] + m_smoothing)
        test_te_feats[f"te_{col}"] = (
            test_df[col].map(smooth_full).fillna(global_target_mean)
        )

    # --------------------------------------
    # Numerical Feature Processing
    # --------------------------------------
    train_num_feats = pd.DataFrame(index=train_df.index)
    test_num_feats = pd.DataFrame(index=test_df.index)

    for col in num_cols:
        tr_val = train_df[col].astype(float)
        te_val = test_df[col].astype(float)

        train_num_feats[f"{col}_isna"] = tr_val.isna().astype(float)
        test_num_feats[f"{col}_isna"] = te_val.isna().astype(float)

        med = tr_val.median()
        if pd.isna(med):
            med = 0.0

        tr_imp = tr_val.fillna(med)
        te_imp = te_val.fillna(med)

        train_num_feats[col] = tr_imp
        test_num_feats[col] = te_imp

        if (tr_imp >= 0).all():
            train_num_feats[f"log1p_{col}"] = np.log1p(tr_imp)
            test_num_feats[f"log1p_{col}"] = np.log1p(te_imp)

    # Assemble and Scale
    X_train_df = pd.concat(
        [
            train_num_feats,
            train_date_feats,
            train_text_feats,
            train_freq_feats,
            train_te_feats,
        ],
        axis=1,
    )
    X_test_df = pd.concat(
        [
            test_num_feats,
            test_date_feats,
            test_text_feats,
            test_freq_feats,
            test_te_feats,
        ],
        axis=1,
    )

    fold_indices = np.zeros(len(train_df), dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        fold_indices[val_idx] = fold

    # --------------------------------------
    # Model Training Setup
    # --------------------------------------
    n_samples = len(X_train_df)
    n_test = len(X_test_df)
    num_folds = 5

    oof_preds = np.zeros((n_samples, 4))
    test_preds = np.zeros((n_test, 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "random_state": 42,
        "n_estimators": 1200,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_estimators": 1200,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": 42,
        "iterations": 1200,
        "verbose": False,
        "early_stopping_rounds": 50,
    }

    # --------------------------------------
    # Cross-Validation Training Loop
    # --------------------------------------
    for fold in range(num_folds):
        val_idx = np.where(fold_indices == fold)[0]
        tr_idx = np.where(fold_indices != fold)[0]

        scaler = StandardScaler()
        X_tr = pd.DataFrame(
            scaler.fit_transform(X_train_df.iloc[tr_idx]),
            columns=X_train_df.columns,
            index=X_train_df.iloc[tr_idx].index,
        ).fillna(0.0)
        X_va = pd.DataFrame(
            scaler.transform(X_train_df.iloc[val_idx]),
            columns=X_train_df.columns,
            index=X_train_df.iloc[val_idx].index,
        ).fillna(0.0)
        X_te = pd.DataFrame(
            scaler.transform(X_test_df),
            columns=X_test_df.columns,
            index=X_test_df.index,
        ).fillna(0.0)
        y_tr_log, y_va_log = y_log[tr_idx], y_log[val_idx]

        # Model 0: LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_p_lgb = np.maximum(0, np.expm1(model_lgb.predict(X_va)))
        test_p_lgb = np.maximum(0, np.expm1(model_lgb.predict(X_te)))

        oof_preds[val_idx, 0] = val_p_lgb
        test_preds[:, 0] += test_p_lgb / num_folds

        # Model 1: XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
        val_p_xgb = np.maximum(0, np.expm1(model_xgb.predict(X_va)))
        test_p_xgb = np.maximum(0, np.expm1(model_xgb.predict(X_te)))

        oof_preds[val_idx, 1] = val_p_xgb
        test_preds[:, 1] += test_p_xgb / num_folds

        # Model 2: CatBoost
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
        val_p_cb = np.maximum(0, np.expm1(model_cb.predict(X_va)))
        test_p_cb = np.maximum(0, np.expm1(model_cb.predict(X_te)))

        oof_preds[val_idx, 2] = val_p_cb
        test_preds[:, 2] += test_p_cb / num_folds

        # Model 3: PyTorch DeepTabularModel
        train_ds = TensorDataset(
            torch.tensor(X_tr.values, dtype=torch.float32),
            torch.tensor(y_tr_log, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_va.values, dtype=torch.float32),
            torch.tensor(y_va_log, dtype=torch.float32),
        )

        tr_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
        va_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)

        pt_model = DeepTabularModel(input_dim=X_tr.shape[1]).to(device)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(pt_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_loss = float("inf")
        best_state = None
        patience, p_counter = 5, 0

        for epoch in range(25):
            pt_model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = pt_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            pt_model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = pt_model(bx)
                    v_loss += criterion(out, by).item() * len(by)
            v_loss /= len(val_ds)

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {
                    k: v.cpu().clone() for k, v in pt_model.state_dict().items()
                }
                p_counter = 0
            else:
                p_counter += 1
                if p_counter >= patience:
                    break

        if best_state is not None:
            pt_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        pt_model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va.values, dtype=torch.float32).to(device)
            val_p_pt = np.maximum(0, np.expm1(pt_model(va_tensor).cpu().numpy()))

            te_tensor = torch.tensor(X_te.values, dtype=torch.float32).to(device)
            test_p_pt = np.maximum(0, np.expm1(pt_model(te_tensor).cpu().numpy()))

        oof_preds[val_idx, 3] = val_p_pt
        test_preds[:, 3] += test_p_pt / num_folds

    # --------------------------------------
    # Ensemble Optimization (SLSQP)
    # --------------------------------------
    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend_oof = np.dot(oof_preds, w)
        return np.sqrt(mean_squared_error(y_raw, blend_oof))

    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0.0, 1.0)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    opt_res = minimize(
        blend_objective,
        x0=init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_weights = opt_res.x / np.sum(opt_res.x)

    final_oof = np.dot(oof_preds, opt_weights)
    final_score = np.sqrt(mean_squared_error(y_raw, final_oof))

    # --------------------------------------
    # Submission Generation
    # --------------------------------------
    final_test_pred = np.maximum(0, np.dot(test_preds, opt_weights))

    sub_df = pd.DataFrame({"id": test_ids, "Amount": final_test_pred})
    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()