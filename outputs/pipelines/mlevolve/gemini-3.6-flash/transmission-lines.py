import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

import catboost as cb
import lightgbm as lgb
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# 1. PyTorch TabularResNet Architecture
# ------------------------------------------------------------------------------
class TabularResBlock(nn.Module):
    """Residual Block with Layer Normalization, SiLU, and Dropout for Tabular Neural Nets."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.drop2(self.act2(self.ln2(self.fc2(out))))
        return residual + out


class TabularResNet(nn.Module):
    """Deep Tabular Residual Neural Network for Voltage Tier Classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_proj(x)
        for block in self.blocks:
            out = block(out)
        logits = self.head(out)
        return logits


# ------------------------------------------------------------------------------
# 2. Data Loading & Feature Engineering
# ------------------------------------------------------------------------------
def load_and_preprocess_data():
    """Loads parquet data, processes geometric and text features, and encodes target."""
    train_path = Path("./input/train.parquet")
    test_path = Path("./input/test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Submission ID mapping
    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    elif "ID" in test_df.columns:
        test_ids = test_df["ID"].values
    elif "OBJECTID" in test_df.columns:
        test_ids = test_df["OBJECTID"].values
    else:
        test_ids = np.arange(len(test_df))

    target_col = "VOLT_CLASS"
    y_raw = train_df[target_col].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    train_df = train_df.drop(columns=[target_col], errors="ignore")
    n_train = len(train_df)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Clean missing sentinels
    sentinels = [
        -999,
        -999999,
        9999,
        -999.0,
        "-999",
        "-999999",
        "NOT AVAILABLE",
        "UNKNOWN",
        "UNAVAILABLE",
        "NONE",
        "NULL",
        "",
    ]
    for col in full_df.columns:
        if full_df[col].dtype == object:
            full_df[col] = full_df[col].astype(str).str.strip()
            full_df[col] = full_df[col].replace(sentinels, np.nan)
        else:
            full_df[col] = full_df[col].replace([-999, -999999, 9999, -999.0], np.nan)

    # Regex Voltage parsing
    def extract_voltage_num(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).lower()
        match = re.search(r"(\d+(?:\.\d+)?)", val_str)
        if match:
            v = float(match.group(1))
            if 10 <= v <= 1500:
                return v
        return np.nan

    voltage_cols = [c for c in full_df.columns if "volt" in c.lower()]
    for vcol in voltage_cols:
        full_df[f"{vcol}_parsed"] = full_df[vcol].apply(extract_voltage_num)

    # Spatial Geometry Features
    spatial_cols = {col.upper(): col for col in full_df.columns}
    xmin_col = spatial_cols.get("XMIN")
    ymin_col = spatial_cols.get("YMIN")
    xmax_col = spatial_cols.get("XMAX")
    ymax_col = spatial_cols.get("YMAX")
    len_col = spatial_cols.get(
        "SHAPE_LENGTH", spatial_cols.get("LENGTH", spatial_cols.get("LINE_LENGTH"))
    )

    if xmin_col and ymin_col and xmax_col and ymax_col:
        dx = (full_df[xmax_col] - full_df[xmin_col]).abs()
        dy = (full_df[ymax_col] - full_df[ymin_col]).abs()
        full_df["bbox_dx"] = dx
        full_df["bbox_dy"] = dy
        full_df["bbox_diag"] = np.sqrt(dx**2 + dy**2)
        full_df["bbox_area"] = dx * dy
        full_df["bbox_aspect_ratio"] = dx / (dy + 1e-6)
        full_df["centroid_x"] = (full_df[xmin_col] + full_df[xmax_col]) / 2.0
        full_df["centroid_y"] = (full_df[ymin_col] + full_df[ymax_col]) / 2.0
        full_df["orientation_angle"] = np.arctan2(dy, dx + 1e-6)

        if len_col and len_col in full_df.columns:
            full_df["sinuosity"] = full_df[len_col] / (full_df["bbox_diag"] + 1e-5)

    # Spatial K-Nearest Neighbors (k-NN) Features
    if "centroid_x" in full_df.columns and "centroid_y" in full_df.columns:
        coords_train = (
            full_df.iloc[:n_train][["centroid_x", "centroid_y"]].fillna(0).values
        )
        coords_test = (
            full_df.iloc[n_train:][["centroid_x", "centroid_y"]].fillna(0).values
        )

        for k in [5, 15, 30]:
            nn_tr = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
            nn_tr.fit(coords_train)
            dists_tr, _ = nn_tr.kneighbors(coords_train)
            dists_tr_k = dists_tr[:, 1:]

            nn_te = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
            nn_te.fit(coords_train)
            dists_te_k, _ = nn_te.kneighbors(coords_test)

            dists_all = np.vstack([dists_tr_k, dists_te_k])

            full_df[f"knn_dist_mean_{k}"] = np.mean(dists_all, axis=1)
            full_df[f"knn_dist_std_{k}"] = np.std(dists_all, axis=1)
            full_df[f"knn_dist_min_{k}"] = np.min(dists_all, axis=1)
            full_df[f"knn_dist_max_{k}"] = np.max(dists_all, axis=1)
            full_df[f"knn_density_{k}"] = 1.0 / (full_df[f"knn_dist_mean_{k}"] + 1e-5)

    # Identify categorical text columns
    string_cols = [
        c
        for c in full_df.columns
        if full_df[c].dtype == object and c not in ["id", "ID", "OBJECTID"]
    ]

    # Frequency & Categorical Label Encoding
    for c in string_cols:
        freq = train_df[c].value_counts(normalize=True).to_dict()
        full_df[f"{c}_freq"] = full_df[c].map(freq).fillna(0)
        le = LabelEncoder()
        full_df[c] = le.fit_transform(full_df[c].fillna("MISSING").astype(str))

    # Clean IDs and boolean types
    drop_cols = [c for c in ["id", "ID", "OBJECTID"] if c in full_df.columns]
    full_df = full_df.drop(columns=drop_cols, errors="ignore")
    for col in full_df.columns:
        if full_df[col].dtype == bool:
            full_df[col] = full_df[col].astype(int)

    X_train = full_df.iloc[:n_train].copy().reset_index(drop=True)
    X_test = full_df.iloc[n_train:].copy().reset_index(drop=True)

    return X_train, y, X_test, test_ids, label_encoder, string_cols


# ------------------------------------------------------------------------------
# 3. Main Training & Cross-Validation Execution
# ------------------------------------------------------------------------------
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Process data
    X_train, y, X_test, test_ids, label_encoder, string_cols = load_and_preprocess_data()
    num_classes = len(label_encoder.classes_)
    n_train = len(X_train)
    n_test = len(X_test)

    # Stratified K-Fold setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Prediction matrices
    oof_lgb = np.zeros((n_train, num_classes))
    oof_cat = np.zeros((n_train, num_classes))
    oof_xgb = np.zeros((n_train, num_classes))
    oof_et = np.zeros((n_train, num_classes))
    oof_nn = np.zeros((n_train, num_classes))

    test_lgb = np.zeros((n_test, num_classes))
    test_cat = np.zeros((n_test, num_classes))
    test_xgb = np.zeros((n_test, num_classes))
    test_et = np.zeros((n_test, num_classes))
    test_nn = np.zeros((n_test, num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[train_idx].copy(), y[train_idx]
        X_va, y_va = X_train.iloc[val_idx].copy(), y[val_idx]
        X_te = X_test.copy()

        # Fold-level TF-IDF and SVD extraction to avoid data leakage
        tr_text = X_tr[string_cols].astype(str).fillna("").agg(" ".join, axis=1)
        va_text = X_va[string_cols].astype(str).fillna("").agg(" ".join, axis=1)
        te_text = X_te[string_cols].astype(str).fillna("").agg(" ".join, axis=1)

        tfidf = TfidfVectorizer(max_features=800, stop_words="english", ngram_range=(1, 2))
        tr_tfidf = tfidf.fit_transform(tr_text)
        va_tfidf = tfidf.transform(va_text)
        te_tfidf = tfidf.transform(te_text)

        svd = TruncatedSVD(n_components=8, random_state=42)
        tr_svd = svd.fit_transform(tr_tfidf)
        va_svd = svd.transform(va_tfidf)
        te_svd = svd.transform(te_tfidf)

        for i in range(8):
            X_tr[f"text_svd_{i}"] = tr_svd[:, i]
            X_va[f"text_svd_{i}"] = va_svd[:, i]
            X_te[f"text_svd_{i}"] = te_svd[:, i]

        # OOF Target Encoding for categorical variables
        smooth_weight = 10.0
        global_priors = np.bincount(y_tr, minlength=num_classes) / len(y_tr)
        for col in string_cols:
            for c in range(num_classes):
                y_tr_binary = (y_tr == c).astype(float)
                stats = pd.DataFrame({"cat": X_tr[col], "target": y_tr_binary}).groupby("cat")["target"].agg(["count", "mean"])
                smooth = (stats["count"] * stats["mean"] + smooth_weight * global_priors[c]) / (stats["count"] + smooth_weight)
                te_map = smooth.to_dict()
                te_col_name = f"{col}_te_class_{c}"
                X_tr[te_col_name] = X_tr[col].map(te_map).fillna(global_priors[c])
                X_va[te_col_name] = X_va[col].map(te_map).fillna(global_priors[c])
                X_te[te_col_name] = X_te[col].map(te_map).fillna(global_priors[c])

        # ------------------- Model 1: LightGBM -------------------
        model_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
        test_lgb += model_lgb.predict_proba(X_te) / n_splits

        # ------------------- Model 2: CatBoost -------------------
        model_cat = cb.CatBoostClassifier(
            loss_function="MultiClass",
            iterations=800,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=42 + fold,
            early_stopping_rounds=40,
            verbose=0,
        )
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_cat[val_idx] = model_cat.predict_proba(X_va)
        test_cat += model_cat.predict_proba(X_te) / n_splits

        # ------------------- Model 3: XGBoost -------------------
        model_xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=800,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=40,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
        test_xgb += model_xgb.predict_proba(X_te) / n_splits

        # ------------------- Model 4: ExtraTrees -------------------
        X_tr_et = X_tr.fillna(X_tr.mean())
        X_va_et = X_va.fillna(X_tr.mean())
        X_te_et = X_te.fillna(X_tr.mean())

        model_et = ExtraTreesClassifier(
            n_estimators=300, max_depth=20, random_state=42 + fold, n_jobs=-1
        )
        model_et.fit(X_tr_et, y_tr)
        oof_et[val_idx] = model_et.predict_proba(X_va_et)
        test_et += model_et.predict_proba(X_te_et) / n_splits

        # ------------------- Model 5: PyTorch TabularResNet -------------------
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_et)
        X_va_scaled = scaler.transform(X_va_et)
        X_te_scaled = scaler.transform(X_te_et)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        net = TabularResNet(
            in_features=X_tr_scaled.shape[1],
            num_classes=num_classes,
            hidden_dim=256,
            num_blocks=3,
            dropout_rate=0.2,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, 31):
            net.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    val_loss += criterion(net(bx), by).item() * len(by)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = net.state_dict().copy()

        net.load_state_dict(best_state)
        torch.save(best_state, f"./working/tabular_resnet_fold{fold}.pt")

        net.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            oof_nn[val_idx] = torch.softmax(net(va_tensor), dim=1).cpu().numpy()

            te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            test_nn += torch.softmax(net(te_tensor), dim=1).cpu().numpy() / n_splits

    # Ensemble SLSQP Constrained Weight Optimization
    oofs = [oof_lgb, oof_cat, oof_xgb, oof_nn, oof_et]
    tests = [test_lgb, test_cat, test_xgb, test_nn, test_et]
    num_models = len(oofs)

    def loss_func(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        blend_prob = sum(w * oof for w, oof in zip(weights, oofs))
        blend_prob = np.clip(blend_prob, 1e-15, 1 - 1e-15)
        one_hot = np.eye(num_classes)[y]
        return -np.mean(np.sum(one_hot * np.log(blend_prob), axis=1))

    init_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0)] * num_models
    constraints = ({"type": "eq", "fun": lambda w: 1.0 - np.sum(w)})

    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)

    oof_blend = sum(w * oof for w, oof in zip(best_weights, oofs))
    oof_preds = np.argmax(oof_blend, axis=1)
    val_acc = accuracy_score(y, oof_preds)

    # Test Prediction Inference
    test_blend = sum(w * test for w, test in zip(best_weights, tests))
    test_preds = np.argmax(test_blend, axis=1)
    test_labels = label_encoder.inverse_transform(test_preds)

    # Export Submission
    submission_df = pd.DataFrame({"id": test_ids, "VOLT_CLASS": test_labels})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {val_acc:.6f}")


if __name__ == "__main__":
    main()