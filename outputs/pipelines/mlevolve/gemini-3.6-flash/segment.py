import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# 1. Feature Engineering Helper Functions
# ---------------------------------------------------------------------------
def normalize_col_name(col: str) -> str:
    return col.lower().replace("-", "_").strip()


def find_column(df: pd.DataFrame, candidates: list) -> str:
    df_cols = {normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = normalize_col_name(cand)
        if cand_norm in df_cols:
            return df_cols[cand_norm]
    return None


def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers domain-derived physical texture and photometric features for 3x3 region segmentation."""
    df_feat = df.copy()

    col_red = find_column(df, ["rawred-mean", "rawred_mean", "rawred"])
    col_green = find_column(df, ["rawgreen-mean", "rawgreen_mean", "rawgreen"])
    col_blue = find_column(df, ["rawblue-mean", "rawblue_mean", "rawblue"])
    col_exred = find_column(df, ["exred-mean", "exred_mean"])
    col_exgreen = find_column(df, ["exgreen-mean", "exgreen_mean"])
    col_exblue = find_column(df, ["exblue-mean", "exblue_mean"])
    col_intensity = find_column(df, ["intensity-mean", "intensity_mean"])
    col_vedge_m = find_column(df, ["vedge-mean", "vedge_mean"])
    col_vedge_sd = find_column(df, ["vedge-sd", "vedge_sd"])
    col_hedge_m = find_column(df, ["hedge-mean", "hedge_mean"])
    col_hedge_sd = find_column(df, ["hedge-sd", "hedge_sd"])
    col_line_5 = find_column(df, ["short-line-density-5", "short_line_density_5"])
    col_line_2 = find_column(df, ["short-line-density-2", "short_line_density_2"])
    col_val = find_column(df, ["value-mean", "value_mean"])
    col_sat = find_column(df, ["saturation-mean", "saturation_mean", "saturatoin-mean"])
    col_hue = find_column(df, ["hue-mean", "hue_mean"])

    eps = 1e-6

    # Photometric Chromaticity & Ratios
    if col_red and col_green and col_blue:
        r = df_feat[col_red].astype(np.float32)
        g = df_feat[col_green].astype(np.float32)
        b = df_feat[col_blue].astype(np.float32)

        total_rgb = r + g + b + eps
        df_feat["feat_total_rgb"] = total_rgb
        df_feat["feat_r_norm"] = r / total_rgb
        df_feat["feat_g_norm"] = g / total_rgb
        df_feat["feat_b_norm"] = b / total_rgb

        df_feat["feat_r_g_ratio"] = r / (g + eps)
        df_feat["feat_b_g_ratio"] = b / (g + eps)
        df_feat["feat_r_b_ratio"] = r / (b + eps)

        df_feat["feat_r_g_diff"] = df_feat["feat_r_norm"] - df_feat["feat_g_norm"]
        df_feat["feat_r_b_diff"] = df_feat["feat_r_norm"] - df_feat["feat_b_norm"]
        df_feat["feat_g_b_diff"] = df_feat["feat_g_norm"] - df_feat["feat_b_norm"]
        df_feat["feat_r_dev"] = df_feat["feat_r_norm"] - (1.0 / 3.0)
        df_feat["feat_g_dev"] = df_feat["feat_g_norm"] - (1.0 / 3.0)
        df_feat["feat_b_dev"] = df_feat["feat_b_norm"] - (1.0 / 3.0)

        if col_intensity:
            rgb_avg = total_rgb / 3.0
            df_feat["feat_intensity_diff"] = (
                df_feat[col_intensity].astype(np.float32) - rgb_avg
            )

    # Excess Color Ratios
    if col_exred and col_exgreen and col_exblue and "feat_total_rgb" in df_feat:
        df_feat["feat_exg_norm"] = df_feat[col_exgreen] / (
            df_feat["feat_total_rgb"] + eps
        )
        df_feat["feat_exr_norm"] = df_feat[col_exred] / (
            df_feat["feat_total_rgb"] + eps
        )
        df_feat["feat_exb_norm"] = df_feat[col_exblue] / (
            df_feat["feat_total_rgb"] + eps
        )

    # Edge Contrast, Variability & Anisotropy
    if col_vedge_m and col_hedge_m:
        v_m = df_feat[col_vedge_m].astype(np.float32)
        h_m = df_feat[col_hedge_m].astype(np.float32)

        edge_sum = v_m + h_m + eps
        df_feat["feat_edge_sum"] = edge_sum
        df_feat["feat_edge_anisotropy"] = (v_m - h_m) / edge_sum
        df_feat["feat_edge_mag"] = np.sqrt(v_m**2 + h_m**2)

        if col_vedge_sd:
            v_sd = df_feat[col_vedge_sd].astype(np.float32)
            df_feat["feat_vedge_cv"] = v_sd / (v_m + eps)
        if col_hedge_sd:
            h_sd = df_feat[col_hedge_sd].astype(np.float32)
            df_feat["feat_hedge_cv"] = h_sd / (h_m + eps)

        if col_vedge_sd and col_hedge_sd:
            v_sd = df_feat[col_vedge_sd].astype(np.float32)
            h_sd = df_feat[col_hedge_sd].astype(np.float32)
            df_feat["feat_edge_total_sd"] = np.sqrt(v_sd**2 + h_sd**2)

    # Line Density Features
    if col_line_5 and col_line_2:
        l5 = df_feat[col_line_5].astype(np.float32)
        l2 = df_feat[col_line_2].astype(np.float32)
        df_feat["feat_total_line_density"] = l5 + l2
        df_feat["feat_line_contrast_ratio"] = (l2 + eps) / (l5 + eps)

    # HSV Non-linear Interactions & Trigonometric Hue
    if col_sat and col_val:
        s = df_feat[col_sat].astype(np.float32)
        v = df_feat[col_val].astype(np.float32)
        df_feat["feat_sat_val_prod"] = s * v
        df_feat["feat_sat_val_ratio"] = s / (v + eps)

    if col_hue:
        hue = df_feat[col_hue].astype(np.float32)
        df_feat["feat_hue_sin"] = np.sin(hue)
        df_feat["feat_hue_cos"] = np.cos(hue)

    return df_feat


# ---------------------------------------------------------------------------
# 2. PyTorch Tabular ResNet Architecture
# ---------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        return self.head(out)


# ---------------------------------------------------------------------------
# 3. Main End-to-End Pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Raw Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    target_col = find_column(train_raw, ["class", "target"])
    if target_col is None:
        raise ValueError("Target column 'class' not found in training dataset.")

    y_raw = train_raw[target_col]
    X_train_raw = train_raw.drop(columns=[target_col])
    X_test_raw = test_raw.copy()

    id_col = find_column(X_test_raw, ["id"])

    # 2. Target Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # 3. Clean Constant and Uninformative Spatial Columns
    constant_cols = [
        col for col in X_train_raw.columns if X_train_raw[col].nunique() <= 1
    ]
    ignored_patterns = ["centroid", "pixel-count", "pixel_count"]
    ignored_cols = [
        col
        for col in X_train_raw.columns
        if any(pat in normalize_col_name(col) for pat in ignored_patterns)
    ]
    cols_to_drop = list(set(constant_cols + ignored_cols))

    if id_col and id_col in cols_to_drop:
        cols_to_drop.remove(id_col)

    X_train_clean = X_train_raw.drop(
        columns=[c for c in cols_to_drop if c in X_train_raw.columns]
    )
    X_test_clean = X_test_raw.drop(
        columns=[c for c in cols_to_drop if c in X_test_raw.columns]
    )

    # 4. Extract Engineered Domain Features
    X_train_eng = extract_domain_features(X_train_clean)
    X_test_eng = extract_domain_features(X_test_clean)

    feature_cols = [c for c in X_train_eng.columns if c != id_col]
    num_classes = len(label_encoder.classes_)

    # 5. 5-Fold Stratified Cross-Validation Setup
    X_all = X_train_eng[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_all = X_test_eng[feature_cols].values.astype(np.float32)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros((len(X_all), num_classes))
    oof_xgb = np.zeros((len(X_all), num_classes))
    oof_cb = np.zeros((len(X_all), num_classes))
    oof_nn = np.zeros((len(X_all), num_classes))
    oof_et = np.zeros((len(X_all), num_classes))
    oof_svc = np.zeros((len(X_all), num_classes))

    test_lgb = np.zeros((len(X_test_all), num_classes))
    test_xgb = np.zeros((len(X_test_all), num_classes))
    test_cb = np.zeros((len(X_test_all), num_classes))
    test_nn = np.zeros((len(X_test_all), num_classes))
    test_et = np.zeros((len(X_test_all), num_classes))
    test_svc = np.zeros((len(X_test_all), num_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_encoded)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y_encoded[tr_idx], y_encoded[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_all)

        # LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)
        test_lgb += lgb_model.predict_proba(X_test_all) / 5.0

        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)
        test_xgb += xgb_model.predict_proba(X_test_all) / 5.0

        # CatBoost Classifier
        cb_model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        oof_cb[val_idx] = cb_model.predict_proba(X_val)
        test_cb += cb_model.predict_proba(X_test_all) / 5.0

        # PyTorch Tabular ResNet
        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

        nn_model = TabularResNet(
            input_dim=len(feature_cols),
            num_classes=num_classes,
            hidden_dim=128,
            num_blocks=3,
            dropout_rate=0.2,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        best_val_loss = float("inf")
        best_nn_weights = None

        for epoch in range(1, 101):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = nn_model(bx)
                    val_loss += criterion(logits, by).item() * bx.size(0)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = nn_model.state_dict().copy()

        if best_nn_weights is not None:
            nn_model.load_state_dict(best_nn_weights)

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(
                torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            )
            oof_nn[val_idx] = torch.softmax(val_logits, dim=1).cpu().numpy()

            test_logits = nn_model(test_tensor.to(device))
            test_nn += torch.softmax(test_logits, dim=1).cpu().numpy() / 5.0

        # ExtraTrees Classifier
        et_model = ExtraTreesClassifier(n_estimators=300, random_state=42)
        et_model.fit(X_tr_scaled, y_tr)
        oof_et[val_idx] = et_model.predict_proba(X_val_scaled)
        test_et += et_model.predict_proba(X_test_scaled) / 5.0

        # Support Vector Classifier
        svc_model = SVC(C=10.0, kernel="rbf", probability=True, random_state=42)
        svc_model.fit(X_tr_scaled, y_tr)
        oof_svc[val_idx] = svc_model.predict_proba(X_val_scaled)
        test_svc += svc_model.predict_proba(X_test_scaled) / 5.0

    # 6. Multi-Model Stacking Meta-Classifier
    oof_meta = np.hstack([oof_lgb, oof_xgb, oof_cb, oof_nn, oof_et, oof_svc])
    test_meta = np.hstack([test_lgb, test_xgb, test_cb, test_nn, test_et, test_svc])

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(oof_meta, y_encoded)

    oof_preds = meta_model.predict(oof_meta)
    final_oof_score = float(accuracy_score(y_encoded, oof_preds))

    # 7. Test Predictions & CSV Export
    test_preds_idx = meta_model.predict(test_meta)
    decoded_classes = label_encoder.inverse_transform(test_preds_idx)

    if id_col and id_col in X_test_raw.columns:
        test_ids = X_test_raw[id_col].values
    else:
        test_ids = np.arange(len(X_test_raw))

    submission_df = pd.DataFrame({"id": test_ids, "class": decoded_classes})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_oof_score}")


if __name__ == "__main__":
    main()
