import copy
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


# ==========================================
# 1. Data Processing & Feature Engineering
# ==========================================
def load_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def process_target(train_df):
    target_col = "Class"
    if target_col in train_df.columns:
        unique_vals = train_df[target_col].unique()
        mapping = {}
        for val in unique_vals:
            val_str = str(val).strip().upper()
            if val_str in ["RB", "1", "TRUE", "1.0"]:
                mapping[val] = 1
            elif val_str in ["NRB", "0", "2", "FALSE", "0.0", "2.0"]:
                mapping[val] = 0
            else:
                mapping[val] = 1 if "RB" in val_str or val == 1 else 0
        train_df["target"] = train_df[target_col].map(mapping).astype(int)
    return train_df


def engineer_features(df):
    df_feat = df.copy()

    # 1. Heteroatom Ratios & Counts
    nO = df_feat["nO"] if "nO" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nN = df_feat["nN"] if "nN" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nX = df_feat["nX"] if "nX" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nHM = df_feat["nHM"] if "nHM" in df_feat.columns else pd.Series(1, index=df_feat.index)

    df_feat["total_heteroatoms"] = nO + nN + nX
    df_feat["heteroatom_ratio"] = df_feat["total_heteroatoms"] / (nHM + 1e-5)

    # Combined halogen indicators
    nF = df_feat["nF"] if "nF" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nCl = df_feat["nCl"] if "nCl" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nBr = df_feat["nBr"] if "nBr" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nI = df_feat["nI"] if "nI" in df_feat.columns else pd.Series(0, index=df_feat.index)
    df_feat["combined_halogens"] = nF + nCl + nBr + nI + nX
    df_feat["has_halogens"] = (df_feat["combined_halogens"] > 0).astype(float)

    # Carbon subtype ratios relative to nHM
    nCb = df_feat["nCb-"] if "nCb-" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nCp = df_feat["nCp"] if "nCp" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nCrt = df_feat["nCrt"] if "nCrt" in df_feat.columns else pd.Series(0, index=df_feat.index)
    df_feat["ratio_nCb_nHM"] = nCb / (nHM + 1e-5)
    df_feat["ratio_nCp_nHM"] = nCp / (nHM + 1e-5)
    df_feat["ratio_nCrt_nHM"] = nCrt / (nHM + 1e-5)

    # Cross-matrix spectral interaction ratios
    if "SpMax_L" in df_feat.columns and "SpMax_A" in df_feat.columns:
        df_feat["ratio_SpMax_L_A"] = df_feat["SpMax_L"] / (df_feat["SpMax_A"] + 1e-5)
    if "SpMax_L" in df_feat.columns and "SpMax_B(m)" in df_feat.columns:
        df_feat["ratio_SpMax_L_Bm"] = df_feat["SpMax_L"] / (df_feat["SpMax_B(m)"] + 1e-5)
    if "SM6_L" in df_feat.columns and "SM6_B(m)" in df_feat.columns:
        df_feat["ratio_SM6_L_Bm"] = df_feat["SM6_L"] / (df_feat["SM6_B(m)"] + 1e-5)

    # 2. Nitrogen Functional Group Aggregates
    nN_N = df_feat["nN-N"] if "nN-N" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nArNO2 = df_feat["nArNO2"] if "nArNO2" in df_feat.columns else pd.Series(0, index=df_feat.index)
    f01_nn = df_feat["F01[N-N]"] if "F01[N-N]" in df_feat.columns else pd.Series(0, index=df_feat.index)
    f02_cn = df_feat["F02[C-N]"] if "F02[C-N]" in df_feat.columns else pd.Series(0, index=df_feat.index)
    f03_cn = df_feat["F03[C-N]"] if "F03[C-N]" in df_feat.columns else pd.Series(0, index=df_feat.index)
    f04_cn = df_feat["F04[C-N]"] if "F04[C-N]" in df_feat.columns else pd.Series(0, index=df_feat.index)

    df_feat["total_N_groups"] = nN + f01_nn + f02_cn + f03_cn + f04_cn + nN_N + nArNO2

    # 3. Oxygen Functional Group Aggregates
    f03_co = df_feat["F03[C-O]"] if "F03[C-O]" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nHDon = df_feat["nHDon"] if "nHDon" in df_feat.columns else pd.Series(0, index=df_feat.index)
    nArCOOR = df_feat["nArCOOR"] if "nArCOOR" in df_feat.columns else pd.Series(0, index=df_feat.index)

    df_feat["total_O_groups"] = nO + f03_co + nArCOOR + nHDon

    # 4. Electronic & Physicochemical Couplings
    if "Me" in df_feat.columns and "Mi" in df_feat.columns:
        df_feat["electroneg_ioniz_prod"] = df_feat["Me"] * df_feat["Mi"]
        df_feat["electroneg_ioniz_ratio"] = df_feat["Me"] / (df_feat["Mi"] + 1e-5)

    if "J_Dz(e)" in df_feat.columns and "Me" in df_feat.columns:
        df_feat["barysz_electroneg_interaction"] = df_feat["J_Dz(e)"] * df_feat["Me"]

    if "SdssC" in df_feat.columns and "SdO" in df_feat.columns:
        df_feat["estate_sum_total"] = df_feat["SdssC"] + df_feat["SdO"]

    # 5. Topological Complexity & Density Metrics
    if "SM6_L" in df_feat.columns and "SpMax_L" in df_feat.columns:
        df_feat["spectral_moment_ratio_L"] = df_feat["SM6_L"] / (
            df_feat["SpMax_L"] + 1e-5
        )

    if "SM6_B(m)" in df_feat.columns and "SpMax_B(m)" in df_feat.columns:
        df_feat["spectral_moment_ratio_B"] = df_feat["SM6_B(m)"] / (
            df_feat["SpMax_B(m)"] + 1e-5
        )

    if "Psi_i_1d" in df_feat.columns and "Psi_i_A" in df_feat.columns:
        df_feat["pseudoconnectivity_prod"] = df_feat["Psi_i_1d"] * df_feat["Psi_i_A"]
        df_feat["pseudoconnectivity_ratio"] = df_feat["Psi_i_1d"] / (
            df_feat["Psi_i_A"] + 1e-5
        )

    if "nCIR" in df_feat.columns:
        df_feat["circuit_density"] = df_feat["nCIR"] / (nHM + 1e-5)
        if "LOC" in df_feat.columns:
            df_feat["centric_circuit_ratio"] = df_feat["LOC"] / (df_feat["nCIR"] + 1.0)

    # 6. Log-Transform for High-Skew Spectral Features
    skew_cols = [
        "SpMax_L",
        "SM6_L",
        "TI2_L",
        "SpMax_A",
        "SpMax_B(m)",
        "SM6_B(m)",
        "J_Dz(e)",
        "HyWi_B(m)",
    ]
    for col in skew_cols:
        if col in df_feat.columns:
            df_feat[f"{col}_log"] = np.log1p(np.abs(df_feat[col])) * np.sign(
                df_feat[col]
            )

    # 7. Descriptor Aggregate Statistics
    numeric_cols = [
        c for c in df_feat.columns if c not in ["id", "Class", "target", "fold"]
    ]
    df_feat["desc_mean"] = df_feat[numeric_cols].mean(axis=1)
    df_feat["desc_std"] = df_feat[numeric_cols].std(axis=1)
    df_feat["desc_sum"] = df_feat[numeric_cols].sum(axis=1)

    return df_feat


def add_unsupervised_features(train_df, test_df, seed=42):
    train_feat = train_df.copy()
    test_feat = test_df.copy()

    ignore_cols = ["id", "Class", "target", "fold"]
    numeric_cols = [c for c in train_feat.columns if c not in ignore_cols]

    X_tr_num = (
        train_feat[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values
    )
    X_te_num = (
        test_feat[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values
    )

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_num)
    X_te_scaled = scaler.transform(X_te_num)

    # PCA Component Projections
    n_pca = 5
    pca = PCA(n_components=n_pca, random_state=seed)
    pca_tr = pca.fit_transform(X_tr_scaled)
    pca_te = pca.transform(X_te_scaled)

    for i in range(n_pca):
        train_feat[f"pca_{i}"] = pca_tr[:, i]
        test_feat[f"pca_{i}"] = pca_te[:, i]

    # Unsupervised Chemical Clustering & Centroid Distances
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_tr_scaled)

    km_tr_dist = kmeans.transform(X_tr_scaled)
    km_te_dist = kmeans.transform(X_te_scaled)

    for i in range(n_clusters):
        train_feat[f"kmeans_dist_{i}"] = km_tr_dist[:, i]
        test_feat[f"kmeans_dist_{i}"] = km_te_dist[:, i]

    return train_feat, test_feat


# ==========================================
# 2. PyTorch Neural Architecture & Models
# ==========================================
class FeatureGatingSE(nn.Module):
    """Squeeze-and-Excitation block for tabular feature importance weighting."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class TabularAttentionNet(nn.Module):
    """
    Modern Tabular Neural Network with Feature Projections, Self-Attention,
    Squeeze-and-Excitation Feature Gating, and Residual Skip Connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        self.se_gate = FeatureGatingSE(hidden_dim, reduction=4)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h = self.norm_attn(h + attn_out.squeeze(1))
        h_gated = self.se_gate(h)
        h = h + h_gated
        h_ffn = self.ffn(h)
        h = self.norm_ffn(h + h_ffn)
        logits = self.head(h)
        return logits.squeeze(-1)


def get_neural_components(input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4):
    model = TabularAttentionNet(
        input_dim=input_dim, hidden_dim=128, num_heads=4, dropout=0.25
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


def get_gbdt_models(seed: int = 42):
    models = {
        "lgb": lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            thread_count=-1,
        )

    return models


# ==========================================
# 3. Training, Evaluation & Ensembling
# ==========================================
def rank_normalize(probs: np.ndarray) -> np.ndarray:
    return rankdata(probs) / len(probs)


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load Data and Process Features
    train_df, test_df = load_data()
    train_df = process_target(train_df)

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    # 2. Stratified 10-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    train_feat["fold"] = -1
    for fold, (tr_idx, val_idx) in enumerate(
        skf.split(train_feat, train_feat["target"])
    ):
        train_feat.loc[val_idx, "fold"] = fold

    ignore_cols = ["id", "Class", "target", "fold"]
    feature_cols = [c for c in train_feat.columns if c not in ignore_cols]
    n_features = len(feature_cols)

    num_folds = 10
    y_true_all = train_feat["target"].values
    n_train = len(train_feat)
    n_test = len(test_feat)

    gbdt_names = ["lgb", "xgb", "hist_gb", "extra_trees"]
    if HAS_CATBOOST:
        gbdt_names.append("catboost")

    all_model_names = gbdt_names + ["tabular_attention_net"]

    oof_preds_dict = {
        name: np.zeros(n_train, dtype=np.float32) for name in all_model_names
    }
    test_preds_dict = {
        name: np.zeros(n_test, dtype=np.float32) for name in all_model_names
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Stratified K-Fold Cross Validation Training Loop
    for fold in range(num_folds):
        train_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        train_sub = train_feat.loc[train_idx].copy()
        val_sub = train_feat.loc[val_idx].copy()
        test_sub = test_feat.copy()

        feature_medians = train_sub[feature_cols].median()

        X_tr = (
            train_sub[feature_cols]
            .fillna(feature_medians)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
        )
        y_tr = train_sub["target"].values

        X_va = (
            val_sub[feature_cols]
            .fillna(feature_medians)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
        )
        y_va = val_sub["target"].values

        X_te = (
            test_sub[feature_cols]
            .fillna(feature_medians)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
        )

        # Compute unsupervised features per fold to prevent leakage
        scaler_unsup = StandardScaler()
        X_tr_unsup = scaler_unsup.fit_transform(X_tr)
        X_va_unsup = scaler_unsup.transform(X_va)
        X_te_unsup = scaler_unsup.transform(X_te)

        pca = PCA(n_components=5, random_state=seed + fold)
        pca_tr = pca.fit_transform(X_tr_unsup)
        pca_va = pca.transform(X_va_unsup)
        pca_te = pca.transform(X_te_unsup)

        kmeans = KMeans(n_clusters=5, random_state=seed + fold, n_init=10)
        kmeans.fit(X_tr_unsup)
        km_tr_dist = kmeans.transform(X_tr_unsup)
        km_va_dist = kmeans.transform(X_va_unsup)
        km_te_dist = kmeans.transform(X_te_unsup)

        X_tr = np.column_stack([X_tr, pca_tr, km_tr_dist])
        X_va = np.column_stack([X_va, pca_va, km_va_dist])
        X_te = np.column_stack([X_te, pca_te, km_te_dist])

        # Train GBDT & Tree Ensemble Models
        fold_models = get_gbdt_models(seed=seed + fold)

        for name, model in fold_models.items():
            if name == "lgb":
                callbacks = [lgb.early_stopping(50, verbose=False)]
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=callbacks,
                )
            elif name == "xgb":
                model.set_params(early_stopping_rounds=50)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif name == "catboost":
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
            else:
                model.fit(X_tr, y_tr)

            val_p = model.predict_proba(X_va)[:, 1]
            test_p = model.predict_proba(X_te)[:, 1]

            oof_preds_dict[name][val_idx] = val_p
            test_preds_dict[name] += test_p / num_folds

        # Train PyTorch TabularAttentionNet
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_te)

        nn_model, criterion, optimizer, scheduler = get_neural_components(
            input_dim=X_tr.shape[1], lr=1e-3, weight_decay=1e-4
        )
        nn_model.to(device)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )

        best_val_auc = 0.0
        best_weights = None
        patience = 15
        patience_counter = 0

        for epoch in range(1, 81):
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
            with torch.no_grad():
                val_x_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
                val_logits = nn_model(val_x_t)
                val_p_nn = torch.sigmoid(val_logits).cpu().numpy()
                val_auc_nn = roc_auc_score(y_va, val_p_nn)

            if val_auc_nn > best_val_auc:
                best_val_auc = val_auc_nn
                best_weights = copy.deepcopy(nn_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            nn_model.load_state_dict(best_weights)

        nn_model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            nn_val_p = torch.sigmoid(nn_model(val_x_t)).cpu().numpy()

            test_x_t = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            nn_test_p = torch.sigmoid(nn_model(test_x_t)).cpu().numpy()

        oof_preds_dict["tabular_attention_net"][val_idx] = nn_val_p
        test_preds_dict["tabular_attention_net"] += nn_test_p / num_folds

    # 4. 2-Level Stacking Meta-Learner
    X_oof_meta = np.column_stack(
        [rank_normalize(oof_preds_dict[name]) for name in all_model_names]
    )
    X_test_meta = np.column_stack(
        [rank_normalize(test_preds_dict[name]) for name in all_model_names]
    )

    meta_scaler = StandardScaler()
    X_oof_meta_scaled = meta_scaler.fit_transform(X_oof_meta)
    X_test_meta_scaled = meta_scaler.transform(X_test_meta)

    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
    meta_model.fit(X_oof_meta_scaled, y_true_all)

    blended_oof = meta_model.predict_proba(X_oof_meta_scaled)[:, 1]
    blended_test = meta_model.predict_proba(X_test_meta_scaled)[:, 1]

    final_score = roc_auc_score(y_true_all, blended_oof)

    # 5. Output Submission & Print Metric
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame({"id": test_feat["id"].values, "Class": blended_test})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()