import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def standardize_columns(df):
    col_map = {
        "V1": "Age",
        "V2": "Gender",
        "V3": "Total_Bilirubin",
        "V4": "Direct_Bilirubin",
        "V5": "Alkphos",
        "V6": "Sgpt",
        "V7": "Sgot",
        "V8": "Total_Proteins",
        "V9": "Albumin",
        "V10": "AG_Ratio",
    }
    return df.rename(columns=col_map)


def create_clinical_features(df):
    data = df.copy()

    if "Gender" in data.columns:
        gender_str = data["Gender"].astype(str).str.upper()
        data["Gender_Numeric"] = (
            gender_str.map({"MALE": 1, "M": 1, "1": 1, "FEMALE": 0, "F": 0, "0": 0})
            .fillna(1)
            .astype(int)
        )
        data = data.drop(columns=["Gender"])

    if "AG_Ratio" in data.columns:
        data["AG_Ratio_Missing"] = data["AG_Ratio"].isna().astype(int)

    # Derived Metabolic Biomarkers
    data["Calculated_Globulin"] = data["Total_Proteins"] - data["Albumin"]
    globulin_safe = np.where(
        data["Calculated_Globulin"] <= 0, 0.01, data["Calculated_Globulin"]
    )
    data["Calculated_AG_Ratio"] = data["Albumin"] / globulin_safe

    if "AG_Ratio" in data.columns:
        data["AG_Ratio_Clean"] = data["AG_Ratio"].fillna(data["Calculated_AG_Ratio"])
        data["AG_Ratio_Discrepancy"] = (
            data["AG_Ratio_Clean"] - data["Calculated_AG_Ratio"]
        ).abs()
    else:
        data["AG_Ratio_Clean"] = data["Calculated_AG_Ratio"]
        data["AG_Ratio_Discrepancy"] = 0.0

    # De Ritis Ratio (AST / ALT = Sgot / Sgpt)
    sgpt_safe = np.where(data["Sgpt"] <= 0, 1.0, data["Sgpt"])
    data["De_Ritis_Ratio"] = data["Sgot"] / sgpt_safe
    data["De_Ritis_Ratio_Squared"] = data["De_Ritis_Ratio"] ** 2

    # Conjugated vs Unconjugated Bilirubin
    tot_prot_safe = np.where(
        data["Total_Proteins"] <= 0, 0.1, data["Total_Proteins"]
    )
    data["Albumin_Total_Protein_Ratio"] = data["Albumin"] / tot_prot_safe
    tot_bil_safe = np.where(data["Total_Bilirubin"] <= 0, 0.1, data["Total_Bilirubin"])
    data["Direct_Total_Bilirubin_Ratio"] = data["Direct_Bilirubin"] / tot_bil_safe
    data["Indirect_Bilirubin"] = np.maximum(
        0.0, data["Total_Bilirubin"] - data["Direct_Bilirubin"]
    )

    # Enzyme Loads
    data["Total_Enzymes"] = data["Alkphos"] + data["Sgpt"] + data["Sgot"]
    data["Transaminase_Sum"] = data["Sgpt"] + data["Sgot"]
    trans_sum_safe = np.where(
        data["Transaminase_Sum"] <= 0, 1.0, data["Transaminase_Sum"]
    )
    data["Transaminase_Diff_Norm"] = (data["Sgot"] - data["Sgpt"]) / trans_sum_safe

    alkphos_safe = np.where(data["Alkphos"] <= 0, 1.0, data["Alkphos"])
    data["Bilirubin_Alkphos_Ratio"] = data["Total_Bilirubin"] / alkphos_safe
    data["Albumin_Bilirubin_Ratio"] = data["Albumin"] / tot_bil_safe

    # Log Transformations for Skewed Distributions
    for col in [
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkphos",
        "Sgpt",
        "Sgot",
        "De_Ritis_Ratio",
    ]:
        data[f"log_{col}"] = np.log1p(np.maximum(0.0, data[col]))

    # Clinical Upper Limits
    data["high_total_bilirubin"] = (data["Total_Bilirubin"] > 1.2).astype(int)
    data["high_direct_bilirubin"] = (data["Direct_Bilirubin"] > 0.3).astype(int)
    data["high_alkphos"] = (data["Alkphos"] > 290.0).astype(int)
    data["high_sgpt"] = (data["Sgpt"] > 40.0).astype(int)
    data["high_sgot"] = (data["Sgot"] > 40.0).astype(int)
    data["low_albumin"] = (data["Albumin"] < 3.5).astype(int)
    data["low_ag_ratio"] = (data["AG_Ratio_Clean"] < 0.9).astype(int)
    data["abnormal_de_ritis"] = (
        (data["De_Ritis_Ratio"] > 2.0) | (data["De_Ritis_Ratio"] < 0.8)
    ).astype(int)

    data["abnormality_count"] = (
        data["high_total_bilirubin"]
        + data["high_direct_bilirubin"]
        + data["high_alkphos"]
        + data["high_sgpt"]
        + data["high_sgot"]
        + data["low_albumin"]
        + data["low_ag_ratio"]
    )

    # Demographic & Non-linear Interactions
    data["Age_x_De_Ritis"] = data["Age"] * data["De_Ritis_Ratio"]
    data["Age_x_Total_Bilirubin"] = data["Age"] * data["Total_Bilirubin"]
    data["Age_x_Sgot"] = data["Age"] * data["Sgot"]
    if "Gender_Numeric" in data.columns:
        data["Gender_x_Sgpt"] = data["Gender_Numeric"] * data["Sgpt"]

    data["Bilirubin_Product"] = data["Total_Bilirubin"] * data["Direct_Bilirubin"]
    data["Protein_Product"] = data["Total_Proteins"] * data["Albumin"]
    data["Age_Group"] = pd.cut(
        data["Age"], bins=[-1, 30, 45, 60, 120], labels=[0, 1, 2, 3]
    ).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def add_subgroup_relative_features(X_tr, X_va, X_te):
    biomarkers = ["Total_Bilirubin", "Sgot", "Sgpt", "Alkphos"]
    group_cols = ["Gender_Numeric", "Age_Group"]

    for bm in biomarkers:
        global_med = X_tr[bm].median()
        group_medians = X_tr.groupby(group_cols)[bm].median().to_dict()

        for df_curr in [X_tr, X_va, X_te]:
            sub_keys = list(zip(df_curr["Gender_Numeric"], df_curr["Age_Group"]))
            med_vals = np.array([group_medians.get(k, global_med) for k in sub_keys])
            df_curr[f"{bm}_subgroup_diff"] = df_curr[bm] - med_vals
            df_curr[f"{bm}_subgroup_ratio"] = df_curr[bm] / np.where(
                med_vals == 0, 1.0, med_vals
            )

    return X_tr, X_va, X_te


# Neural Network Architecture Definitions
class TabularFeatureAttention(nn.Module):

    def __init__(self, in_features, reduction=4):
        super(TabularFeatureAttention, self).__init__()
        reduced_dim = max(4, in_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_features, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResNetBlock(nn.Module):

    def __init__(self, dim, dropout_rate=0.25):
        super(ResNetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.attn = TabularFeatureAttention(dim)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.attn(out)
        return residual + out


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.25):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.3, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Main Execution Pipeline
if __name__ == "__main__":
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    train_df = standardize_columns(train_df)
    test_df = standardize_columns(test_df)

    target_col = "Class"
    if target_col in train_df.columns:
        y_raw = train_df[target_col].copy()
        train_df = train_df.drop(columns=[target_col])
        if y_raw.dtype == object:
            y_str = y_raw.astype(str).str.strip()
            y = (
                y_str.map(
                    {"1": 1, "2": 0, "1.0": 1, "2.0": 0, "Patient": 1, "Healthy": 0}
                )
                .fillna(1)
                .astype(int)
            )
        else:
            unique_vals = sorted(y_raw.dropna().unique())
            if set(unique_vals) == {1, 2}:
                y = y_raw.map({1: 1, 2: 0}).astype(int)
            elif set(unique_vals) == {0, 1}:
                y = y_raw.astype(int)
            else:
                y = (y_raw == unique_vals[0]).astype(int)

    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )
    for c in ["id"]:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in test_df.columns:
            test_df = test_df.drop(columns=[c])

    # Feature Engineering
    X_train_full = create_clinical_features(train_df)
    X_test_full = create_clinical_features(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    num_train = len(X_train_full)
    num_test = len(X_test_full)

    # Meta-learner input containers: 7 base models
    oof_meta_features = np.zeros((num_train, 7))
    test_meta_features = np.zeros((num_test, 7))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
        X_tr, y_tr = X_train_full.iloc[train_idx].copy(), y.iloc[train_idx].values
        X_va, y_va = X_train_full.iloc[val_idx].copy(), y.iloc[val_idx].values
        X_te = X_test_full.copy()

        # Compute clinical relative deviations strictly per fold
        X_tr, X_va, X_te = add_subgroup_relative_features(X_tr, X_va, X_te)

        # Leak-free per-fold median imputation
        col_medians = X_tr.median()
        X_tr = X_tr.fillna(col_medians)
        X_va = X_va.fillna(col_medians)
        X_te = X_te.fillna(col_medians)

        # ------------------ Model 1: Tabular ResNet (PyTorch) ------------------
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_sc, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        )
        tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)

        nn_model = TabularResNet(
            input_dim=X_tr.shape[1], hidden_dim=128, num_blocks=3, dropout_rate=0.25
        ).to(device)
        criterion = FocalLoss(alpha=0.3, gamma=2.0)
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=80, eta_min=1e-5
        )

        best_val_auc = 0.0
        best_nn_val_preds = None
        best_nn_test_preds = None

        for epoch in range(80):
            nn_model.train()
            for batch_x, batch_y in tr_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_x_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
                val_logits = nn_model(val_x_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                val_auc = roc_auc_score(y_va, val_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_nn_val_preds = val_probs
                    test_x_t = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
                    test_logits = nn_model(test_x_t)
                    best_nn_test_preds = (
                        torch.sigmoid(test_logits).cpu().numpy().ravel()
                    )

        oof_meta_features[val_idx, 0] = best_nn_val_preds
        test_meta_features[:, 0] += best_nn_test_preds / n_splits

        # ------------------ Model 2: CatBoost ------------------
        cb_model = cb.CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            random_seed=SEED + fold,
            eval_metric="AUC",
            verbose=False,
            early_stopping_rounds=30,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        p_cb_val = cb_model.predict_proba(X_va)[:, 1]
        p_cb_test = cb_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 1] = p_cb_val
        test_meta_features[:, 1] += p_cb_test / n_splits

        # ------------------ Model 3: LightGBM ------------------
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=12,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=SEED + fold,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        p_lgb_val = lgb_model.predict_proba(X_va)[:, 1]
        p_lgb_test = lgb_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 2] = p_lgb_val
        test_meta_features[:, 2] += p_lgb_test / n_splits

        # ------------------ Model 4: XGBoost ------------------
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric="auc",
            random_state=SEED + fold,
            early_stopping_rounds=30,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb_val = xgb_model.predict_proba(X_va)[:, 1]
        p_xgb_test = xgb_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 3] = p_xgb_val
        test_meta_features[:, 3] += p_xgb_test / n_splits

        # ------------------ Model 5: Random Forest ------------------
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        rf_model.fit(X_tr, y_tr)
        p_rf_val = rf_model.predict_proba(X_va)[:, 1]
        p_rf_test = rf_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 4] = p_rf_val
        test_meta_features[:, 4] += p_rf_test / n_splits

        # ------------------ Model 6: Extra Trees ------------------
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        p_et_val = et_model.predict_proba(X_va)[:, 1]
        p_et_test = et_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 5] = p_et_val
        test_meta_features[:, 5] += p_et_test / n_splits

        # ------------------ Model 7: HistGradientBoosting ------------------
        hgb_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            random_state=SEED + fold,
        )
        hgb_model.fit(X_tr, y_tr)
        p_hgb_val = hgb_model.predict_proba(X_va)[:, 1]
        p_hgb_test = hgb_model.predict_proba(X_te)[:, 1]

        oof_meta_features[val_idx, 6] = p_hgb_val
        test_meta_features[:, 6] += p_hgb_test / n_splits

    # Rank Normalization of Meta-Features
    oof_meta_features_ranked = np.zeros_like(oof_meta_features)
    test_meta_features_ranked = np.zeros_like(test_meta_features)

    for col in range(oof_meta_features.shape[1]):
        oof_meta_features_ranked[:, col] = rankdata(oof_meta_features[:, col]) / len(
            oof_meta_features
        )
        test_meta_features_ranked[:, col] = rankdata(test_meta_features[:, col]) / len(
            test_meta_features
        )

    # Stacking Meta-Learner Training
    meta_learner = LogisticRegression(
        C=0.5, penalty="l2", solver="lbfgs", random_state=SEED
    )
    meta_learner.fit(oof_meta_features_ranked, y.values)

    oof_final_preds = meta_learner.predict_proba(oof_meta_features_ranked)[:, 1]
    test_final_preds = meta_learner.predict_proba(test_meta_features_ranked)[:, 1]

    # Validation Metric Calculation
    final_auc = roc_auc_score(y, oof_final_preds)

    # Export Test Submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame({"id": test_ids, "Class": test_final_preds})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_auc}")