import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Domain Feature Engineering
# ==========================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize ID column name if present
    if "ID" in df.columns and "id" not in df.columns:
        df["id"] = df["ID"]

    # --- 1. Age Features ---
    if "age_at_hct" in df.columns and "donor_age" in df.columns:
        df["age_diff"] = df["age_at_hct"] - df["donor_age"]
        df["abs_age_diff"] = np.abs(df["age_diff"])
        df["is_pediatric"] = (df["age_at_hct"] < 18).astype(float)
        df["is_older_donor"] = (df["donor_age"] > 40).astype(float)

    # --- 2. HLA Match Aggregations ---
    high_res_cols = [
        "hla_match_a_high",
        "hla_match_b_high",
        "hla_match_c_high",
        "hla_match_drb1_high",
        "hla_match_dqb1_high",
    ]
    low_res_cols = [
        "hla_match_a_low",
        "hla_match_b_low",
        "hla_match_c_low",
        "hla_match_drb1_low",
        "hla_match_dqb1_low",
    ]

    avail_high = [c for c in high_res_cols if c in df.columns]
    avail_low = [c for c in low_res_cols if c in df.columns]

    if avail_high:
        df["hla_high_res_sum"] = df[avail_high].sum(axis=1)
        df["hla_high_res_mismatch_count"] = (df[avail_high] < 2).sum(axis=1)

    if avail_low:
        df["hla_low_res_sum"] = df[avail_low].sum(axis=1)
        df["hla_low_res_mismatch_count"] = (df[avail_low] < 2).sum(axis=1)

    if "hla_high_res_10" in df.columns:
        df["hla_high_res_10_ratio"] = df["hla_high_res_10"] / 10.0
    if "hla_high_res_8" in df.columns:
        df["hla_high_res_8_ratio"] = df["hla_high_res_8"] / 8.0

    # --- 3. Comorbidities & Performance Status ---
    comorbid_cols = [
        "arrhythmia",
        "cardiac",
        "diabetes",
        "hepatic_mild",
        "hepatic_severe",
        "obesity",
        "peptic_ulcer",
        "psych_disturb",
        "pulm_moderate",
        "pulm_severe",
        "renal_issue",
        "rheum_issue",
        "prior_tumor",
        "vent_hist",
    ]
    avail_comorbid = [c for c in comorbid_cols if c in df.columns]
    if avail_comorbid:
        df["comorbidity_yes_count"] = (
            df[avail_comorbid].apply(lambda x: x == "Yes").sum(axis=1)
        )

    if "karnofsky_score" in df.columns and "comorbidity_score" in df.columns:
        df["karnofsky_comorbid_interaction"] = df["comorbidity_score"] * (
            100.0 - df["karnofsky_score"].fillna(80.0)
        )

    # --- 4. High-Risk Sex Mismatch Flag ---
    if "sex_match" in df.columns:
        df["is_female_donor_male_recipient"] = (df["sex_match"] == "F-M").astype(float)

    # --- 5. Ordinal Disease Risk Index Mapping ---
    dri_map = {
        "Low": 1,
        "Intermediate": 2,
        "High": 3,
        "Very High": 4,
        "N/A - pediatric": 2,
        "N/A - non-malignant indication": 1,
        "TBD": 2,
    }
    if "dri_score" in df.columns:
        df["dri_score_ordinal"] = df["dri_score"].map(dri_map).fillna(2).astype(float)

    cyto_map = {
        "Favorable": 1,
        "Normal": 2,
        "Intermediate": 3,
        "Poor": 4,
        "Other": 2,
        "Not tested": 2,
        "TBD": 2,
    }
    if "cyto_score" in df.columns:
        df["cyto_score_ordinal"] = (
            df["cyto_score"].map(cyto_map).fillna(2).astype(float)
        )

    # --- 6. Temporal Features ---
    if "year_hct" in df.columns:
        df["year_hct_normalized"] = df["year_hct"] - 2008

    return df


def process_and_prepare_data():
    os.makedirs("./working", exist_ok=True)

    # Load raw data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    # Extract target and time variables from train
    target_col = "efs"
    time_col = "efs_time"

    y_train = train_raw[target_col].values if target_col in train_raw else None
    train_time = train_raw[time_col].values if time_col in train_raw else None

    # Apply domain feature engineering
    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)

    drop_cols = [
        col
        for col in [
            "id",
            "ID",
            "efs",
            "efs_time",
            "efs_time_years",
            "race_group",
        ]
        if col in train_fe.columns
    ]

    feature_cols = [c for c in train_fe.columns if c not in drop_cols]

    cat_cols = [
        c
        for c in feature_cols
        if train_fe[c].dtype == "object" or train_fe[c].dtype.name == "category"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train_processed = train_fe.copy()
    test_processed = test_fe.copy()

    for c in cat_cols:
        train_processed[c] = train_processed[c].fillna("Missing").astype(str)
        test_processed[c] = test_processed[c].fillna("Missing").astype(str)

        freq = train_processed[c].value_counts(normalize=True).to_dict()
        train_processed[f"{c}_freq"] = train_processed[c].map(freq).fillna(0)
        test_processed[f"{c}_freq"] = (
            test_processed[c].map(freq).fillna(0).astype(float)
        )

        categories = list(
            set(train_processed[c].unique()).union(set(test_processed[c].unique()))
        )
        cat_to_code = {cat: i for i, cat in enumerate(categories)}
        train_processed[c] = train_processed[c].map(cat_to_code).astype(int)
        test_processed[c] = test_processed[c].map(cat_to_code).astype(int)

    for c in num_cols:
        if train_processed[c].isna().sum() > 0 or test_processed[c].isna().sum() > 0:
            train_processed[f"{c}_isna"] = train_processed[c].isna().astype(float)
            test_processed[f"{c}_isna"] = test_processed[c].isna().astype(float)

        median_val = train_processed[c].median()
        train_processed[c] = train_processed[c].fillna(median_val)
        test_processed[c] = test_processed[c].fillna(median_val)

    final_feature_cols = [
        c
        for c in train_processed.columns
        if c not in drop_cols and c not in ["efs", "efs_time"]
    ]

    train_processed["efs"] = y_train
    if train_time is not None:
        train_processed["efs_time"] = train_time
    if "race_group" in train_raw.columns:
        train_processed["race_group"] = train_raw["race_group"]
        test_processed["race_group"] = test_raw["race_group"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(train_processed), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, y_train)):
        fold_assignments[val_idx] = fold

    train_processed["fold"] = fold_assignments

    train_processed.to_parquet("./working/train_processed.parquet", index=False)
    test_processed.to_parquet("./working/test_processed.parquet", index=False)

    metadata = {
        "feature_cols": final_feature_cols,
        "cat_cols": cat_cols,
        "num_cols": [c for c in final_feature_cols if c not in cat_cols],
        "n_train": len(train_processed),
        "n_test": len(test_processed),
    }

    with open("./working/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return train_processed, test_processed, metadata


# ==========================================
# 2. PyTorch Tabular ResNet Architecture
# ==========================================
class ResidualBlock(nn.Module):
    """Residual block for tabular risk representation learning."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        return out + residual


class TabularResNet(nn.Module):
    """Deep Tabular ResNet for predicting patient hazard/survival risk score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        risk_score = self.head(out)
        return risk_score.squeeze(-1)


class SmoothConcordanceLoss(nn.Module):
    """Pairwise smooth surrogate loss directly optimizing for Concordance Index."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, pred_risk: torch.Tensor, target_efs: torch.Tensor
    ) -> torch.Tensor:
        diff = pred_risk.unsqueeze(1) - pred_risk.unsqueeze(0)
        target_diff = target_efs.unsqueeze(1) - target_efs.unsqueeze(0)
        valid_pairs = (target_diff < 0).float()

        if valid_pairs.sum() == 0:
            return torch.tensor(0.0, device=pred_risk.device, requires_grad=True)

        loss_matrix = F.binary_cross_entropy_with_logits(
            diff * self.temperature, torch.ones_like(diff)
        )
        loss = (loss_matrix * valid_pairs).sum() / (valid_pairs.sum() + 1e-8)
        return loss


def calculate_concordance_error(y_true: np.ndarray, y_pred_risk: np.ndarray) -> float:
    """Calculate official metric: 1.0 - Concordance Index (lower is better)."""
    n = len(y_true)
    if n == 0:
        return 1.0

    efs_0_idx = np.where(y_true == 0)[0]
    efs_1_idx = np.where(y_true == 1)[0]

    if len(efs_0_idx) == 0 or len(efs_1_idx) == 0:
        return 0.5

    risk_0 = y_pred_risk[efs_0_idx]
    risk_1 = y_pred_risk[efs_1_idx]

    diffs = risk_0[:, None] - risk_1[None, :]
    concordant = np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
    total_pairs = len(efs_0_idx) * len(efs_1_idx)

    c_index = concordant / total_pairs
    return float(1.0 - c_index)


# ==========================================
# 3. Main Training & Evaluation Loop
# ==========================================
def main():
    os.makedirs("./submission", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets
    train_df, test_df, metadata = process_and_prepare_data()

    feature_cols = metadata["feature_cols"]
    y_true = train_df["efs"].values.astype(np.float32)
    y_event = 1.0 - y_true  # Event indicator
    folds = train_df["fold"].values

    X_train_full = train_df[feature_cols].values.astype(np.float32)
    X_test_full = test_df[feature_cols].values.astype(np.float32)

    n_splits = len(np.unique(folds))
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    for fold in range(n_splits):
        tr_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_tr, y_tr_event, y_tr_efs = (
            X_train_full[tr_idx],
            y_event[tr_idx],
            y_true[tr_idx],
        )
        X_val, y_val_event, y_val_efs = (
            X_train_full[val_idx],
            y_event[val_idx],
            y_true[val_idx],
        )

        # ----------------------------------
        # Model 1: LightGBM
        # ----------------------------------
        dtrain = lgb.Dataset(X_tr, label=y_tr_event)
        dval = lgb.Dataset(X_val, label=y_val_event, reference=dtrain)

        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred_lgb = lgb_model.predict(X_val)
        test_pred_lgb = lgb_model.predict(X_test_full)

        # ----------------------------------
        # Model 2: XGBoost
        # ----------------------------------
        xgb_model = xgb.XGBClassifier(
            **xgb_params, n_estimators=1000, early_stopping_rounds=50
        )
        xgb_model.fit(X_tr, y_tr_event, eval_set=[(X_val, y_val_event)], verbose=False)

        val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
        test_pred_xgb = xgb_model.predict_proba(X_test_full)[:, 1]

        # ----------------------------------
        # Model 3: PyTorch TabularResNet
        # ----------------------------------
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_full)

        train_ds = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr_efs, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_ds, batch_size=1024, shuffle=True, drop_last=False
        )

        nn_model = TabularResNet(
            input_dim=X_train_full.shape[1], hidden_dim=128, num_blocks=2
        ).to(device)
        optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = SmoothConcordanceLoss(temperature=1.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        best_val_err = float("inf")
        best_nn_weights = None

        for epoch in range(30):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred_risk = nn_model(bx)
                loss = criterion(pred_risk, by)
                loss.backward()
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
                val_risk_nn = nn_model(val_tensor).cpu().numpy()
                val_err = calculate_concordance_error(y_val_efs, val_risk_nn)

                if val_err < best_val_err:
                    best_val_err = val_err
                    best_nn_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

        nn_model.load_state_dict(best_nn_weights)
        nn_model.eval()
        with torch.no_grad():
            val_pred_nn = (
                torch.sigmoid(nn_model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)))
                .cpu()
                .numpy()
            )
            test_pred_nn = (
                torch.sigmoid(nn_model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)))
                .cpu()
                .numpy()
            )

        # ----------------------------------
        # Blend & Record Fold Results
        # ----------------------------------
        val_pred_fold = 0.4 * val_pred_lgb + 0.4 * val_pred_xgb + 0.2 * val_pred_nn
        test_pred_fold = 0.4 * test_pred_lgb + 0.4 * test_pred_xgb + 0.2 * test_pred_nn

        oof_preds[val_idx] = val_pred_fold
        test_preds += test_pred_fold / n_splits

        fold_score = calculate_concordance_error(y_val_efs, val_pred_fold)

    # Compute overall out-of-fold validation score
    oof_score = calculate_concordance_error(y_true, oof_preds)

    # Format and save test predictions
    test_raw = pd.read_parquet("./input/test.parquet")
    if "id" in test_raw.columns:
        test_ids = test_raw["id"].values
    elif "ID" in test_raw.columns:
        test_ids = test_raw["ID"].values
    else:
        test_ids = np.arange(len(test_preds))

    sub_df = pd.DataFrame({"id": test_ids, "efs": test_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {oof_score}")


if __name__ == "__main__":
    main()