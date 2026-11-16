import os
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 0. Reproducibility and Environment Setup
# =========================================================================


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
# 1. Domain Feature Engineering
# =========================================================================


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs clinical, immunogenetic, and demographic interaction features for HCT survival.
    """
    df = df.copy()

    # 1. Age & Demographic Interactions
    if "donor_age" in df.columns and "age_at_hct" in df.columns:
        df["age_diff"] = df["donor_age"] - df["age_at_hct"]
        df["age_ratio"] = df["donor_age"] / (df["age_at_hct"] + 1.0)
        df["is_donor_older"] = (df["donor_age"] > df["age_at_hct"]).astype(float)
        df["is_pediatric"] = (df["age_at_hct"] < 18.0).astype(float)
        df["is_elderly_recipient"] = (df["age_at_hct"] >= 60.0).astype(float)
        df["is_young_donor"] = (df["donor_age"] < 30.0).astype(float)
        df["is_old_donor"] = (df["donor_age"] >= 50.0).astype(float)

    # Sex mismatch immunologic risk
    if "sex_match" in df.columns:
        df["is_female_to_male"] = (df["sex_match"] == "F-M").astype(float)
        df["is_sex_matched"] = df["sex_match"].isin(["M-M", "F-F"]).astype(float)

    # 2. HLA Match Metrics & Discrepancies
    hla_high_cols = [
        c
        for c in [
            "hla_match_a_high",
            "hla_match_b_high",
            "hla_match_c_high",
            "hla_match_dqb1_high",
            "hla_match_drb1_high",
        ]
        if c in df.columns
    ]
    hla_low_cols = [
        c
        for c in [
            "hla_match_a_low",
            "hla_match_b_low",
            "hla_match_c_low",
            "hla_match_dqb1_low",
            "hla_match_drb1_low",
        ]
        if c in df.columns
    ]

    if hla_high_cols:
        df["hla_high_sum"] = df[hla_high_cols].sum(axis=1)
        df["hla_high_missing_count"] = df[hla_high_cols].isna().sum(axis=1)
        df["hla_high_mismatch_count"] = (df[hla_high_cols] < 2.0).sum(axis=1)
        df["hla_high_perfect_count"] = (df[hla_high_cols] == 2.0).sum(axis=1)

    if hla_low_cols:
        df["hla_low_sum"] = df[hla_low_cols].sum(axis=1)

    if "hla_high_res_10" in df.columns and "hla_low_res_10" in df.columns:
        df["hla_res_10_diff"] = df["hla_high_res_10"] - df["hla_low_res_10"]
    if "hla_high_res_8" in df.columns and "hla_low_res_8" in df.columns:
        df["hla_res_8_diff"] = df["hla_high_res_8"] - df["hla_low_res_8"]
    if "hla_high_res_6" in df.columns and "hla_low_res_6" in df.columns:
        df["hla_res_6_diff"] = df["hla_high_res_6"] - df["hla_low_res_6"]

    # 3. Comorbidity Burdens & Performance Index
    comorbidities = [
        c
        for c in [
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
            "vent_hist",
        ]
        if c in df.columns
    ]

    if comorbidities:
        df["comorbidity_yes_count"] = (df[comorbidities] == "Yes").sum(axis=1)
        df["comorbidity_not_done_count"] = (df[comorbidities] == "Not done").sum(axis=1)

    if (
        "hepatic_severe" in df.columns
        and "pulm_severe" in df.columns
        and "renal_issue" in df.columns
    ):
        df["severe_organ_risk"] = (
            (df["hepatic_severe"] == "Yes")
            | (df["pulm_severe"] == "Yes")
            | (df["renal_issue"] == "Yes")
        ).astype(float)

    if "karnofsky_score" in df.columns and "comorbidity_score" in df.columns:
        karnofsky = df["karnofsky_score"].fillna(80.0)
        comorb = df["comorbidity_score"].fillna(0.0)
        df["karnofsky_comorb_ratio"] = karnofsky / (comorb + 1.0)
        df["karnofsky_low"] = (karnofsky < 80.0).astype(float)

    # 4. Clinical Regime & Temporal Evolution
    if "year_hct" in df.columns:
        df["year_relative"] = df["year_hct"] - 2008.0

    if "prim_disease_hct" in df.columns and "mrd_hct" in df.columns:
        df["disease_mrd_inter"] = (
            df["prim_disease_hct"].astype(str) + "_" + df["mrd_hct"].astype(str)
        )
    if "graft_type" in df.columns and "donor_related" in df.columns:
        df["graft_donor_inter"] = (
            df["graft_type"].astype(str) + "_" + df["donor_related"].astype(str)
        )
    if "conditioning_intensity" in df.columns and "tbi_status" in df.columns:
        df["cond_tbi_inter"] = (
            df["conditioning_intensity"].astype(str)
            + "_"
            + df["tbi_status"].astype(str)
        )
    if "race_group" in df.columns and "ethnicity" in df.columns:
        df["race_ethnicity_inter"] = (
            df["race_group"].astype(str) + "_" + df["ethnicity"].astype(str)
        )

    return df


# =========================================================================
# 2. Evaluation Metric: Stratified Concordance Index
# =========================================================================


def compute_c_index(
    durations: np.ndarray, risk_scores: np.ndarray, events: np.ndarray
) -> float:
    """
    Computes Harrell's Concordance Index for right-censored survival data.
    Higher risk_score implies earlier event / worse survival prognosis.
    """
    durations = np.asarray(durations, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    events = np.asarray(events, dtype=float)

    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(durations, -risk_scores, events))
    except Exception:
        pass

    order = np.argsort(durations)
    durations = durations[order]
    risk_scores = risk_scores[order]
    events = events[order]

    concordant = 0.0
    total_pairs = 0.0

    event_indices = np.where(events == 1)[0]
    for i in event_indices:
        t_i = durations[i]
        r_i = risk_scores[i]

        mask = durations > t_i
        if not np.any(mask):
            continue

        r_j = risk_scores[mask]
        concordant += np.sum(r_i > r_j) + 0.5 * np.sum(r_i == r_j)
        total_pairs += len(r_j)

    if total_pairs == 0:
        return 0.5
    return float(concordant / total_pairs)


def compute_stratified_c_index(
    df: pd.DataFrame,
    risk_col: str = "risk",
    race_col: str = "race_group",
    time_col: str = "efs_time",
    event_col: str = "efs",
) -> float:
    """
    Computes demographic subgroup-stratified Concordance Index across race groups.
    """
    scores = []
    for race, group in df.groupby(race_col):
        if group[event_col].sum() > 0 and len(group) > 5:
            c = compute_c_index(
                group[time_col].values, group[risk_col].values, group[event_col].values
            )
            scores.append(c)
    return float(np.mean(scores)) if len(scores) > 0 else 0.5


# =========================================================================
# 3. Survival Losses and Neural Architecture
# =========================================================================


class CoxPartialLogLikelihoodLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        risk_scores = risk_scores.view(-1)
        durations = durations.view(-1)
        events = events.view(-1)

        sorted_indices = torch.argsort(durations, descending=True)
        sorted_risk = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices]

        max_risk = torch.max(sorted_risk)
        exp_risk = torch.exp(sorted_risk - max_risk)
        risk_set_sum = torch.cumsum(exp_risk, dim=0) + self.eps
        log_risk_set_sum = torch.log(risk_set_sum) + max_risk

        uncensored_loss = sorted_events * (log_risk_set_sum - sorted_risk)
        num_events = torch.sum(sorted_events)
        if num_events == 0:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
        return torch.sum(uncensored_loss) / num_events


class FastPairwiseRankingLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        risk_scores = risk_scores.view(-1)
        durations = durations.view(-1)
        events = events.view(-1)

        time_diff = durations.unsqueeze(1) - durations.unsqueeze(0)
        event_i = events.unsqueeze(1)
        mask = (time_diff < 0) & (event_i == 1)

        if not torch.any(mask):
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
        loss_matrix = torch.log1p(torch.exp(-self.sigma * risk_diff))
        return torch.sum(loss_matrix * mask.float()) / (torch.sum(mask.float()) + 1e-7)


class CompositeSurvivalLoss(nn.Module):
    def __init__(self, cox_weight: float = 0.7, rank_weight: float = 0.3):
        super().__init__()
        self.cox_loss = CoxPartialLogLikelihoodLoss()
        self.rank_loss = FastPairwiseRankingLoss()
        self.cox_weight = cox_weight
        self.rank_weight = rank_weight

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        return self.cox_weight * self.cox_loss(
            risk_scores, durations, events
        ) + self.rank_weight * self.rank_loss(risk_scores, durations, events)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act1 = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.drop1(self.act1(self.linear1(self.norm1(x))))
        out = self.drop2(self.linear2(self.norm2(out)))
        return res + out


class HCTSurvivalResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(self.head_norm(h))


# =========================================================================
# 4. Pipeline Execution: Preprocessing, Training, and Evaluation
# =========================================================================


def main():
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # 1. Load Raw Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    id_col = "ID" if "ID" in train_df.columns else "id"
    test_id_col = "ID" if "ID" in test_df.columns else "id"

    # 2. Build Engineered Features
    train_feat = build_features(train_df)
    test_feat = build_features(test_df)

    # 3. Categorical Encodings (Frequency + Ordinal Encoding)
    exclude_cols = {id_col, "efs", "efs_time"}
    categorical_cols = [
        c
        for c in train_feat.columns
        if c not in exclude_cols
        and (train_feat[c].dtype == "object" or train_feat[c].dtype.name == "category")
    ]

    for cat_col in categorical_cols:
        train_feat[cat_col] = train_feat[cat_col].fillna("MISSING").astype(str)
        test_feat[cat_col] = test_feat[cat_col].fillna("MISSING").astype(str)

        freq_map = train_feat[cat_col].value_counts(normalize=True).to_dict()
        train_feat[f"{cat_col}_freq"] = (
            train_feat[cat_col].map(freq_map).fillna(0.0).astype(float)
        )
        test_feat[f"{cat_col}_freq"] = (
            test_feat[cat_col].map(freq_map).fillna(0.0).astype(float)
        )

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_feat[f"{cat_col}_code"] = oe.fit_transform(train_feat[[cat_col]]).astype(
            float
        )
        test_feat[f"{cat_col}_code"] = oe.transform(test_feat[[cat_col]]).astype(float)

    # 4. Feature Selection (Numeric and Encoded Columns Only)
    feature_cols = [
        c
        for c in train_feat.columns
        if c not in exclude_cols
        and c not in categorical_cols
        and np.issubdtype(train_feat[c].dtype, np.number)
    ]

    print(
        f"Dataset loaded: {len(train_feat)} train, {len(test_feat)} test. Total Features: {len(feature_cols)}"
    )

    # Impute medians computed from training data
    medians = train_feat[feature_cols].median()
    X_train_full = train_feat[feature_cols].fillna(medians).values.astype(np.float32)
    X_test_full = test_feat[feature_cols].fillna(medians).values.astype(np.float32)

    durations_full = train_feat["efs_time"].values.astype(np.float32)
    events_full = train_feat["efs"].values.astype(np.float32)

    # 5. Stratified Cross-Validation Setup
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = np.zeros(len(train_feat), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_feat, train_feat["efs"])):
        folds[val_idx] = fold

    oof_nn_preds = np.zeros(len(train_feat))
    oof_xgb_preds = np.zeros(len(train_feat))
    test_nn_preds = np.zeros(len(test_feat))
    test_xgb_preds = np.zeros(len(test_feat))

    # 6. Model Training Loop
    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        # Feature Scaling (Fit on Train fold, Transform Val/Test)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_full[train_idx])
        X_va = scaler.transform(X_train_full[val_idx])
        X_te = scaler.transform(X_test_full)

        t_tr, e_tr = durations_full[train_idx], events_full[train_idx]
        t_va, e_va = durations_full[val_idx], events_full[val_idx]

        # Model 1: Tabular Deep Survival ResNet
        nn_model = HCTSurvivalResNet(
            input_dim=len(feature_cols),
            hidden_dim=192,
            num_res_blocks=3,
            dropout_rate=0.25,
        ).to(device)

        criterion = CompositeSurvivalLoss(cox_weight=0.7, rank_weight=0.3)
        optimizer = AdamW(nn_model.parameters(), lr=1.5e-3, weight_decay=1e-4, eps=1e-8)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(t_tr, dtype=torch.float32),
            torch.tensor(e_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, drop_last=True
        )

        best_c_index = -1.0
        best_state = None

        for epoch in range(20):
            nn_model.train()
            for bx, bt, be in train_loader:
                bx, bt, be = bx.to(device), bt.to(device), be.to(device)
                optimizer.zero_grad()
                pred_risk = nn_model(bx)
                loss = criterion(pred_risk, bt, be)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
                val_risk = nn_model(val_tensor).cpu().numpy().flatten()

            fold_c = compute_c_index(t_va, val_risk, e_va)
            if fold_c > best_c_index:
                best_c_index = fold_c
                best_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict(best_state)
        nn_model.eval()
        with torch.no_grad():
            oof_nn_preds[val_idx] = (
                nn_model(torch.tensor(X_va, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
            test_nn_fold = (
                nn_model(torch.tensor(X_te, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
            test_nn_preds += test_nn_fold / n_folds

        # Model 2: XGBoost Survival:Cox
        y_xgb_tr = np.where(e_tr == 1, t_tr, -t_tr)
        y_xgb_va = np.where(e_va == 1, t_va, -t_va)

        dtrain = xgb.DMatrix(X_tr, label=y_xgb_tr)
        dval = xgb.DMatrix(X_va, label=y_xgb_va)
        dtest = xgb.DMatrix(X_te)

        xgb_params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.5,
            "reg_lambda": 3.0,
            "random_state": 42 + fold,
        }

        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=600,
            evals=[(dval, "val")],
            verbose_eval=False,
        )

        oof_xgb_preds[val_idx] = xgb_model.predict(dval)
        test_xgb_preds += xgb_model.predict(dtest) / n_folds

        fold_nn_c = compute_c_index(t_va, oof_nn_preds[val_idx], e_va)
        fold_xgb_c = compute_c_index(t_va, oof_xgb_preds[val_idx], e_va)
        print(
            f"Fold {fold + 1}/{n_folds} | NN C-index: {fold_nn_c:.4f} | XGB C-index: {fold_xgb_c:.4f}"
        )

    # 7. Out-Of-Fold Ensembling & Validation Scoring
    oof_nn_rank = rankdata(oof_nn_preds) / len(oof_nn_preds)
    oof_xgb_rank = rankdata(oof_xgb_preds) / len(oof_xgb_preds)
    oof_ensemble = 0.5 * oof_nn_rank + 0.5 * oof_xgb_rank

    train_feat["risk_pred"] = oof_ensemble
    global_c_index = compute_c_index(durations_full, oof_ensemble, events_full)
    strat_c_index = compute_stratified_c_index(
        train_feat,
        risk_col="risk_pred",
        race_col="race_group",
        time_col="efs_time",
        event_col="efs",
    )

    c_index_error = 1.0 - strat_c_index
    print(
        f"Validation Global C-Index: {global_c_index:.5f} | Stratified Subgroup C-Index: {strat_c_index:.5f} | C-Index Error: {c_index_error:.5f}"
    )

    # 8. Test Inference and Submission Generation
    test_nn_rank = rankdata(test_nn_preds) / len(test_nn_preds)
    test_xgb_rank = rankdata(test_xgb_preds) / len(test_xgb_preds)
    test_final_preds = 0.5 * test_nn_rank + 0.5 * test_xgb_rank

    sub_id = (
        test_df[test_id_col].values
        if test_id_col in test_df.columns
        else np.arange(len(test_df))
    )
    submission_df = pd.DataFrame(
        {
            "id": sub_id,
            "efs": test_final_preds,
        }
    )

    sub_path = "./submission/submission.csv"
    submission_df.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path), "Submission file was not created!"
    assert len(submission_df) == len(test_df), "Submission row count mismatch!"
    assert list(submission_df.columns) == ["id", "efs"], "Submission column mismatch!"
    assert not submission_df.isna().any().any(), "Submission contains null values!"

    print(f"Submission saved to {sub_path} with shape {submission_df.shape}")
    print(f"Final Validation Score: {c_index_error}")


if __name__ == "__main__":
    main()
