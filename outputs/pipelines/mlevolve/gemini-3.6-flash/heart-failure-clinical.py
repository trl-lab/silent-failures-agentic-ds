import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
import lightgbm as lgb
import xgboost as xgb

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------
# 1. Feature Engineering & Preprocessing Function
# -------------------------------------------------------------------------
def load_and_preprocess_data(train_path: str, test_path: str):
    """
    Load raw heart failure clinical datasets, perform domain-specific medical
    feature engineering, handle skewed distributions, fit scale models without leakage,
    and assign stratified K-folds.
    """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "id" not in train_df.columns:
        train_df["id"] = train_df.index
    if "id" not in test_df.columns:
        test_df["id"] = test_df.index

    target_col = "DEATH_EVENT"

    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()

        # Log transformations for heavily right-skewed biomarkers
        df_feat["log_cpk"] = np.log1p(df_feat["creatinine_phosphokinase"].clip(lower=0))
        df_feat["log_serum_creatinine"] = np.log1p(
            df_feat["serum_creatinine"].clip(lower=0)
        )
        df_feat["log_platelets"] = np.log1p(df_feat["platelets"].clip(lower=0))

        # Clinical Risk Threshold Flags (Medical domain knowledge)
        df_feat["flag_hyponatremia"] = (df_feat["serum_sodium"] < 135).astype(int)
        df_feat["flag_severe_low_ef"] = (df_feat["ejection_fraction"] < 30).astype(int)
        df_feat["flag_high_creatinine"] = (df_feat["serum_creatinine"] > 1.2).astype(
            int
        )
        df_feat["flag_high_cpk"] = (df_feat["creatinine_phosphokinase"] > 200).astype(
            int
        )

        # Cardiorenal interaction ratios & products
        df_feat["cardiorenal_ratio"] = df_feat["serum_creatinine"] / (
            df_feat["ejection_fraction"] + 1e-5
        )
        df_feat["heart_sodium_prod"] = (
            df_feat["serum_sodium"] * df_feat["ejection_fraction"]
        )
        df_feat["age_ef_ratio"] = df_feat["age"] / (df_feat["ejection_fraction"] + 1e-5)
        df_feat["age_followup_ratio"] = df_feat["age"] / (df_feat["time"] + 1.0)
        df_feat["platelet_creatinine_ratio"] = df_feat["platelets"] / (
            df_feat["serum_creatinine"] + 1e-5
        )

        # Survival interaction features with follow-up time
        df_feat["creatinine_time_prod"] = df_feat["serum_creatinine"] * df_feat["time"]
        df_feat["ef_time_ratio"] = df_feat["ejection_fraction"] / (df_feat["time"] + 1.0)
        df_feat["age_time_prod"] = df_feat["age"] * df_feat["time"]

        # eGFR proxy using serum creatinine, age, and sex
        sex_factor = np.where(df_feat["sex"] == 0, 0.742, 1.0) if "sex" in df_feat.columns else 1.0
        df_feat["egfr_proxy"] = 175.0 * (df_feat["serum_creatinine"].clip(lower=0.1) ** -1.154) * (df_feat["age"] ** -0.203) * sex_factor

        # Comorbidity Burden Index
        binary_risk_cols = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]
        df_feat["comorbidity_count"] = df_feat[binary_risk_cols].sum(axis=1)

        # MAGGIC risk index proxy (combining age, ejection fraction, serum creatinine, hypertension, diabetes, smoking, sex)
        sex_col = df_feat["sex"] if "sex" in df_feat.columns else 0.5
        df_feat["maggic_proxy"] = (
            (df_feat["age"] * 0.05)
            - (df_feat["ejection_fraction"] * 0.04)
            + (df_feat["serum_creatinine"] * 0.5)
            + (df_feat["high_blood_pressure"] * 0.3)
            + (df_feat["diabetes"] * 0.2)
            + (df_feat["smoking"] * 0.2)
            + (sex_col * 0.2)
        )

        # Cardiac Shock Index proxy (CPK / ejection_fraction)
        df_feat["cardiac_shock_index"] = df_feat["creatinine_phosphokinase"] / (
            df_feat["ejection_fraction"] + 1e-5
        )

        # Renal-electrolyte synergy index
        df_feat["renal_electrolyte_synergy"] = df_feat["serum_creatinine"] / (
            df_feat["serum_sodium"] + 1e-5
        )

        # Non-linear log-hazard exposure ratio against follow-up time
        df_feat["log_hazard_exposure"] = np.log1p(
            df_feat["age"] * df_feat["serum_creatinine"] / (df_feat["ejection_fraction"] + 1e-5)
        ) / np.log1p(df_feat["time"] + 1.0)

        # Non-linear time-decay hazard interactions
        df_feat["time_decay_50"] = np.exp(-df_feat["time"] / 50.0)
        df_feat["decay_creatinine"] = df_feat["time_decay_50"] * df_feat["serum_creatinine"]
        df_feat["decay_ef"] = df_feat["time_decay_50"] * df_feat["ejection_fraction"]
        df_feat["decay_age"] = df_feat["time_decay_50"] * df_feat["age"]

        # Ejection Fraction clinical risk bucket indicators (<30%, 30-40%, >40%)
        df_feat["ef_bucket_low"] = (df_feat["ejection_fraction"] < 30).astype(int)
        df_feat["ef_bucket_mid"] = (
            (df_feat["ejection_fraction"] >= 30) & (df_feat["ejection_fraction"] <= 40)
        ).astype(int)
        df_feat["ef_bucket_high"] = (df_feat["ejection_fraction"] > 40).astype(int)

        # Serum Creatinine risk tiers (<1.2, 1.2-2.0, >2.0)
        df_feat["creatinine_tier_normal"] = (df_feat["serum_creatinine"] <= 1.2).astype(int)
        df_feat["creatinine_tier_moderate"] = (
            (df_feat["serum_creatinine"] > 1.2) & (df_feat["serum_creatinine"] <= 2.0)
        ).astype(int)
        df_feat["creatinine_tier_severe"] = (df_feat["serum_creatinine"] > 2.0).astype(int)

        # Second-order polynomial features among top medical predictors (time, serum_creatinine, ejection_fraction, age)
        df_feat["time_sq"] = df_feat["time"] ** 2
        df_feat["creatinine_sq"] = df_feat["serum_creatinine"] ** 2
        df_feat["ef_sq"] = df_feat["ejection_fraction"] ** 2
        df_feat["age_sq"] = df_feat["age"] ** 2
        df_feat["time_ef_prod"] = df_feat["time"] * df_feat["ejection_fraction"]
        df_feat["creatinine_ef_prod"] = df_feat["serum_creatinine"] * df_feat["ejection_fraction"]
        df_feat["creatinine_age_prod"] = df_feat["serum_creatinine"] * df_feat["age"]
        df_feat["ef_age_prod"] = df_feat["ejection_fraction"] * df_feat["age"]

        return df_feat

    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)

    # 10-fold Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_eng["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_eng, train_eng[target_col])):
        train_eng.loc[val_idx, "fold"] = fold

    # Standard Scaling strictly fit on train set continuous features
    continuous_cols = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
        "log_cpk",
        "log_serum_creatinine",
        "log_platelets",
        "cardiorenal_ratio",
        "heart_sodium_prod",
        "age_ef_ratio",
        "age_followup_ratio",
        "platelet_creatinine_ratio",
        "creatinine_time_prod",
        "ef_time_ratio",
        "age_time_prod",
        "egfr_proxy",
        "maggic_proxy",
        "cardiac_shock_index",
        "renal_electrolyte_synergy",
        "log_hazard_exposure",
        "time_decay_50",
        "decay_creatinine",
        "decay_ef",
        "decay_age",
        "time_sq",
        "creatinine_sq",
        "ef_sq",
        "age_sq",
        "time_ef_prod",
        "creatinine_ef_prod",
        "creatinine_age_prod",
        "ef_age_prod",
    ]

    return train_eng, test_eng


# -------------------------------------------------------------------------
# 2. PyTorch Model Architecture & Model Builders
# -------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        return self.act2(out + residual)


class CardioNetMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block = ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block(x)
        logits = self.head(x)
        return logits


def build_pytorch_model_and_optimizer(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_dim: int = 64,
    dropout_rate: float = 0.3,
):
    model = CardioNetMLP(
        input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


def build_lgb_model(
    n_estimators: int = 300, learning_rate: float = 0.03, random_state: int = 42
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        num_leaves=7,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )


def build_xgb_model(
    n_estimators: int = 300, learning_rate: float = 0.03, random_state: int = 42
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        random_state=random_state,
    )


def build_catboost_model(
    n_estimators: int = 300, learning_rate: float = 0.03, random_state: int = 42
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=3,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=0,
    )


def build_extratrees_model(
    n_estimators: int = 300, random_state: int = 42
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def build_randomforest_model(
    n_estimators: int = 300, random_state: int = 42
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )


def build_logistic_regression_model(
    C: float = 0.1, random_state: int = 42
) -> LogisticRegression:
    return LogisticRegression(
        C=C, penalty="l2", max_iter=1000, random_state=random_state
    )


# -------------------------------------------------------------------------
# 3. Main Training, Evaluation, and Ensembling Loop
# -------------------------------------------------------------------------
def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df, test_df = load_and_preprocess_data(train_path, test_path)

    target_col = "DEATH_EVENT"
    ignore_cols = ["id", target_col, "fold"]

    # Features for GBDT models
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    continuous_cols = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
        "log_cpk",
        "log_serum_creatinine",
        "log_platelets",
        "cardiorenal_ratio",
        "heart_sodium_prod",
        "age_ef_ratio",
        "age_followup_ratio",
        "platelet_creatinine_ratio",
        "creatinine_time_prod",
        "ef_time_ratio",
        "age_time_prod",
        "egfr_proxy",
        "maggic_proxy",
        "cardiac_shock_index",
        "renal_electrolyte_synergy",
        "log_hazard_exposure",
    ]
    scaled_feature_names = [f"{col}_scaled" for col in continuous_cols]
    binary_cols = [c for c in feature_cols if c not in continuous_cols]
    nn_feature_cols = scaled_feature_names + binary_cols

    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    oof_et = np.zeros(len(train_df))
    oof_rf = np.zeros(len(train_df))
    oof_lr = np.zeros(len(train_df))
    oof_nn = np.zeros(len(train_df))

    test_preds_lgb = np.zeros(len(test_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    test_preds_et = np.zeros(len(test_df))
    test_preds_rf = np.zeros(len(test_df))
    test_preds_lr = np.zeros(len(test_df))
    test_preds_nn = np.zeros(len(test_df))

    n_folds = train_df["fold"].nunique()

    continuous_cols = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
        "log_cpk",
        "log_serum_creatinine",
        "log_platelets",
        "cardiorenal_ratio",
        "heart_sodium_prod",
        "age_ef_ratio",
        "age_followup_ratio",
        "platelet_creatinine_ratio",
        "creatinine_time_prod",
        "ef_time_ratio",
        "age_time_prod",
        "egfr_proxy",
        "maggic_proxy",
        "cardiac_shock_index",
        "renal_electrolyte_synergy",
        "log_hazard_exposure",
    ]
    scaled_feature_names = [f"{col}_scaled" for col in continuous_cols]

    for fold in range(n_folds):
        train_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        train_fold = train_df[train_idx].copy()
        val_fold = train_df[val_idx].copy()
        test_fold = test_df.copy()

        scaler = StandardScaler()
        scaler.fit(train_fold[continuous_cols])
        train_fold[scaled_feature_names] = scaler.transform(train_fold[continuous_cols])
        val_fold[scaled_feature_names] = scaler.transform(val_fold[continuous_cols])
        test_fold[scaled_feature_names] = scaler.transform(test_fold[continuous_cols])

        # Data arrays for GBDTs
        X_tr = train_fold[feature_cols].values
        y_tr = train_fold[target_col].values
        X_va = val_fold[feature_cols].values
        y_va = val_fold[target_col].values
        X_test = test_fold[feature_cols].values

        # Data tensors for PyTorch NN
        X_tr_nn = torch.tensor(
            train_fold[nn_feature_cols].values, dtype=torch.float32
        ).to(device)
        y_tr_nn = (
            torch.tensor(train_fold[target_col].values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        X_va_nn = torch.tensor(
            val_fold[nn_feature_cols].values, dtype=torch.float32
        ).to(device)
        X_test_nn = torch.tensor(
            test_fold[nn_feature_cols].values, dtype=torch.float32
        ).to(device)

        # LightGBM
        lgb_clf = build_lgb_model(
            n_estimators=500, learning_rate=0.02, random_state=42 + fold
        )
        lgb_clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        oof_lgb[val_idx] = lgb_clf.predict_proba(X_va)[:, 1]
        test_preds_lgb += lgb_clf.predict_proba(X_test)[:, 1] / n_folds

        # XGBoost
        xgb_clf = build_xgb_model(
            n_estimators=500, learning_rate=0.02, random_state=42 + fold
        )
        xgb_clf.set_params(early_stopping_rounds=50)
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)[:, 1]
        test_preds_xgb += xgb_clf.predict_proba(X_test)[:, 1] / n_folds

        # CatBoost
        cat_clf = build_catboost_model(
            n_estimators=500, learning_rate=0.02, random_state=42 + fold
        )
        cat_clf.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
        )

        oof_cat[val_idx] = cat_clf.predict_proba(X_va)[:, 1]
        test_preds_cat += cat_clf.predict_proba(X_test)[:, 1] / n_folds

        # ExtraTrees
        et_clf = build_extratrees_model(
            n_estimators=300, random_state=42 + fold
        )
        et_clf.fit(X_tr, y_tr)

        oof_et[val_idx] = et_clf.predict_proba(X_va)[:, 1]
        test_preds_et += et_clf.predict_proba(X_test)[:, 1] / n_folds

        # RandomForest
        rf_clf = build_randomforest_model(
            n_estimators=300, random_state=42 + fold
        )
        rf_clf.fit(X_tr, y_tr)

        oof_rf[val_idx] = rf_clf.predict_proba(X_va)[:, 1]
        test_preds_rf += rf_clf.predict_proba(X_test)[:, 1] / n_folds

        # Logistic Regression
        lr_clf = build_logistic_regression_model(
            C=0.1, random_state=42 + fold
        )
        lr_clf.fit(X_tr_nn.cpu().numpy(), y_tr)

        oof_lr[val_idx] = lr_clf.predict_proba(X_va_nn.cpu().numpy())[:, 1]
        test_preds_lr += lr_clf.predict_proba(X_test_nn.cpu().numpy())[:, 1] / n_folds

        # PyTorch Neural Network
        nn_model, criterion, optimizer = build_pytorch_model_and_optimizer(
            input_dim=len(nn_feature_cols),
            lr=2e-3,
            weight_decay=1e-2,
            hidden_dim=64,
            dropout_rate=0.3,
        )
        nn_model.to(device)

        best_val_auc = 0.0
        best_weights = None

        for epoch in range(1, 201):
            nn_model.train()
            optimizer.zero_grad()
            logits = nn_model(X_tr_nn)
            loss = criterion(logits, y_tr_nn)
            loss.backward()
            optimizer.step()

            nn_model.eval()
            with torch.no_grad():
                val_logits = nn_model(X_va_nn)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_auc = roc_auc_score(y_va, val_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

        if best_weights is not None:
            nn_model.load_state_dict(best_weights)

        nn_model.eval()
        with torch.no_grad():
            val_probs_best = torch.sigmoid(nn_model(X_va_nn)).cpu().numpy().flatten()
            test_probs_best = torch.sigmoid(nn_model(X_test_nn)).cpu().numpy().flatten()

        oof_nn[val_idx] = val_probs_best
        test_preds_nn += test_probs_best / n_folds

    y_true = train_df[target_col].values

    # Convert OOF and Test predictions to fractional ranks to normalize probability scale
    def to_rank(arr):
        return rankdata(arr) / len(arr)

    oof_lgb_r = to_rank(oof_lgb)
    oof_xgb_r = to_rank(oof_xgb)
    oof_cat_r = to_rank(oof_cat)
    oof_et_r = to_rank(oof_et)
    oof_rf_r = to_rank(oof_rf)
    oof_lr_r = to_rank(oof_lr)
    oof_nn_r = to_rank(oof_nn)

    test_lgb_r = to_rank(test_preds_lgb)
    test_xgb_r = to_rank(test_preds_xgb)
    test_cat_r = to_rank(test_preds_cat)
    test_et_r = to_rank(test_preds_et)
    test_rf_r = to_rank(test_preds_rf)
    test_lr_r = to_rank(test_preds_lr)
    test_nn_r = to_rank(test_preds_nn)

    X_oof_ranks = np.column_stack([
        oof_lgb_r, oof_xgb_r, oof_cat_r, oof_et_r, oof_rf_r, oof_lr_r, oof_nn_r
    ])
    X_test_ranks = np.column_stack([
        test_lgb_r, test_xgb_r, test_cat_r, test_et_r, test_rf_r, test_lr_r, test_nn_r
    ])

    # Strict nested out-of-fold meta-stacking & SLSQP weight optimization
    n_models = X_oof_ranks.shape[1]
    meta_oof = np.zeros(len(train_df))
    meta_test = np.zeros(len(test_df))
    slsqp_oof = np.zeros(len(train_df))
    slsqp_test = np.zeros(len(test_df))

    for fold in range(n_folds):
        tr_idx = train_df["fold"] != fold
        va_idx = train_df["fold"] == fold

        # 1. Meta Logistic Regression on out-of-fold splits
        meta_model = LogisticRegression(C=0.5, penalty="l2", random_state=42 + fold)
        meta_model.fit(X_oof_ranks[tr_idx], y_true[tr_idx])
        meta_oof[va_idx] = meta_model.predict_proba(X_oof_ranks[va_idx])[:, 1]
        meta_test += meta_model.predict_proba(X_test_ranks)[:, 1] / n_folds

        # 2. Scipy SLSQP ensemble weight optimization directly maximizing ROC-AUC
        def loss_func(weights):
            w = np.array(weights)
            w = w / (np.sum(w) + 1e-15)
            pred = X_oof_ranks[tr_idx] @ w
            return -roc_auc_score(y_true[tr_idx], pred)

        init_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = ({"type": "eq", "fun": lambda w: 1.0 - sum(w)})
        res = minimize(
            loss_func,
            init_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200},
        )
        opt_w = res.x / np.sum(res.x)

        slsqp_oof[va_idx] = X_oof_ranks[va_idx] @ opt_w
        slsqp_test += (X_test_ranks @ opt_w) / n_folds

    oof_final = 0.5 * to_rank(meta_oof) + 0.5 * to_rank(slsqp_oof)
    test_final = 0.5 * to_rank(meta_test) + 0.5 * to_rank(slsqp_test)

    final_validation_score = roc_auc_score(y_true, oof_final)

    # Save final predictions to `./submission/submission.csv`
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame(
        {
            "id": (
                test_df["id"].values
                if "id" in test_df.columns
                else np.arange(len(test_df))
            ),
            "DEATH_EVENT": test_final,
        }
    )
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_validation_score}")


if __name__ == "__main__":
    main()