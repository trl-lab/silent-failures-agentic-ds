import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# 1. Reproducibility & Environment Setup
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ------------------------------------------------------------------------------
# 2. Model Architecture Definitions
# ------------------------------------------------------------------------------
class TabularResBlock(nn.Module):

    def __init__(self, in_features, hidden_features, dropout_rate=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, in_features),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# ------------------------------------------------------------------------------
# 3. Data Processing & Feature Engineering Utilities
# ------------------------------------------------------------------------------
def standardize_columns(df):
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if "id" in c_lower and col != "id":
            col_map[col] = "id"
        elif "preg" in c_lower:
            col_map[col] = "Pregnancies"
        elif "glu" in c_lower or "plas" in c_lower:
            col_map[col] = "Glucose"
        elif (
            "blood" in c_lower or "press" in c_lower or "pres" in c_lower
        ) and "pedi" not in c_lower:
            col_map[col] = "BloodPressure"
        elif "skin" in c_lower or "tricep" in c_lower:
            col_map[col] = "SkinThickness"
        elif "insu" in c_lower:
            col_map[col] = "Insulin"
        elif "bmi" in c_lower or "mass" in c_lower:
            col_map[col] = "BMI"
        elif (
            "pedig" in c_lower
            or "pedi" in c_lower
            or "func" in c_lower
            or "dpf" in c_lower
        ):
            col_map[col] = "DiabetesPedigreeFunction"
        elif "age" in c_lower:
            col_map[col] = "Age"
        elif "class" in c_lower or "target" in c_lower or "outcome" in c_lower:
            col_map[col] = "class"
    return df.rename(columns=col_map)


zero_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def preprocess_physiological_zeros(df_input):
    df = df_input.copy()
    for col in zero_missing_cols:
        df[col] = df[col].replace(0, np.nan)
        df[f"{col}_is_missing"] = df[col].isna().astype(float)
    missing_flags = [f"{c}_is_missing" for c in zero_missing_cols]
    df["missing_med_count"] = df[missing_flags].sum(axis=1).astype(float)
    return df


def add_domain_interactions(df_imputed):
    d = df_imputed.copy()
    d["HOMA_IR"] = (d["Glucose"] * d["Insulin"]) / 405.0
    d["Glucose_BMI_ratio"] = d["Glucose"] / (d["BMI"] + 1e-5)
    d["Insulin_Glucose_ratio"] = d["Insulin"] / (d["Glucose"] + 1e-5)
    d["Pregnancies_Age_ratio"] = d["Pregnancies"] / (d["Age"] + 1e-5)
    d["BP_Age_ratio"] = d["BloodPressure"] / (d["Age"] + 1e-5)

    d["Glucose_Age"] = d["Glucose"] * d["Age"]
    d["BMI_Age"] = d["BMI"] * d["Age"]
    d["DPF_Age"] = d["DiabetesPedigreeFunction"] * d["Age"]
    d["Glucose_BP"] = d["Glucose"] * d["BloodPressure"]

    d["Glucose_Prediabetes"] = (d["Glucose"] >= 100).astype(float)
    d["Glucose_Diabetes"] = (d["Glucose"] >= 126).astype(float)
    d["BMI_Overweight"] = (d["BMI"] >= 25).astype(float)
    d["BMI_Obese"] = (d["BMI"] >= 30).astype(float)
    d["Age_Risk"] = (d["Age"] >= 35).astype(float)
    d["High_Pregnancies"] = (d["Pregnancies"] >= 4).astype(float)

    d["Metabolic_Risk_Score"] = (
        d["Glucose_Diabetes"]
        + d["BMI_Obese"]
        + d["Age_Risk"]
        + (d["HOMA_IR"] > 1.9).astype(float)
    )
    d["BMI_Age_Interaction"] = (d["BMI"] ** 1.5) * np.sqrt(d["Age"])

    d["log_Insulin"] = np.log1p(np.maximum(0, d["Insulin"]))
    d["log_DPF"] = np.log1p(np.maximum(0, d["DiabetesPedigreeFunction"]))
    d["log_SkinThickness"] = np.log1p(np.maximum(0, d["SkinThickness"]))

    d["Glucose_dev_normal"] = np.abs(d["Glucose"] - 90)
    d["BP_dev_normal"] = np.abs(d["BloodPressure"] - 80)
    d["BMI_dev_normal"] = np.abs(d["BMI"] - 22.5)
    return d


# ------------------------------------------------------------------------------
# 4. Neural Network Training Routine
# ------------------------------------------------------------------------------
def train_nn_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    fold,
    epochs=120,
    batch_size=32,
    lr=1e-3,
):
    input_dim = X_tr.shape[1]
    model = TabularResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout_rate=0.25
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    best_val_probs = None
    best_te_probs = None
    model_save_path = f"./working/resnet_fold_{fold}.pt"

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_va, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_probs = val_probs
                te_logits = model(X_te_t)
                best_te_probs = torch.sigmoid(te_logits).cpu().numpy().flatten()
                torch.save(model.state_dict(), model_save_path)

    return best_val_probs, best_te_probs


# ------------------------------------------------------------------------------
# 5. Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train_df = standardize_columns(train_raw)
    test_df = standardize_columns(test_raw)

    if "class" in train_df.columns:
        y_raw = train_df["class"]
        train_df = train_df.drop(columns=["class"])
        label_map = {
            "tested_positive": 1,
            "tested_negative": 0,
            "positive": 1,
            "negative": 0,
            "1": 1,
            "0": 0,
            1: 1,
            0: 0,
            True: 1,
            False: 0,
        }
        y = y_raw.map(label_map).fillna(0).astype(int).values
    else:
        raise ValueError("Target column 'class' not found in training dataset.")

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
        test_df = test_df.drop(columns=["id"])
    else:
        test_ids = np.arange(len(test_df))

    if "id" in train_df.columns:
        train_df = train_df.drop(columns=["id"])

    feature_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
        if col not in test_df.columns:
            test_df[col] = 0.0

    train_df = train_df[feature_cols].copy()
    test_df = test_df[feature_cols].copy()

    train_pre = preprocess_physiological_zeros(train_df)
    test_pre = preprocess_physiological_zeros(test_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_models = 5
    oof_meta = np.zeros((len(train_pre), n_models))
    test_meta_folds = np.zeros((skf.n_splits, len(test_pre), n_models))
    raw_cols = train_pre.columns.tolist()

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pre, y)):
        X_tr_raw = train_pre.iloc[train_idx].copy()
        X_va_raw = train_pre.iloc[val_idx].copy()
        X_te_raw = test_pre.copy()
        y_tr = y[train_idx]
        y_va = y[val_idx]

        # Strict Fold Isolation for Imputation
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=30, max_depth=6, random_state=42
            ),
            max_iter=10,
            random_state=42,
        )
        imputer.fit(X_tr_raw)

        X_tr_imp = pd.DataFrame(imputer.transform(X_tr_raw), columns=raw_cols)
        X_va_imp = pd.DataFrame(imputer.transform(X_va_raw), columns=raw_cols)
        X_te_imp = pd.DataFrame(imputer.transform(X_te_raw), columns=raw_cols)

        # Feature Engineering
        X_tr_fe = add_domain_interactions(X_tr_imp)
        X_va_fe = add_domain_interactions(X_va_imp)
        X_te_fe = add_domain_interactions(X_te_imp)

        # Standard Scaling fit strictly on training fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_fe)
        X_va_scaled = scaler.transform(X_va_fe)
        X_te_scaled = scaler.transform(X_te_fe)

        # Model 1: PyTorch Tabular ResNet
        val_pred_nn, te_pred_nn = train_nn_fold(
            X_tr_scaled,
            y_tr,
            X_va_scaled,
            y_va,
            X_te_scaled,
            fold=fold,
            epochs=120,
            batch_size=32,
            lr=1e-3,
        )

        # Model 2: CatBoost Classifier
        m_cat = CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
        )
        m_cat.fit(X_tr_fe, y_tr)
        m_cat.save_model(f"./working/catboost_fold_{fold}.cbm")
        val_pred_cat = m_cat.predict_proba(X_va_fe)[:, 1]
        te_pred_cat = m_cat.predict_proba(X_te_fe)[:, 1]

        # Model 3: LightGBM Classifier
        m_lgb = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=42 + fold,
            verbose=-1,
        )
        m_lgb.fit(X_tr_fe, y_tr)
        m_lgb.booster_.save_model(f"./working/lgb_fold_{fold}.txt")
        val_pred_lgb = m_lgb.predict_proba(X_va_fe)[:, 1]
        te_pred_lgb = m_lgb.predict_proba(X_te_fe)[:, 1]

        # Model 4: ExtraTrees Classifier
        m_et = ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_split=5, random_state=42 + fold
        )
        m_et.fit(X_tr_scaled, y_tr)
        joblib.dump(m_et, f"./working/extratrees_fold_{fold}.joblib")
        val_pred_et = m_et.predict_proba(X_va_scaled)[:, 1]
        te_pred_et = m_et.predict_proba(X_te_scaled)[:, 1]

        # Model 5: XGBoost Classifier
        m_xgb = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="logloss",
        )
        m_xgb.fit(X_tr_fe, y_tr)
        val_pred_xgb = m_xgb.predict_proba(X_va_fe)[:, 1]
        te_pred_xgb = m_xgb.predict_proba(X_te_fe)[:, 1]

        oof_meta[val_idx, 0] = val_pred_nn
        oof_meta[val_idx, 1] = val_pred_cat
        oof_meta[val_idx, 2] = val_pred_lgb
        oof_meta[val_idx, 3] = val_pred_et
        oof_meta[val_idx, 4] = val_pred_xgb

        test_meta_folds[fold, :, 0] = te_pred_nn
        test_meta_folds[fold, :, 1] = te_pred_cat
        test_meta_folds[fold, :, 2] = te_pred_lgb
        test_meta_folds[fold, :, 3] = te_pred_et
        test_meta_folds[fold, :, 4] = te_pred_xgb

        fold_blend = np.mean(
            [val_pred_nn, val_pred_cat, val_pred_lgb, val_pred_et, val_pred_xgb],
            axis=0,
        )
        fold_auc = roc_auc_score(y_va, fold_blend)
        print(f"Fold {fold + 1} Sub-models mean ROC-AUC: {fold_auc:.5f}")

    # Meta-Learner Stacking
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_meta, y)

    oof_preds = meta_model.predict_proba(oof_meta)[:, 1]
    test_meta_avg = np.mean(test_meta_folds, axis=0)
    test_preds = meta_model.predict_proba(test_meta_avg)[:, 1]

    final_auc = roc_auc_score(y, oof_preds)

    sub = pd.DataFrame({"id": test_ids, "class": test_preds})
    sub.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_auc:.5f}")


if __name__ == "__main__":
    main()
