import os
import random
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def evaluate_qwk(y_true, y_pred_discrete):
    """Computes exact Quadratic Weighted Kappa score."""
    return cohen_kappa_score(
        y_true.astype(int), y_pred_discrete.astype(int), weights="quadratic"
    )


def load_and_process_data():
    """Loads raw train and test parquet data, splits train cleanly (80/20 stratified),

    and engineers domain-specific features without target leakage.
    """
    print("Loading data from ./input/...")
    raw_train = pd.read_parquet("./input/train.parquet")
    raw_test = pd.read_parquet("./input/test.parquet")

    # 1. Stratified Train/Validation split before any stateful transformations
    train_df, val_df = train_test_split(
        raw_train, test_size=0.20, random_state=SEED, stratify=raw_train["Response"]
    )
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = raw_test.copy().reset_index(drop=True)

    # 2. Extract domain features
    def extract_domain_features(df):
        df_feat = df.copy()

        # Product_Info_2 Decomposition
        if "Product_Info_2" in df_feat.columns:
            df_feat["Product_Info_2_char"] = (
                df_feat["Product_Info_2"].astype(str).str[0]
            )
            df_feat["Product_Info_2_num"] = (
                pd.to_numeric(
                    df_feat["Product_Info_2"].astype(str).str[1:], errors="coerce"
                )
                .fillna(-1)
                .astype(int)
            )

        # Biometric & Physical Interaction Features
        df_feat["BMI_Age_ratio"] = df_feat["BMI"] / (df_feat["Ins_Age"] + 0.001)
        df_feat["BMI_Age_prod"] = df_feat["BMI"] * df_feat["Ins_Age"]
        df_feat["Ht_Wt_ratio"] = df_feat["Ht"] / (df_feat["Wt"] + 0.001)
        df_feat["Ht_Wt_prod"] = df_feat["Ht"] * df_feat["Wt"]

        # Reconstructed BMI anomaly deviation score
        calc_bmi = df_feat["Wt"] / ((df_feat["Ht"] + 1e-5) ** 2)
        df_feat["BMI_diff"] = df_feat["BMI"] - calc_bmi

        # Medical Keyword Aggregations
        med_kw_cols = [c for c in df_feat.columns if c.startswith("Medical_Keyword_")]
        if med_kw_cols:
            df_feat["Med_Keyword_sum"] = df_feat[med_kw_cols].sum(axis=1)
            df_feat["Med_Keyword_mean"] = df_feat[med_kw_cols].mean(axis=1)
            df_feat["Med_Keyword_std"] = df_feat[med_kw_cols].std(axis=1)
            df_feat["Med_Keyword_Age_prod"] = (
                df_feat["Med_Keyword_sum"] * df_feat["Ins_Age"]
            )
            df_feat["Med_Keyword_BMI_prod"] = (
                df_feat["Med_Keyword_sum"] * df_feat["BMI"]
            )

        # Family History Aggregations
        fam_cols = [c for c in df_feat.columns if c.startswith("Family_Hist_")]
        if fam_cols:
            df_feat["Family_Hist_null_count"] = df_feat[fam_cols].isnull().sum(axis=1)
            df_feat["Family_Hist_mean"] = df_feat[fam_cols].mean(axis=1)
            df_feat["Family_Hist_max"] = df_feat[fam_cols].max(axis=1)

        # Medical History Aggregations
        med_hist_cols = [c for c in df_feat.columns if c.startswith("Medical_History_")]
        if med_hist_cols:
            df_feat["Med_History_null_count"] = (
                df_feat[med_hist_cols].isnull().sum(axis=1)
            )

        # Total missing values count
        df_feat["total_null_count"] = df_feat.isnull().sum(axis=1)

        return df_feat

    train_feat = extract_domain_features(train_df)
    val_feat = extract_domain_features(val_df)
    test_feat = extract_domain_features(test_df)

    # 3. Frequency Encodings & Group Aggregations (Fitted strictly on Train)
    cat_cols = [
        "Product_Info_2",
        "Product_Info_2_char",
        "Employment_Info_2",
        "Medical_History_2",
        "Medical_History_10",
    ]
    cat_cols = [c for c in cat_cols if c in train_feat.columns]

    for c in cat_cols:
        freq = train_feat[c].value_counts(normalize=True).to_dict()
        train_feat[f"{c}_freq"] = train_feat[c].map(freq).fillna(0).astype(np.float32)
        val_feat[f"{c}_freq"] = val_feat[c].map(freq).fillna(0).astype(np.float32)
        test_feat[f"{c}_freq"] = test_feat[c].map(freq).fillna(0).astype(np.float32)

    group_stats = train_feat.groupby("Product_Info_2")[
        ["BMI", "Ins_Age", "Med_Keyword_sum"]
    ].agg(["mean", "std"])
    group_stats.columns = ["_".join(col).strip() for col in group_stats.columns.values]
    group_stats = group_stats.reset_index()

    train_feat = train_feat.merge(group_stats, on="Product_Info_2", how="left")
    val_feat = val_feat.merge(group_stats, on="Product_Info_2", how="left")
    test_feat = test_feat.merge(group_stats, on="Product_Info_2", how="left")

    os.makedirs("./working", exist_ok=True)
    train_feat.to_parquet("./working/train_processed.parquet", index=False)
    val_feat.to_parquet("./working/val_processed.parquet", index=False)
    test_feat.to_parquet("./working/test_processed.parquet", index=False)

    return train_feat, val_feat, test_feat


def prepare_features(train_df, val_df, test_df):
    target_col = "Response"
    drop_cols = [c for c in ["Id", "id", target_col] if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    y_train = train_df[target_col].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(np.float32)

    cat_cols = [
        c
        for c in feature_cols
        if str(train_df[c].dtype) in ["category", "object"]
        or c.startswith("Product_Info_2")
        or c in ["Employment_Info_2", "Medical_History_2", "Medical_History_10"]
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    cat_dims = {}
    for c in cat_cols:
        tr_s = X_train[c].astype(str).fillna("missing")
        va_s = X_val[c].astype(str).fillna("missing")
        te_s = X_test[c].astype(str).fillna("missing")

        uniques = list(set(tr_s.unique()))
        val_map = {val: idx for idx, val in enumerate(uniques)}
        unk_idx = len(uniques)

        X_train[c] = tr_s.map(val_map).fillna(unk_idx).astype(int)
        X_val[c] = va_s.map(val_map).fillna(unk_idx).astype(int)
        X_test[c] = te_s.map(val_map).fillna(unk_idx).astype(int)
        cat_dims[c] = unk_idx + 1

    # LightGBM categorical DataFrames
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_val_lgb[c] = X_val_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    # Numerical median imputation fit strictly on train
    num_medians = X_train[num_cols].median()
    X_train_num = X_train.copy()
    X_val_num = X_val.copy()
    X_test_num = X_test.copy()

    X_train_num[num_cols] = X_train_num[num_cols].fillna(num_medians)
    X_val_num[num_cols] = X_val_num[num_cols].fillna(num_medians)
    X_test_num[num_cols] = X_test_num[num_cols].fillna(num_medians)

    return (
        X_train_lgb,
        X_val_lgb,
        X_test_lgb,
        X_train_num,
        X_val_num,
        X_test_num,
        y_train,
        y_val,
        cat_cols,
        num_cols,
        cat_dims,
    )


class GatedTabularNet(nn.Module):
    """Deep Gated Residual Neural Network for Tabular Risk Assessment."""

    def __init__(self, num_features, cat_dims, embed_dim=8, hidden_dim=128):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, embed_dim)
                for _, num_classes in cat_dims.items()
            ]
        )
        total_input_dim = num_features + len(cat_dims) * embed_dim

        self.in_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gate1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gate2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.drop2 = nn.Dropout(0.2)

        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat):
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(x_cat[:, i]))

        x = torch.cat([x_num] + embedded, dim=1) if len(embedded) > 0 else x_num
        h = self.in_proj(x)

        res1 = self.drop1(torch.relu(self.bn1(self.fc1(h))))
        g1 = self.gate1(h)
        h = h + res1 * g1

        res2 = self.drop2(torch.relu(self.bn2(self.fc2(h))))
        g2 = self.gate2(h)
        h = h + res2 * g2

        return self.out_proj(h).squeeze(-1)


def train_pytorch_net(
    X_train_num, X_val_num, X_test_num, y_train, y_val, cat_cols, num_cols, cat_dims
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(X_train_num[num_cols].values).astype(
        np.float32
    )
    val_num_scaled = scaler.transform(X_val_num[num_cols].values).astype(np.float32)
    test_num_scaled = scaler.transform(X_test_num[num_cols].values).astype(np.float32)

    train_cat = X_train_num[cat_cols].values.astype(np.int64)
    val_cat = X_val_num[cat_cols].values.astype(np.int64)
    test_cat = X_test_num[cat_cols].values.astype(np.int64)

    train_ds = TensorDataset(
        torch.tensor(train_num_scaled), torch.tensor(train_cat), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(val_num_scaled), torch.tensor(val_cat), torch.tensor(y_val)
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    model = GatedTabularNet(len(num_cols), cat_dims, embed_dim=8, hidden_dim=128).to(
        device
    )
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b, x_cat_b, y_b = (
                x_num_b.to(device),
                x_cat_b.to(device),
                y_b.to(device),
            )
            optimizer.zero_grad()
            preds = model(x_num_b, x_cat_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num_b, x_cat_b, y_b in val_loader:
                x_num_b, x_cat_b, y_b = (
                    x_num_b.to(device),
                    x_cat_b.to(device),
                    y_b.to(device),
                )
                preds = model(x_num_b, x_cat_b)
                val_loss += criterion(preds, y_b).item() * len(y_b)

        val_loss /= len(y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch+1:02d}/15 - Loss: {running_loss/len(y_train):.4f} - Val Loss: {val_loss:.4f}"
        )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        val_preds = (
            model(
                torch.tensor(val_num_scaled).to(device),
                torch.tensor(val_cat).to(device),
            )
            .cpu()
            .numpy()
        )
        test_preds = (
            model(
                torch.tensor(test_num_scaled).to(device),
                torch.tensor(test_cat).to(device),
            )
            .cpu()
            .numpy()
        )

    return val_preds, test_preds


def main():
    print("Starting Execution...")
    train_df, val_df, test_df = load_and_process_data()
    (
        X_train_lgb,
        X_val_lgb,
        X_test_lgb,
        X_train_num,
        X_val_num,
        X_test_num,
        y_train,
        y_val,
        cat_cols,
        num_cols,
        cat_dims,
    ) = prepare_features(train_df, val_df, test_df)

    # 1. LightGBM Regressor
    print("Training LightGBM Regressor...")
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=30,
        random_state=SEED,
        n_jobs=-1,
    )
    lgb_reg.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_reg_val = lgb_reg.predict(X_val_lgb)
    lgb_reg_test = lgb_reg.predict(X_test_lgb)

    # 2. LightGBM Multi-Class Classifier Expected Value E[Y]
    print("Training LightGBM Multi-class Classifier for Ordinal Expectation...")
    lgb_cls = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    y_train_cls = (y_train - 1).astype(int)
    y_val_cls = (y_val - 1).astype(int)
    lgb_cls.fit(
        X_train_lgb,
        y_train_cls,
        eval_set=[(X_val_lgb, y_val_cls)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    class_weights = np.arange(1, 9)
    lgb_cls_val = np.sum(lgb_cls.predict_proba(X_val_lgb) * class_weights, axis=1)
    lgb_cls_test = np.sum(lgb_cls.predict_proba(X_test_lgb) * class_weights, axis=1)

    # 3. CatBoost Regressor
    print("Training CatBoost Regressor...")
    cat_reg = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_reg.fit(X_train_num, y_train, eval_set=(X_val_num, y_val), verbose=False)
    cat_val = cat_reg.predict(X_val_num)
    cat_test = cat_reg.predict(X_test_num)

    # 4. XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_reg.fit(X_train_num, y_train, eval_set=[(X_val_num, y_val)], verbose=False)
    xgb_val = xgb_reg.predict(X_val_num)
    xgb_test = xgb_reg.predict(X_test_num)

    # 5. PyTorch Deep GatedTabularNet
    print("Training PyTorch GatedTabularNet...")
    nn_val, nn_test = train_pytorch_net(
        X_train_num, X_val_num, X_test_num, y_train, y_val, cat_cols, num_cols, cat_dims
    )

    # 6. Continuous Model Blending via SLSQP
    val_preds_matrix = [lgb_reg_val, lgb_cls_val, cat_val, xgb_val, nn_val]
    test_preds_matrix = [lgb_reg_test, lgb_cls_test, cat_test, xgb_test, nn_test]

    def blend_loss(w):
        w_norm = np.array(w) / np.sum(w)
        pred = sum(wi * pi for wi, pi in zip(w_norm, val_preds_matrix))
        return np.mean((y_val - pred) ** 2)

    init_weights = [0.30, 0.20, 0.20, 0.20, 0.10]
    opt_w_res = minimize(
        blend_loss, x0=init_weights, bounds=[(0, 1)] * 5, method="SLSQP"
    )
    opt_w = opt_w_res.x / np.sum(opt_w_res.x)

    val_blend = sum(wi * pi for wi, pi in zip(opt_w, val_preds_matrix))
    test_blend = sum(wi * pi for wi, pi in zip(opt_w, test_preds_matrix))

    # 7. Multi-Start Quantile-Initialized Powell Threshold Optimization
    class_counts = np.bincount(y_train.astype(int))[1:9]
    cum_probs = np.cumsum(class_counts) / len(y_train)
    init_thresholds = np.quantile(val_blend, cum_probs[:-1])

    def qwk_objective(thresholds, preds, targets):
        if not np.all(np.diff(thresholds) > 0):
            return 1e5
        discrete_preds = np.digitize(preds, thresholds) + 1
        return -evaluate_qwk(targets, discrete_preds)

    best_score = -1.0
    best_thresholds = None

    for search_idx in range(10):
        if search_idx == 0:
            x0 = init_thresholds.copy()
        else:
            x0 = init_thresholds + np.random.normal(0, 0.05, size=len(init_thresholds))
            x0 = np.sort(x0)

        res = minimize(
            qwk_objective,
            x0=x0,
            args=(val_blend, y_val),
            method="Powell",
            options={"maxiter": 1000},
        )
        score = -res.fun
        if score > best_score:
            best_score = score
            best_thresholds = res.x

    # 8. Model Inference for Validation & Test
    val_discrete_preds = np.digitize(val_blend, best_thresholds) + 1
    final_val_qwk = evaluate_qwk(y_val, val_discrete_preds)

    test_discrete_preds = np.digitize(test_blend, best_thresholds) + 1

    # 9. Format Final Submission
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame(
        {
            "id": np.arange(len(test_discrete_preds)),
            "Response": test_discrete_preds.astype(int),
        }
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_qwk:.5f}")


if __name__ == "__main__":
    main()
