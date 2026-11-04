import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

# ------------------------------------------------------------------------------
# 0. Global Setup & Seed
# ------------------------------------------------------------------------------
SEED = 42
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------------------
# 1. Load Data & Target Standardization
# ------------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
if target_col in train_df.columns:
    unique_targets = train_df[target_col].unique()
    if set(unique_targets) == {"good", "bad"}:
        target_map = {"good": 0, "bad": 1}
    elif set(unique_targets) == {"1", "2"}:
        target_map = {"1": 0, "2": 1}
    elif set(unique_targets) == {1, 2}:
        target_map = {1: 0, 2: 1}
    else:
        target_map = {val: idx for idx, val in enumerate(sorted(unique_targets))}

    train_df["target"] = train_df[target_col].map(target_map).astype(int)
else:
    raise ValueError("Target column 'class' not found in training data.")


# ------------------------------------------------------------------------------
# 2. Domain Feature Engineering
# ------------------------------------------------------------------------------
def engineer_domain_features(df):
    df_feat = df.copy()
    col_map = {col.lower().replace(" ", "_"): col for col in df_feat.columns}

    def get_col(candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
            for col in df_feat.columns:
                if c in col.lower():
                    return col
        return None

    duration_col = get_col(["duration", "attribute2"])
    credit_amount_col = get_col(
        ["credit_amount", "creditamount", "amount", "attribute5"]
    )
    installment_col = get_col(["installment", "rate", "attribute8"])
    age_col = get_col(["age", "attribute13"])
    existing_credits_col = get_col(["existing_credits", "credits", "attribute16"])
    dependents_col = get_col(["dependents", "liable", "attribute18"])

    if duration_col and credit_amount_col:
        df_feat["feat_monthly_payment"] = df_feat[credit_amount_col] / (
            df_feat[duration_col] + 1e-5
        )

        if installment_col:
            df_feat["feat_est_disposable_income"] = df_feat["feat_monthly_payment"] / (
                df_feat[installment_col] / 100.0 + 1e-5
            )
            df_feat["feat_net_remaining_income"] = (
                df_feat["feat_est_disposable_income"] - df_feat["feat_monthly_payment"]
            )

    if credit_amount_col and age_col:
        df_feat["feat_credit_to_age"] = df_feat[credit_amount_col] / (
            df_feat[age_col] + 1e-5
        )

    if age_col and "feat_monthly_payment" in df_feat.columns:
        df_feat["feat_monthly_to_age"] = df_feat["feat_monthly_payment"] / (
            df_feat[age_col] + 1e-5
        )

    if credit_amount_col and existing_credits_col:
        df_feat["feat_credit_per_existing_loan"] = df_feat[credit_amount_col] / (
            df_feat[existing_credits_col] + 1e-5
        )

    if dependents_col and "feat_est_disposable_income" in df_feat.columns:
        df_feat["feat_income_per_dependent"] = df_feat["feat_est_disposable_income"] / (
            df_feat[dependents_col] + 1e-5
        )

    checking_col = get_col(["checking", "status_of_existing_checking", "attribute1"])
    if checking_col and not pd.api.types.is_numeric_dtype(df_feat[checking_col]):
        s = df_feat[checking_col].astype(str).str.lower()
        df_feat["feat_checking_ordinal"] = np.select(
            [s.str.contains("no checking"), s.str.contains("< 0") | s.str.contains("<0"), s.str.contains("200")],
            [-1, 0, 2],
            default=1
        )

    savings_col = get_col(["savings", "status_of_savings", "attribute6"])
    if savings_col and not pd.api.types.is_numeric_dtype(df_feat[savings_col]):
        s = df_feat[savings_col].astype(str).str.lower()
        df_feat["feat_savings_ordinal"] = np.select(
            [s.str.contains("unknown") | s.str.contains("no savings"), s.str.contains("< 100") | s.str.contains("<100"), s.str.contains(">= 1000") | s.str.contains(">=1000")],
            [-1, 0, 3],
            default=1
        )

    employment_col = get_col(["employment", "present_employment", "attribute7"])
    if employment_col and not pd.api.types.is_numeric_dtype(df_feat[employment_col]):
        s = df_feat[employment_col].astype(str).str.lower()
        df_feat["feat_employment_ordinal"] = np.select(
            [s.str.contains("unemployed"), s.str.contains("< 1 year") | s.str.contains("<1 year"), s.str.contains(">= 7") | s.str.contains(">=7")],
            [0, 1, 4],
            default=2
        )

    return df_feat


train_processed = engineer_domain_features(train_df)
test_processed = engineer_domain_features(test_df)

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_processed["fold"] = -1
for fold, (tr_idx, val_idx) in enumerate(
    skf.split(train_processed, train_processed["target"])
):
    train_processed.loc[val_idx, "fold"] = fold

cat_cols = [
    col
    for col in train_processed.columns
    if not pd.api.types.is_numeric_dtype(train_processed[col])
    and col not in ["class", "target", "id"]
]

credit_col = [
    c for c in train_processed.columns if "credit_amount" in c or "amount" in c
]
credit_col = credit_col[0] if credit_col else None

if credit_col:
    train_processed["feat_credit_to_purpose_median"] = 1.0
    test_processed["feat_credit_to_purpose_median"] = 1.0

    purpose_col = [c for c in cat_cols if "purpose" in c or "attribute4" in c.lower()]
    if purpose_col:
        p_col = purpose_col[0]

        global_medians = train_processed.groupby(p_col)[credit_col].median()
        test_medians = test_processed[p_col].map(global_medians).fillna(1e-5)
        test_processed["feat_credit_to_purpose_median"] = test_processed[credit_col] / (
            test_medians + 1e-5
        )

        for fold in range(5):
            tr_mask = train_processed["fold"] != fold
            va_mask = train_processed["fold"] == fold

            fold_tr = train_processed[tr_mask]
            fold_medians = fold_tr.groupby(p_col)[credit_col].median()

            va_medians = (
                train_processed.loc[va_mask, p_col].map(fold_medians).fillna(1e-5)
            )
            train_processed.loc[va_mask, "feat_credit_to_purpose_median"] = (
                train_processed.loc[va_mask, credit_col] / (va_medians + 1e-5)
            )


# ------------------------------------------------------------------------------
# 3. Model Architecture Definition (PyTorch Tabular ResNet)
# ------------------------------------------------------------------------------
class TabularResNet(nn.Module):

    def __init__(self, num_features: int, cat_dims: list = None, hidden_dim: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList()
        emb_out_dim = 0
        if cat_dims:
            for num_classes in cat_dims:
                emb_dim = max(2, min(30, (num_classes + 1) // 2))
                self.embeddings.append(nn.Embedding(num_classes + 1, emb_dim))
                emb_out_dim += emb_dim

        self.emb_drop = nn.Dropout(0.1)
        in_dim = num_features + emb_out_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.head = nn.Linear(hidden_dim, 1)
        self.act = nn.SiLU()

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        if x_cat is not None and len(self.embeddings) > 0:
            emb_list = [
                emb(x_cat[:, i].clamp(min=0)) for i, emb in enumerate(self.embeddings)
            ]
            x_emb = torch.cat(emb_list, dim=1)
            x_emb = self.emb_drop(x_emb)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        h = self.input_layer(x)
        h = self.act(h + self.block1(h))
        return self.head(h).squeeze(-1)


# ------------------------------------------------------------------------------
# 4. Training & Validation Setup
# ------------------------------------------------------------------------------
ignore_cols = ["class", "target", "fold", "id"]
feature_cols = [c for c in train_processed.columns if c not in ignore_cols]
num_cols = [
    c for c in feature_cols if pd.api.types.is_numeric_dtype(train_processed[c])
]
cat_cols = [c for c in feature_cols if c not in num_cols]

model_names = ["lgb", "xgb", "hist_gb", "extra_trees", "nn"]
n_train = len(train_processed)
n_test = len(test_processed)
n_folds = 5

oof_preds = {m: np.zeros(n_train) for m in model_names}
test_preds = {m: np.zeros(n_test) for m in model_names}

pos_weight = (len(train_processed) - train_processed["target"].sum()) / (
    train_processed["target"].sum() + 1e-5
)

for fold in range(n_folds):
    tr_idx = train_processed["fold"] != fold
    va_idx = train_processed["fold"] == fold

    X_tr_raw = train_processed.loc[tr_idx, feature_cols].copy()
    y_tr = train_processed.loc[tr_idx, "target"].values

    X_va_raw = train_processed.loc[va_idx, feature_cols].copy()
    y_va = train_processed.loc[va_idx, "target"].values

    X_te_raw = test_processed[feature_cols].copy()

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    if cat_cols:
        X_tr_cat = encoder.fit_transform(X_tr_raw[cat_cols].astype(str)) + 1
        X_va_cat = encoder.transform(X_va_raw[cat_cols].astype(str)) + 1
        X_te_cat = encoder.transform(X_te_raw[cat_cols].astype(str)) + 1

        cat_dims = [
            int(max(X_tr_cat[:, i].max(), X_va_cat[:, i].max(), X_te_cat[:, i].max()))
            + 1
            for i in range(len(cat_cols))
        ]
    else:
        X_tr_cat, X_va_cat, X_te_cat = None, None, None
        cat_dims = []

    X_tr_num = X_tr_raw[num_cols].values.astype(np.float32)
    X_va_num = X_va_raw[num_cols].values.astype(np.float32)
    X_te_num = X_te_raw[num_cols].values.astype(np.float32)

    num_imputer = SimpleImputer(strategy="median")
    X_tr_num = num_imputer.fit_transform(X_tr_num)
    X_va_num = num_imputer.transform(X_va_num)
    X_te_num = num_imputer.transform(X_te_num)

    scaler = StandardScaler()
    X_tr_num_scaled = scaler.fit_transform(X_tr_num)
    X_va_num_scaled = scaler.transform(X_va_num)
    X_te_num_scaled = scaler.transform(X_te_num)

    if cat_cols:
        X_tr_tree = np.hstack([X_tr_num, X_tr_cat])
        X_va_tree = np.hstack([X_va_num, X_va_cat])
        X_te_tree = np.hstack([X_te_num, X_te_cat])
    else:
        X_tr_tree, X_va_tree, X_te_tree = X_tr_num, X_va_num, X_te_num

    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_tree,
        y_tr,
        eval_set=[(X_va_tree, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds["lgb"][va_idx] = lgb_model.predict_proba(X_va_tree)[:, 1]
    test_preds["lgb"] += lgb_model.predict_proba(X_te_tree)[:, 1] / n_folds

    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=pos_weight,
        random_state=SEED + fold,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr_tree, y_tr, eval_set=[(X_va_tree, y_va)], verbose=False)
    oof_preds["xgb"][va_idx] = xgb_model.predict_proba(X_va_tree)[:, 1]
    test_preds["xgb"] += xgb_model.predict_proba(X_te_tree)[:, 1] / n_folds

    # Model 3: HistGradientBoosting
    hist_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.025,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED + fold,
        early_stopping=True,
        n_iter_no_change=30,
        scoring="roc_auc",
    )
    hist_model.fit(X_tr_tree, y_tr)
    oof_preds["hist_gb"][va_idx] = hist_model.predict_proba(X_va_tree)[:, 1]
    test_preds["hist_gb"] += hist_model.predict_proba(X_te_tree)[:, 1] / n_folds

    # Model 4: ExtraTrees
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr_tree, y_tr)
    oof_preds["extra_trees"][va_idx] = et_model.predict_proba(X_va_tree)[:, 1]
    test_preds["extra_trees"] += et_model.predict_proba(X_te_tree)[:, 1] / n_folds

    # Model 5: PyTorch Tabular ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = TabularResNet(
        num_features=X_tr_num_scaled.shape[1],
        cat_dims=cat_dims,
        hidden_dim=64,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = optim.AdamW(nn_model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    tr_tensor_num = torch.tensor(X_tr_num_scaled, dtype=torch.float32)
    tr_tensor_cat = torch.tensor(X_tr_cat, dtype=torch.long) if cat_cols else None
    tr_y_tensor = torch.tensor(y_tr, dtype=torch.float32)

    va_tensor_num = torch.tensor(X_va_num_scaled, dtype=torch.float32).to(device)
    va_tensor_cat = (
        torch.tensor(X_va_cat, dtype=torch.long).to(device) if cat_cols else None
    )

    te_tensor_num = torch.tensor(X_te_num_scaled, dtype=torch.float32).to(device)
    te_tensor_cat = (
        torch.tensor(X_te_cat, dtype=torch.long).to(device) if cat_cols else None
    )

    if cat_cols:
        dataset = TensorDataset(tr_tensor_num, tr_tensor_cat, tr_y_tensor)
    else:
        dataset = TensorDataset(tr_tensor_num, tr_y_tensor)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_va_loss = float("inf")
    best_va_preds = None
    best_te_preds = None

    for epoch in range(80):
        nn_model.train()
        for batch in loader:
            optimizer.zero_grad()
            if cat_cols:
                bx_num, bx_cat, by = (
                    batch[0].to(device),
                    batch[1].to(device),
                    batch[2].to(device),
                )
                logits = nn_model(bx_num, bx_cat)
            else:
                bx_num, by = batch[0].to(device), batch[1].to(device)
                logits = nn_model(bx_num)

            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            va_logits = nn_model(va_tensor_num, va_tensor_cat)
            va_loss = criterion(
                va_logits, torch.tensor(y_va, dtype=torch.float32).to(device)
            ).item()
            va_probs = torch.sigmoid(va_logits).cpu().numpy()

            if va_loss < best_va_loss:
                best_va_loss = va_loss
                best_va_preds = va_probs
                te_logits = nn_model(te_tensor_num, te_tensor_cat)
                best_te_preds = torch.sigmoid(te_logits).cpu().numpy()

    oof_preds["nn"][va_idx] = best_va_preds
    test_preds["nn"] += best_te_preds / n_folds

# ------------------------------------------------------------------------------
# 5. Ensemble Optimization & Final Submission
# ------------------------------------------------------------------------------
y_true = train_processed["target"].values


def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(weights, 0)
    if weights.sum() > 0:
        weights /= weights.sum()
    blend = sum(weights[i] * oof_preds[m] for i, m in enumerate(model_names))
    return -roc_auc_score(y_true, blend)


init_weights = [1.0 / len(model_names)] * len(model_names)
bounds = [(0, 1)] * len(model_names)
res = minimize(loss_func, init_weights, bounds=bounds, method="SLSQP")

best_weights = np.maximum(res.x, 0)
best_weights /= best_weights.sum()

oof_blend = sum(best_weights[i] * oof_preds[m] for i, m in enumerate(model_names))
final_oof_auc = roc_auc_score(y_true, oof_blend)

test_blend = sum(best_weights[i] * test_preds[m] for i, m in enumerate(model_names))

submission_df = pd.DataFrame({"id": np.arange(len(test_blend)), "class": test_blend})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_oof_auc:.5f}")
