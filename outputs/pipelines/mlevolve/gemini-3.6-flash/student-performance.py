import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------
# 1. Load Raw Data
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path).reset_index(drop=True)
test_df = pd.read_parquet(test_path).reset_index(drop=True)

target_col = "G3"
has_target = target_col in train_df.columns
y_train_full = train_df[target_col].values if has_target else None

train_features = train_df.drop(columns=[target_col]) if has_target else train_df.copy()
test_features = test_df.copy()

# Binary mapping dictionary
binary_cols_map = {
    "school": {"GP": 1, "MS": 0},
    "sex": {"F": 1, "M": 0},
    "address": {"U": 1, "R": 0},
    "famsize": {"LE3": 1, "GT3": 0},
    "Pstatus": {"T": 1, "A": 0},
    "schoolsup": {"yes": 1, "no": 0},
    "famsup": {"yes": 1, "no": 0},
    "paid": {"yes": 1, "no": 0},
    "activities": {"yes": 1, "no": 0},
    "nursery": {"yes": 1, "no": 0},
    "higher": {"yes": 1, "no": 0},
    "internet": {"yes": 1, "no": 0},
    "romantic": {"yes": 1, "no": 0},
}


def clean_binary_columns(df):
    df = df.copy()
    for col, mapping in binary_cols_map.items():
        if col in df.columns:
            if df[col].dtype == object or df[col].dtype == "string":
                df[col] = df[col].map(mapping).fillna(0).astype(int)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)
    return df


train_features = clean_binary_columns(train_features)
test_features = clean_binary_columns(test_features)

# Impute Missing Values safely using train-fit statistics
num_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train_features.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

for col in num_cols:
    median_val = train_features[col].median()
    train_features[col] = train_features[col].fillna(median_val)
    test_features[col] = test_features[col].fillna(median_val)

for col in cat_cols:
    mode_val = (
        train_features[col].mode()[0]
        if not train_features[col].mode().empty
        else "missing"
    )
    train_features[col] = train_features[col].fillna(mode_val)
    test_features[col] = test_features[col].fillna(mode_val)


# ---------------------------------------------------------
# 2. Domain-Specific Feature Engineering
# ---------------------------------------------------------
def create_domain_features(df):
    df = df.copy()

    # Academic Trajectory & Progression Dynamics
    if "G1" in df.columns and "G2" in df.columns:
        df["G_mean"] = (df["G1"] + df["G2"]) / 2.0
        df["G_diff"] = df["G2"] - df["G1"]
        df["G_ratio"] = (df["G2"] + 1.0) / (df["G1"] + 1.0)
        df["G1_zero"] = (df["G1"] == 0).astype(int)
        df["G2_zero"] = (df["G2"] == 0).astype(int)
        df["G_zero_count"] = df["G1_zero"] + df["G2_zero"]
        df["G_prod"] = df["G1"] * df["G2"]
        df["G_acceleration"] = df["G2"] - df["G1"]
        df["G2_sq"] = df["G2"] ** 2
        df["G2_is_zero"] = (df["G2"] == 0).astype(int)
        df["both_zero"] = ((df["G1"] == 0) & (df["G2"] == 0)).astype(int)
        df["weighted_trend"] = 1.5 * df["G2"] - 0.5 * df["G1"]
        if "failures" in df.columns:
            df["failure_g2_ratio"] = df["failures"] / (df["G2"] + 1.0)
        if "absences" in df.columns and "failures" in df.columns:
            df["absence_risk_score"] = (
                df["absences"] * df["failures"] / (df["G2"] + 1.0)
            )

    # Parental Socioeconomic Synergy
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["Pedu_max"] = np.maximum(df["Medu"], df["Fedu"])
        df["Pedu_min"] = np.minimum(df["Medu"], df["Fedu"])
        df["Pedu_mean"] = (df["Medu"] + df["Fedu"]) / 2.0
        df["Pedu_diff"] = df["Medu"] - df["Fedu"]
        df["Pedu_both_higher"] = ((df["Medu"] >= 3) & (df["Fedu"] >= 3)).astype(int)
        df["Pedu_both_none"] = ((df["Medu"] == 0) & (df["Fedu"] == 0)).astype(int)

    # Social & Lifestyle Balance Indices
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["alc_total"] = df["Dalc"] + df["Walc"]
        df["alc_weighted"] = df["Dalc"] * 5.0 + df["Walc"] * 2.0
        df["alc_weekend_ratio"] = df["Walc"] / (df["Dalc"] + 1e-5)
        df["alc_heavy"] = (df["alc_total"] >= 6).astype(int)

    if "studytime" in df.columns and "goout" in df.columns and "freetime" in df.columns:
        df["study_to_social_ratio"] = df["studytime"] / (
            df["goout"] + df["freetime"] + 1e-5
        )
        df["social_total"] = df["goout"] + df["freetime"]
        df["study_minus_goout"] = df["studytime"] - df["goout"]

    # Academic Support Index
    support_cols = [
        c
        for c in ["schoolsup", "famsup", "paid", "internet", "higher"]
        if c in df.columns
    ]
    if support_cols:
        df["support_score"] = df[support_cols].sum(axis=1)

    # Failures & Academic Risk Features
    if "failures" in df.columns:
        df["has_failures"] = (df["failures"] > 0).astype(int)
        if "studytime" in df.columns:
            df["failures_per_study"] = df["failures"] / (df["studytime"] + 0.1)
        if "G1" in df.columns and "G2" in df.columns:
            df["G_per_failure"] = (df["G1"] + df["G2"]) / (df["failures"] + 1.0)

    # Absences & Health Impact
    if "absences" in df.columns:
        df["absences_log"] = np.log1p(np.maximum(0, df["absences"]))
        df["absences_chronic"] = (df["absences"] > 10).astype(int)
        if "health" in df.columns:
            df["health_absences_ratio"] = df["absences"] / (df["health"] + 0.1)

    # Overall Well-being Score
    if (
        "famrel" in df.columns
        and "health" in df.columns
        and "Dalc" in df.columns
        and "Walc" in df.columns
    ):
        df["wellbeing_score"] = df["famrel"] + df["health"] - (df["Dalc"] + df["Walc"])

    return df


train_fe = create_domain_features(train_features)
test_fe = create_domain_features(test_features)

# Frequency Encoding for Nominal Categoricals
nominal_cols = ["Mjob", "Fjob", "reason", "guardian"]
nominal_cols = [c for c in nominal_cols if c in train_fe.columns]

for col in nominal_cols:
    freq_map = train_fe[col].value_counts(normalize=True).to_dict()
    train_fe[f"{col}_freq"] = train_fe[col].map(freq_map).fillna(0)
    test_fe[f"{col}_freq"] = test_fe[col].map(freq_map).fillna(0)

# One-Hot Encoding
train_fe = pd.get_dummies(train_fe, columns=nominal_cols, drop_first=True)
test_fe = pd.get_dummies(test_fe, columns=nominal_cols, drop_first=True)
train_fe, test_fe = train_fe.align(test_fe, join="left", axis=1, fill_value=0)

# Out-of-Fold Target Encoding
if nominal_cols and has_target:
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in ["Mjob", "Fjob", "reason", "guardian"]:
        if col in train_features.columns:
            te_col_name = f"{col}_te"
            train_fe[te_col_name] = 0.0
            test_fe[te_col_name] = 0.0

            for tr_idx, val_idx in kf_te.split(train_features):
                tr_cat = train_features.iloc[tr_idx][col]
                tr_y = y_train_full[tr_idx]
                cat_means = (
                    pd.DataFrame({col: tr_cat, "target": tr_y})
                    .groupby(col)["target"]
                    .mean()
                    .to_dict()
                )
                global_mean = tr_y.mean()

                val_cat = train_features.iloc[val_idx][col]
                train_fe.loc[val_idx, te_col_name] = val_cat.map(cat_means).fillna(
                    global_mean
                )

            full_cat_means = (
                pd.DataFrame({col: train_features[col], "target": y_train_full})
                .groupby(col)["target"]
                .mean()
                .to_dict()
            )
            global_full_mean = y_train_full.mean()
            test_fe[te_col_name] = (
                test_features[col].map(full_cat_means).fillna(global_full_mean)
            )

# Add 5-Fold Stratified CV indicators
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins = (
    pd.qcut(y_train_full, q=5, labels=False, duplicates="drop")
    if has_target
    else np.zeros(len(train_fe))
)

folds = np.zeros(len(train_fe), dtype=int)
for fold_idx, (_, val_idx) in enumerate(kf.split(train_fe, target_bins)):
    folds[val_idx] = fold_idx

train_fe["fold"] = folds
if has_target:
    train_fe["G3"] = y_train_full

# Save processed datasets for auditing
train_fe.to_parquet("./working/train_processed.parquet", index=False)
test_fe.to_parquet("./working/test_processed.parquet", index=False)

# ---------------------------------------------------------
# 3. Model Architecture & Cross-Validation Ensemble
# ---------------------------------------------------------
feature_cols = [c for c in train_fe.columns if c not in [target_col, "fold"]]
X_train = train_fe[feature_cols].values
y_train = train_fe[target_col].values
folds = train_fe["fold"].values
X_test = test_fe[feature_cols].values

n_train = len(train_fe)
n_test = len(test_fe)
n_folds = 5

try:
    from catboost import CatBoostRegressor

    cb_available = True
except ImportError:
    cb_available = False


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn0 = nn.BatchNorm1d(hidden_dim)
        self.act0 = nn.SiLU()

        self.block1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1_act1 = nn.SiLU()
        self.block1_drop = nn.Dropout(dropout)
        self.block1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn2 = nn.BatchNorm1d(hidden_dim)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.act0(self.bn0(self.input_layer(x)))
        residual = h
        out = self.block1_drop(self.block1_act1(self.block1_bn1(self.block1_fc1(h))))
        out = self.block1_bn2(self.block1_fc2(out))
        h = torch.relu(out + residual)
        return self.head(h).squeeze(-1)


def train_pytorch_fold(
    X_tr, y_tr, X_va, y_va, X_te, fold_idx, epochs=90, batch_size=32, lr=2e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

    ds_tr = TensorDataset(X_tr_t, y_tr_t)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularResNet(X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_model_path = f"./working/best_nn_fold_{fold_idx}.pt"

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t.to(device))
            val_loss = criterion(val_preds, y_va_t.to(device)).item()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        best_va_preds = model(X_va_t.to(device)).cpu().numpy()
        best_te_preds = model(X_te_t.to(device)).cpu().numpy()

    return best_va_preds, best_te_preds


num_models = 5
oof_matrix = np.zeros((n_train, num_models))
test_matrix = np.zeros((n_test, num_models))

for fold in range(n_folds):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    # Model 0: LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    oof_matrix[va_idx, 0] = model_lgb.predict(X_va)
    test_matrix[:, 0] += model_lgb.predict(X_test) / n_folds

    # Model 1: XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_matrix[va_idx, 1] = model_xgb.predict(X_va)
    test_matrix[:, 1] += model_xgb.predict(X_test) / n_folds

    # Model 2: CatBoost or ExtraTrees
    if cb_available:
        model_cb = CatBoostRegressor(
            iterations=800,
            learning_rate=0.03,
            depth=4,
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=30,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_matrix[va_idx, 2] = model_cb.predict(X_va)
        test_matrix[:, 2] += model_cb.predict(X_test) / n_folds
    else:
        model_et = ExtraTreesRegressor(
            n_estimators=250, max_depth=6, random_state=42 + fold, n_jobs=-1
        )
        model_et.fit(X_tr, y_tr)
        oof_matrix[va_idx, 2] = model_et.predict(X_va)
        test_matrix[:, 2] += model_et.predict(X_test) / n_folds

    # Model 3: Standardized Ridge Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_test)

    model_ridge = Ridge(alpha=15.0)
    model_ridge.fit(X_tr_s, y_tr)
    oof_matrix[va_idx, 3] = model_ridge.predict(X_va_s)
    test_matrix[:, 3] += model_ridge.predict(X_te_s) / n_folds

    # Model 4: PyTorch Tabular ResNet
    nn_va_preds, nn_te_preds = train_pytorch_fold(X_tr, y_tr, X_va, y_va, X_test, fold)
    oof_matrix[va_idx, 4] = nn_va_preds
    test_matrix[:, 4] += nn_te_preds / n_folds


# ---------------------------------------------------------
# 4. SLSQP Optimization & Submission Generation
# ---------------------------------------------------------
g2_train_zero_mask = (
    (train_fe["G2"] == 0).values
    if "G2" in train_fe.columns
    else np.zeros(len(train_fe), dtype=bool)
)
g2_test_zero_mask = (
    (test_fe["G2"] == 0).values
    if "G2" in test_fe.columns
    else np.zeros(len(test_fe), dtype=bool)
)


def apply_post_processing(preds, is_zero_mask):
    p = np.clip(preds, 0.0, 20.0)
    p[is_zero_mask] = 0.0
    return p


def blend_objective(weights):
    w = np.array(weights)
    pred = apply_post_processing(oof_matrix @ w, g2_train_zero_mask)
    return root_mean_squared_error(y_train, pred)


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x

final_oof_preds = apply_post_processing(
    oof_matrix @ best_weights, g2_train_zero_mask
)
final_test_preds = apply_post_processing(
    test_matrix @ best_weights, g2_test_zero_mask
)

final_score = root_mean_squared_error(y_train, final_oof_preds)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "G3": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
