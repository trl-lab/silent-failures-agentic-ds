import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define file paths
input_dir = "./input"
working_dir = "./working"
os.makedirs(working_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

# 1. Load Raw Data
print("Loading dataset...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "IMU Score"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

# Standardize Row ID
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

# Combine datasets temporarily for structural feature engineering
y_train = train_df[target_col].values.astype(np.float32)
train_df_no_target = train_df.drop(columns=[target_col])
full_df = pd.concat([train_df_no_target, test_df], axis=0, ignore_index=True)

# 2. Identify Column Modalities
feature_cols = [c for c in full_df.columns if c != "id"]
num_cols = []
cat_cols = []

for c in feature_cols:
    if pd.api.types.is_numeric_dtype(full_df[c]):
        if full_df[c].nunique() < 10 and full_df[c].dtype == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        cat_cols.append(c)

print(
    f"Identified {len(num_cols)} numeric features and {len(cat_cols)} categorical features."
)

# 3. Missingness Indicators & Numeric Preprocessing
processed_df = pd.DataFrame({"id": full_df["id"]})

train_len = len(train_df)

for col in num_cols:
    if full_df[col].isna().sum() > 0:
        processed_df[f"{col}_isna"] = full_df[col].isna().astype(np.int8)

    median_val = full_df[col].iloc[:train_len].median()
    processed_df[col] = full_df[col].fillna(median_val)

# Log-transform highly skewed continuous features
for col in num_cols:
    skewness = full_df[col].iloc[:train_len].skew()
    if skewness > 1.5 and (processed_df[col] >= 0).all():
        processed_df[f"{col}_log1p"] = np.log1p(processed_df[col])

# 4. Pairwise Interaction Ratios
if len(num_cols) >= 2:
    top_num_cols = num_cols[: min(8, len(num_cols))]
    for i in range(len(top_num_cols)):
        for j in range(i + 1, len(top_num_cols)):
            col1, col2 = top_num_cols[i], top_num_cols[j]
            processed_df[f"{col1}_mult_{col2}"] = (
                processed_df[col1] * processed_df[col2]
            )
            processed_df[f"{col1}_ratio_{col2}"] = processed_df[col1] / (
                processed_df[col2] + 1e-5
            )

# 5. Group Aggregations
group_cols = [c for c in cat_cols if 2 <= full_df[c].iloc[:train_len].nunique() <= 500]

for g_col in group_cols:
    str_group = full_df[g_col].astype(str).fillna("Missing")
    str_group_tr = str_group.iloc[:train_len]
    for n_col in num_cols[: min(5, len(num_cols))]:
        temp_tr_df = pd.DataFrame({"g": str_group_tr, "n": processed_df[n_col].iloc[:train_len]})
        grouped = temp_tr_df.groupby("g")["n"].agg(["mean", "std"]).reset_index()
        grouped.columns = ["g", f"{n_col}_mean_by_{g_col}", f"{n_col}_std_by_{g_col}"]

        merged = str_group.to_frame("g").merge(grouped, on="g", how="left")
        tr_mean = processed_df[n_col].iloc[:train_len].mean()
        processed_df[f"{n_col}_mean_by_{g_col}"] = merged[
            f"{n_col}_mean_by_{g_col}"
        ].fillna(tr_mean)
        processed_df[f"{n_col}_std_by_{g_col}"] = merged[
            f"{n_col}_std_by_{g_col}"
        ].fillna(0.0)

        processed_df[f"{n_col}_diff_from_{g_col}_mean"] = (
            processed_df[n_col] - processed_df[f"{n_col}_mean_by_{g_col}"]
        )

# 6. Categorical Frequency & Categorical Codes
for c in cat_cols:
    cat_str = full_df[c].astype(str).fillna("Missing")
    freq_map = cat_str.iloc[:train_len].value_counts(normalize=True).to_dict()
    processed_df[f"{c}_freq"] = cat_str.map(freq_map).fillna(0.0).astype(np.float32)
    processed_df[c] = cat_str.astype("category")

# Separate into Train and Test subsets
train_processed = processed_df.iloc[: len(train_df)].copy()
test_processed = processed_df.iloc[len(train_df) :].copy()

train_processed[target_col] = y_train

# 7. 5-Fold Cross-Validation Setup & Leak-Free OOF Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
train_processed["fold"] = -1

for fold, (_, val_idx) in enumerate(kf.split(train_processed)):
    train_processed.loc[val_idx, "fold"] = fold

smoothing_m = 10.0
global_mean = y_train.mean()

for c in cat_cols:
    te_col = f"{c}_te"
    train_processed[te_col] = global_mean
    test_processed[te_col] = global_mean

    for fold in range(5):
        tr_mask = train_processed["fold"] != fold
        val_mask = train_processed["fold"] == fold

        tr_fold = train_processed[tr_mask]
        stats = tr_fold.groupby(c, observed=False)[target_col].agg(["count", "mean"])

        smoothed = (stats["count"] * stats["mean"] + smoothing_m * global_mean) / (
            stats["count"] + smoothing_m
        )
        te_map = smoothed.to_dict()

        train_processed.loc[val_mask, te_col] = (
            train_processed.loc[val_mask, c]
            .astype(str)
            .map(te_map)
            .fillna(global_mean)
            .astype(np.float32)
        )

    full_stats = train_processed.groupby(c, observed=False)[target_col].agg(
        ["count", "mean"]
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing_m * global_mean
    ) / (full_stats["count"] + smoothing_m)
    test_processed[te_col] = (
        test_processed[c]
        .astype(str)
        .map(full_smoothed.to_dict())
        .fillna(global_mean)
        .astype(np.float32)
    )

# Prepare clean float32 matrix representations for GBDT and PyTorch
ignore_cols = ["id", target_col, "fold"]
model_feature_cols = [c for c in train_processed.columns if c not in ignore_cols]

X_train_df = train_processed[model_feature_cols].copy()
X_test_df = test_processed[model_feature_cols].copy()

for col in model_feature_cols:
    if X_train_df[col].dtype.name == "category" or X_train_df[col].dtype == "object":
        if X_train_df[col].dtype.name == "category":
            X_train_df[col] = X_train_df[col].cat.codes.astype(np.float32)
            X_test_df[col] = X_test_df[col].cat.codes.astype(np.float32)
        else:
            cat_type = pd.CategoricalDtype(categories=full_df[col].astype(str).unique())
            X_train_df[col] = X_train_df[col].astype(cat_type).cat.codes.astype(np.float32)
            X_test_df[col] = X_test_df[col].astype(cat_type).cat.codes.astype(np.float32)
    else:
        X_train_df[col] = X_train_df[col].astype(np.float32)
        X_test_df[col] = X_test_df[col].astype(np.float32)

train_medians = X_train_df.median()
X_train_mat = X_train_df.fillna(train_medians).values.astype(np.float32)
X_test_mat = X_test_df.fillna(train_medians).values.astype(np.float32)

num_samples, num_features = X_train_mat.shape
print(
    f"Preprocessed Dataset Ready. Train Samples: {num_samples}, Test Samples: {len(X_test_mat)}, Features: {num_features}"
)


# -------------------------------------------------------------------------
# PyTorch Neural Architecture: Gated Residual Network (TabularGRN)
# -------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        out = self.ln(res + g * h)
        return out


class TabularGRN(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.block1 = GatedResidualBlock(hidden_dim, dropout)
        self.block2 = GatedResidualBlock(hidden_dim, dropout)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.out_head(h).squeeze(-1)


# -------------------------------------------------------------------------
# Base Model 1: LightGBM
# -------------------------------------------------------------------------
print("Training Base Model 1: LightGBM...")
lgb_oof = np.zeros(num_samples, dtype=np.float32)
lgb_test_preds = np.zeros((5, len(X_test_mat)), dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.025,
    "num_leaves": 31,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "random_state": SEED,
    "verbose": -1,
}

for fold in range(5):
    tr_mask = train_processed["fold"] != fold
    va_mask = train_processed["fold"] == fold

    X_tr, y_tr = X_train_mat[tr_mask], y_train[tr_mask]
    X_va, y_va = X_train_mat[va_mask], y_train[va_mask]

    tr_data = lgb.Dataset(X_tr, label=y_tr)
    va_data = lgb.Dataset(X_va, label=y_va, reference=tr_data)

    model = lgb.train(
        lgb_params,
        tr_data,
        num_boost_round=1800,
        valid_sets=[tr_data, va_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    lgb_oof[va_mask] = model.predict(X_va, num_iteration=model.best_iteration)
    lgb_test_preds[fold] = model.predict(X_test_mat, num_iteration=model.best_iteration)

lgb_test = lgb_test_preds.mean(axis=0)

# -------------------------------------------------------------------------
# Base Model 2: XGBoost
# -------------------------------------------------------------------------
print("Training Base Model 2: XGBoost...")
xgb_oof = np.zeros(num_samples, dtype=np.float32)
xgb_test_preds = np.zeros((5, len(X_test_mat)), dtype=np.float32)

for fold in range(5):
    tr_mask = train_processed["fold"] != fold
    va_mask = train_processed["fold"] == fold

    X_tr, y_tr = X_train_mat[tr_mask], y_train[tr_mask]
    X_va, y_va = X_train_mat[va_mask], y_train[va_mask]

    model = xgb.XGBRegressor(
        n_estimators=1800,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=3,
        tree_method="hist",
        random_state=SEED + fold,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    xgb_oof[va_mask] = model.predict(X_va)
    xgb_test_preds[fold] = model.predict(X_test_mat)

xgb_test = xgb_test_preds.mean(axis=0)

# -------------------------------------------------------------------------
# Base Model 3: CatBoost
# -------------------------------------------------------------------------
print("Training Base Model 3: CatBoost...")
cat_oof = np.zeros(num_samples, dtype=np.float32)
cat_test_preds = np.zeros((5, len(X_test_mat)), dtype=np.float32)

for fold in range(5):
    tr_mask = train_processed["fold"] != fold
    va_mask = train_processed["fold"] == fold

    X_tr, y_tr = X_train_mat[tr_mask], y_train[tr_mask]
    X_va, y_va = X_train_mat[va_mask], y_train[va_mask]

    model = cb.CatBoostRegressor(
        iterations=1800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=4.0,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="RMSE",
    )

    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    cat_oof[va_mask] = model.predict(X_va)
    cat_test_preds[fold] = model.predict(X_test_mat)

cat_test = cat_test_preds.mean(axis=0)

# -------------------------------------------------------------------------
# Base Model 4: PyTorch TabularGRN
# -------------------------------------------------------------------------
print("Training Base Model 4: PyTorch TabularGRN...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_oof = np.zeros(num_samples, dtype=np.float32)
nn_test_preds = np.zeros((5, len(X_test_mat)), dtype=np.float32)

epochs = 35
batch_size = 128

for fold in range(5):
    tr_mask = train_processed["fold"] != fold
    va_mask = train_processed["fold"] == fold

    X_tr, y_tr = X_train_mat[tr_mask].copy(), y_train[tr_mask]
    X_va, y_va = X_train_mat[va_mask].copy(), y_train[va_mask]
    X_te = X_test_mat.copy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TabularGRN(input_dim=X_tr.shape[1], hidden_dim=128, dropout=0.15).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_va_preds = None
    best_te_preds = None

    va_tensor = torch.tensor(X_va, dtype=torch.float32).to(device)
    te_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_out = model(va_tensor)
            val_loss = criterion(va_out, torch.tensor(y_va).to(device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_va_preds = va_out.cpu().numpy()
                best_te_preds = model(te_tensor).cpu().numpy()

    nn_oof[va_mask] = best_va_preds
    nn_test_preds[fold] = best_te_preds

nn_test = nn_test_preds.mean(axis=0)

base_oofs = [lgb_oof, xgb_oof, cat_oof, nn_oof]
base_tests = [lgb_test, xgb_test, cat_test, nn_test]
model_names = ["LightGBM", "XGBoost", "CatBoost", "TabularGRN"]

for name, oof in zip(model_names, base_oofs):
    score = np.sqrt(mean_squared_error(y_train, oof))
    print(f"Base Model {name} OOF RMSE: {score:.5f}")

# -------------------------------------------------------------------------
# Stage-2 Stacking Meta-Learner with Domain Target Clipping
# -------------------------------------------------------------------------
print("Training Stage-2 Stacking Meta-Learner...")


def extract_meta_features(oof_list):
    mat = np.column_stack(oof_list)
    mean_feat = mat.mean(axis=1, keepdims=True)
    std_feat = mat.std(axis=1, keepdims=True)
    max_feat = mat.max(axis=1, keepdims=True)
    min_feat = mat.min(axis=1, keepdims=True)
    return np.hstack([mat, mean_feat, std_feat, max_feat, min_feat])


meta_X_train = extract_meta_features(base_oofs)
meta_X_test = extract_meta_features(base_tests)

meta_oof = np.zeros(num_samples, dtype=np.float32)
meta_test = np.zeros(len(X_test_mat), dtype=np.float32)

for fold in range(5):
    tr_mask = train_processed["fold"] != fold
    va_mask = train_processed["fold"] == fold

    meta_tr_X, meta_tr_y = meta_X_train[tr_mask], y_train[tr_mask]
    meta_va_X, meta_va_y = meta_X_train[va_mask], y_train[va_mask]

    meta_model = Ridge(alpha=10.0, random_state=SEED)
    meta_model.fit(meta_tr_X, meta_tr_y)

    meta_oof[va_mask] = meta_model.predict(meta_va_X)
    meta_test += meta_model.predict(meta_X_test) / 5.0

# Apply domain-constrained clipping to valid target bounds
min_target, max_target = float(y_train.min()), float(y_train.max())
meta_oof = np.clip(meta_oof, min_target, max_target)
meta_test = np.clip(meta_test, min_target, max_target)

# Save test predictions to submission file
os.makedirs("./submission", exist_ok=True)
sub_df = pd.DataFrame({"id": test_processed["id"].values, "IMU Score": meta_test})
sub_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved successfully to ./submission/submission.csv")

final_val_score = np.sqrt(mean_squared_error(y_train, meta_oof))
print(f"Final Validation Score: {final_val_score:.5f}")