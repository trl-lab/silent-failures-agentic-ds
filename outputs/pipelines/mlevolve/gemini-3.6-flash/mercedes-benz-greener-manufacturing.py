import copy
import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# 1. Reproducibility Setup
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 3. Data Processing & Feature Engineering
cat_cols = [
    c for c in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"] if c in train_df.columns
]
binary_cols = [c for c in train_df.columns if c not in cat_cols + ["y", "ID", "id"]]

# Re-introduce ID as a numerical feature and normalized temporal trend
id_col_name = "ID" if "ID" in train_df.columns else ("id" if "id" in train_df.columns else None)
if id_col_name:
    min_id = min(train_df[id_col_name].min(), test_df[id_col_name].min())
    max_id = max(train_df[id_col_name].max(), test_df[id_col_name].max())
    train_df["ID_norm"] = (train_df[id_col_name] - min_id) / (max_id - min_id)
    test_df["ID_norm"] = (test_df[id_col_name] - min_id) / (max_id - min_id)

# Remove zero variance binary features strictly based on training set
zero_var_cols = [c for c in binary_cols if train_df[c].nunique() <= 1]
binary_cols = [c for c in binary_cols if c not in zero_var_cols]

# Remove duplicate binary columns strictly based on training set
dup_cols = train_df[binary_cols].columns[train_df[binary_cols].T.duplicated()].tolist()
binary_cols = [c for c in binary_cols if c not in dup_cols]

# Binary row-level statistical aggregations
for df in [train_df, test_df]:
    mat = df[binary_cols].values
    df["binary_sum"] = mat.sum(axis=1)
    df["binary_mean"] = mat.mean(axis=1)
    df["binary_std"] = mat.std(axis=1)

# Categorical group aggregations for binary_sum (grouped by X0, X1, X2)
global_binary_sum_mean = train_df["binary_sum"].mean()
global_binary_sum_std = train_df["binary_sum"].std()

for col in ["X0", "X1", "X2"]:
    group_mean = train_df.groupby(col)["binary_sum"].mean().to_dict()
    group_std = train_df.groupby(col)["binary_sum"].std().to_dict()

    train_df[f"{col}_binary_sum_mean"] = (
        train_df[col].map(group_mean).fillna(global_binary_sum_mean)
    )
    test_df[f"{col}_binary_sum_mean"] = (
        test_df[col].map(group_mean).fillna(global_binary_sum_mean)
    )

    train_df[f"{col}_binary_sum_std"] = (
        train_df[col].map(group_std).fillna(global_binary_sum_std)
    )
    test_df[f"{col}_binary_sum_std"] = (
        test_df[col].map(group_std).fillna(global_binary_sum_std)
    )

group_3_mean = (
    train_df.groupby(["X0", "X1", "X2"])["binary_sum"].mean().to_dict()
)
group_3_std = (
    train_df.groupby(["X0", "X1", "X2"])["binary_sum"].std().to_dict()
)

train_df["X0_X1_X2_binary_sum_mean"] = (
    train_df.set_index(["X0", "X1", "X2"])
    .index.map(group_3_mean)
    .fillna(global_binary_sum_mean)
)
test_df["X0_X1_X2_binary_sum_mean"] = (
    test_df.set_index(["X0", "X1", "X2"])
    .index.map(group_3_mean)
    .fillna(global_binary_sum_mean)
)

train_df["X0_X1_X2_binary_sum_std"] = (
    train_df.set_index(["X0", "X1", "X2"])
    .index.map(group_3_std)
    .fillna(global_binary_sum_std)
)
test_df["X0_X1_X2_binary_sum_std"] = (
    test_df.set_index(["X0", "X1", "X2"])
    .index.map(group_3_std)
    .fillna(global_binary_sum_std)
)

# Latent Manifold Decomposition on Sparse Binary Matrix
n_components = 10
pca = PCA(n_components=n_components, random_state=42)
svd = TruncatedSVD(n_components=n_components, random_state=42)
ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
fa = FactorAnalysis(n_components=n_components, random_state=42)

# Fit dimensionality reduction techniques strictly on train_df
pca_train = pca.fit_transform(train_df[binary_cols])
pca_test = pca.transform(test_df[binary_cols])

svd_train = svd.fit_transform(train_df[binary_cols])
svd_test = svd.transform(test_df[binary_cols])

ica_train = ica.fit_transform(train_df[binary_cols])
ica_test = ica.transform(test_df[binary_cols])

fa_train = fa.fit_transform(train_df[binary_cols])
fa_test = fa.transform(test_df[binary_cols])

for i in range(n_components):
    train_df[f"pca_{i}"] = pca_train[:, i]
    test_df[f"pca_{i}"] = pca_test[:, i]

    train_df[f"svd_{i}"] = svd_train[:, i]
    test_df[f"svd_{i}"] = svd_test[:, i]

    train_df[f"ica_{i}"] = ica_train[:, i]
    test_df[f"ica_{i}"] = ica_test[:, i]

    train_df[f"fa_{i}"] = fa_train[:, i]
    test_df[f"fa_{i}"] = fa_test[:, i]

# Categorical Feature Engineering: Pairwise Interactions & Frequency Encoding
top_cats = ["X0", "X1", "X2", "X5"]
interaction_cols = []
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        col1, col2 = top_cats[i], top_cats[j]
        new_col = f"{col1}_{col2}"
        train_df[new_col] = (
            train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
        )
        test_df[new_col] = test_df[col1].astype(str) + "_" + test_df[col2].astype(str)
        interaction_cols.append(new_col)

all_cat_cols = cat_cols + interaction_cols

for col in all_cat_cols:
    freq_map = train_df[col].value_counts().to_dict()
    train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq_map).fillna(0)

    le = LabelEncoder()
    unique_vals = list(
        set(train_df[col].astype(str).unique()).union(
            set(test_df[col].astype(str).unique())
        )
    )
    le.fit(unique_vals)
    train_df[f"{col}_encoded"] = le.transform(train_df[col].astype(str))
    test_df[f"{col}_encoded"] = le.transform(test_df[col].astype(str))

# Out-Of-Fold (OOF) Target Encoding for primary categorical variables
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold

global_mean = train_df["y"].mean()
for col in cat_cols:
    target_enc_col = f"{col}_te"
    train_df[target_enc_col] = np.nan

    for fold in range(5):
        tr_mask = train_df["fold"] != fold
        val_mask = train_df["fold"] == fold

        cat_means = train_df[tr_mask].groupby(col)["y"].mean().to_dict()
        train_df.loc[val_mask, target_enc_col] = (
            train_df.loc[val_mask, col].map(cat_means).fillna(global_mean)
        )

    full_cat_means = train_df.groupby(col)["y"].mean().to_dict()
    test_df[target_enc_col] = test_df[col].map(full_cat_means).fillna(global_mean)

# Clipped target for robust training objective
train_df["y_clipped"] = np.clip(train_df["y"], a_min=70.0, a_max=150.0)


# 4. Neural Network Architecture Definition
class TabularResNet(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.3
    ):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        out = self.head(x)
        return out.squeeze(-1)


# 5. Model Configurations
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.015,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.4,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1500,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.015,
    "max_depth": 3,
    "subsample": 0.7,
    "colsample_bytree": 0.4,
    "reg_alpha": 3.0,
    "reg_lambda": 10.0,
    "random_state": 42,
    "n_estimators": 1500,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.015,
    "depth": 4,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}


# 6. Feature Selection
non_feature_cols = ["y", "y_clipped", "fold"]
all_candidate_cols = [
    c for c in train_df.columns if c not in non_feature_cols and c in test_df.columns
]
numeric_feature_cols = [
    c for c in all_candidate_cols if pd.api.types.is_numeric_dtype(train_df[c])
]
feature_cols = numeric_feature_cols

y_raw = train_df["y"].values
folds = train_df["fold"].values

n_samples = len(train_df)
n_test = len(test_df)

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cb = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)
oof_ridge = np.zeros(n_samples)
oof_et = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cb = np.zeros(n_test)
test_nn = np.zeros(n_test)
test_ridge = np.zeros(n_test)
test_et = np.zeros(n_test)

# 7. Cross-Validation Training Loop
for fold in range(5):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_tr, y_tr = train_df.iloc[tr_idx][feature_cols].values.astype(np.float32), y_raw[tr_idx]
    X_va, y_va = train_df.iloc[val_idx][feature_cols].values.astype(np.float32), y_raw[val_idx]
    X_te = test_df[feature_cols].values.astype(np.float32)

    # A. LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_te) / 5.0

    # B. XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_te) / 5.0

    # C. Ridge Regressor
    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=250.0, random_state=42)),
    ])
    ridge_pipe.fit(X_tr, y_tr)
    oof_ridge[val_idx] = ridge_pipe.predict(X_va)
    test_ridge += ridge_pipe.predict(X_te) / 5.0

    # D. ExtraTrees Regressor
    et_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,
                max_features=0.4,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])
    et_pipe.fit(X_tr, y_tr)
    oof_et[val_idx] = et_pipe.predict(X_va)
    test_et += et_pipe.predict(X_te) / 5.0

    # E. CatBoost
    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = model_cb.predict(X_va)
    test_cb += model_cb.predict(X_te) / 5.0

    print(f"Fold {fold+1}/5 complete.")


# 8. SciPy SLSQP Ensemble Weight Optimization
def objective_loss(weights):
    w = weights / np.sum(weights)
    blend = (
        w[0] * oof_lgb
        + w[1] * oof_xgb
        + w[2] * oof_ridge
        + w[3] * oof_et
        + w[4] * oof_cb
    )
    return -r2_score(y_raw, blend)


init_weights = [0.25, 0.25, 0.15, 0.15, 0.20]
bounds = [(0, 1)] * 5
opt_res = opt.minimize(objective_loss, init_weights, method="SLSQP", bounds=bounds)
best_w = opt_res.x / np.sum(opt_res.x)

final_oof_preds = (
    best_w[0] * oof_lgb
    + best_w[1] * oof_xgb
    + best_w[2] * oof_ridge
    + best_w[3] * oof_et
    + best_w[4] * oof_cb
)
final_r2 = r2_score(y_raw, final_oof_preds)

# 9. Test Predictions & Post-Processing
test_preds = (
    best_w[0] * test_lgb
    + best_w[1] * test_xgb
    + best_w[2] * test_ridge
    + best_w[3] * test_et
    + best_w[4] * test_cb
)

# 10. Generate Submission
os.makedirs("./submission", exist_ok=True)

sub_id = np.arange(len(test_df))

submission_df = pd.DataFrame({"id": sub_id, "y": test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

# 11. Output Validation Result
print(f"Final Validation Score: {final_r2}")