import copy
import os
import random
import lightgbm as lgb
import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ---------------------------------------------------------
# 1. Load Data & Validation Split (Strict Isolation)
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

id_col = "id" if "id" in train_df.columns else None
target_col = "target"
feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

X_raw = train_df[feature_cols].values.astype(np.float32)
y_raw = train_df[target_col].values
X_test_raw = test_df[feature_cols].values.astype(np.float32)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

# Stratified Split BEFORE any transformation to prevent data leakage
train_idx, val_idx = train_test_split(
    np.arange(len(y_encoded)),
    test_size=0.20,
    stratify=y_encoded,
    random_state=42,
)

X_train_raw = X_raw[train_idx]
y_train = y_encoded[train_idx]
X_val_raw = X_raw[val_idx]
y_val = y_encoded[val_idx]


# ---------------------------------------------------------
# 2. Feature Engineering & Stateful Transformations
# ---------------------------------------------------------
def compute_row_summary_features(X):
    row_sum = np.sum(X, axis=1, keepdims=True)
    row_nonzero_count = np.sum(X > 0, axis=1, keepdims=True)
    safe_nz = np.maximum(row_nonzero_count, 1)
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    row_sparsity = np.mean(X == 0, axis=1, keepdims=True)

    # Non-zero distribution moments
    row_nonzero_mean = row_sum / safe_nz
    diff = np.where(X > 0, X - row_nonzero_mean, 0.0)
    row_nonzero_std = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True) / safe_nz)
    row_nonzero_min = np.min(np.where(X > 0, X, 999999.0), axis=1, keepdims=True)
    row_nonzero_min = np.where(row_nonzero_min == 999999.0, 0.0, row_nonzero_min)

    # High-percentile quantiles
    p75 = np.percentile(X, 75, axis=1, keepdims=True)
    p90 = np.percentile(X, 90, axis=1, keepdims=True)

    # Sample-wise distribution entropy
    tf_norm = X / (row_sum + 1e-6)
    entropy = -np.sum(tf_norm * np.log(tf_norm + 1e-9), axis=1, keepdims=True)

    return np.hstack(
        [
            row_sum,
            row_nonzero_count,
            row_mean,
            row_std,
            row_max,
            row_sparsity,
            row_nonzero_mean,
            row_nonzero_std,
            row_nonzero_min,
            p75,
            p90,
            entropy,
            tf_norm,
        ]
    )


# 2.1 Log1p transformation for count distributions
X_train_log = np.log1p(X_train_raw)
X_val_log = np.log1p(X_val_raw)
X_test_log = np.log1p(X_test_raw)

# 2.2 Row summary statistics and normalized profiles
X_train_stats = compute_row_summary_features(X_train_raw)
X_val_stats = compute_row_summary_features(X_val_raw)
X_test_stats = compute_row_summary_features(X_test_raw)

# 2.3 TF-IDF transformation (fitted strictly on training split)
tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
X_train_tfidf = tfidf.fit_transform(X_train_raw).toarray().astype(np.float32)
X_val_tfidf = tfidf.transform(X_val_raw).toarray().astype(np.float32)
X_test_tfidf = tfidf.transform(X_test_raw).toarray().astype(np.float32)

# 2.4 TruncatedSVD latent semantic representations (fitted on train only)
svd = TruncatedSVD(n_components=20, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf).astype(np.float32)
X_val_svd = svd.transform(X_val_tfidf).astype(np.float32)
X_test_svd = svd.transform(X_test_tfidf).astype(np.float32)

# 2.5 Log-transformed SVD components
svd_log = TruncatedSVD(n_components=15, random_state=42)
X_train_svd_log = svd_log.fit_transform(X_train_log).astype(np.float32)
X_val_svd_log = svd_log.transform(X_val_log).astype(np.float32)
X_test_svd_log = svd_log.transform(X_test_log).astype(np.float32)

# 2.6 Standardized Log Features & PCA components (fitted on train only)
scaler = StandardScaler()
X_train_log_scaled = scaler.fit_transform(X_train_log)
X_val_log_scaled = scaler.transform(X_val_log)
X_test_log_scaled = scaler.transform(X_test_log)

pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_log_scaled).astype(np.float32)
X_val_pca = pca.transform(X_val_log_scaled).astype(np.float32)
X_test_pca = pca.transform(X_test_log_scaled).astype(np.float32)

# Concatenate all engineered feature sets
X_train_full = np.hstack(
    [
        X_train_raw,
        X_train_log,
        X_train_stats,
        X_train_tfidf,
        X_train_svd,
        X_train_svd_log,
        X_train_pca,
    ]
).astype(np.float32)

X_val_full = np.hstack(
    [
        X_val_raw,
        X_val_log,
        X_val_stats,
        X_val_tfidf,
        X_val_svd,
        X_val_svd_log,
        X_val_pca,
    ]
).astype(np.float32)

X_test_full = np.hstack(
    [
        X_test_raw,
        X_test_log,
        X_test_stats,
        X_test_tfidf,
        X_test_svd,
        X_test_svd_log,
        X_test_pca,
    ]
).astype(np.float32)

in_features = X_train_full.shape[1]


# ---------------------------------------------------------
# 3. Model Architectures & Definitions
# ---------------------------------------------------------
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * w


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.se = SqueezeExcitationBlock(hidden_dim, reduction=4)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.se(out)
        return self.activation(out + residual)


class TabularResNetSE(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int = 9,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


# ---------------------------------------------------------
# 4. Model Training: LightGBM
# ---------------------------------------------------------
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

lgb_train_data = lgb.Dataset(X_train_full, label=y_train)
lgb_val_data = lgb.Dataset(X_val_full, label=y_val, reference=lgb_train_data)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train_data,
    num_boost_round=1200,
    valid_sets=[lgb_val_data],
    callbacks=callbacks,
)

val_preds_lgb = lgb_model.predict(X_val_full, num_iteration=lgb_model.best_iteration)
test_preds_lgb = lgb_model.predict(X_test_full, num_iteration=lgb_model.best_iteration)
lgb_val_loss = log_loss(y_val, val_preds_lgb)
print(f"LightGBM Validation Log Loss: {lgb_val_loss:.5f}")


# ---------------------------------------------------------
# 4.2 Model Training: XGBoost
# ---------------------------------------------------------
xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

dtrain = xgb.DMatrix(X_train_full, label=y_train)
dval = xgb.DMatrix(X_val_full, label=y_val)
dtest = xgb.DMatrix(X_test_full)

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

val_preds_xgb = xgb_model.predict(dval)
test_preds_xgb = xgb_model.predict(dtest)
xgb_val_loss = log_loss(y_val, val_preds_xgb)
print(f"XGBoost Validation Log Loss: {xgb_val_loss:.5f}")


# ---------------------------------------------------------
# 5. Model Training: TabularResNetSE (PyTorch with Input Scaling)
# ---------------------------------------------------------
# Standardize dense engineered features specifically for neural network representations
nn_scaler = StandardScaler()
X_train_nn = nn_scaler.fit_transform(X_train_full).astype(np.float32)
X_val_nn = nn_scaler.transform(X_val_full).astype(np.float32)
X_test_nn = nn_scaler.transform(X_test_full).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn_model = TabularResNetSE(
    in_features=in_features,
    num_classes=num_classes,
    hidden_dim=512,
    num_blocks=3,
    dropout_rate=0.25,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=40, eta_min=1e-6)

train_tensor_x = torch.tensor(X_train_nn, dtype=torch.float32)
train_tensor_y = torch.tensor(y_train, dtype=torch.long)
val_tensor_x = torch.tensor(X_val_nn, dtype=torch.float32)
val_tensor_y = torch.tensor(y_val, dtype=torch.long)
test_tensor_x = torch.tensor(X_test_nn, dtype=torch.float32)

train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

epochs = 40
patience = 8
patience_counter = 0
best_val_loss = float("inf")
best_model_weights = None

for epoch in range(1, epochs + 1):
    nn_model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = nn_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * len(batch_y)

    lr_scheduler.step()

    nn_model.eval()
    val_probs_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            logits = nn_model(batch_x)
            probs = F.softmax(logits, dim=-1)
            val_probs_list.append(probs.cpu().numpy())

    epoch_val_probs = np.vstack(val_probs_list)
    epoch_val_loss = log_loss(y_val, epoch_val_probs)
    epoch_train_loss = running_loss / len(train_dataset)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.4f} - Val Log Loss: {epoch_val_loss:.5f}"
    )

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_weights = copy.deepcopy(nn_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

nn_model.load_state_dict(best_model_weights)

# Validation and Test predictions from the best neural checkpoint
nn_model.eval()
with torch.no_grad():
    val_logits = nn_model(val_tensor_x.to(device))
    val_preds_nn = F.softmax(val_logits, dim=-1).cpu().numpy()

    test_loader = DataLoader(
        TensorDataset(test_tensor_x), batch_size=512, shuffle=False
    )
    test_probs_list = []
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        test_logits = nn_model(batch_x)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
        test_probs_list.append(test_probs)
    test_preds_nn = np.vstack(test_probs_list)

print(f"TabularResNetSE Best Validation Log Loss: {best_val_loss:.5f}")


# ---------------------------------------------------------
# 6. Ensemble Optimization & Blending (3-Way SLSQP)
# ---------------------------------------------------------
def loss_func(weights):
    w1, w2, w3 = weights
    blended = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_nn
    blended = np.clip(blended, 1e-15, 1.0 - 1e-15)
    blended /= np.sum(blended, axis=1, keepdims=True)
    return log_loss(y_val, blended)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

res = minimize(
    loss_func,
    x0=init_weights,
    bounds=bounds,
    constraints=constraints,
    method="SLSQP",
)

opt_w = res.x / np.sum(res.x)
print(
    f"Optimal Ensemble Weights -> LightGBM: {opt_w[0]:.4f}, XGBoost: {opt_w[1]:.4f}, TabularResNet: {opt_w[2]:.4f}"
)

val_preds_ensemble = (
    opt_w[0] * val_preds_lgb
    + opt_w[1] * val_preds_xgb
    + opt_w[2] * val_preds_nn
)
val_preds_ensemble = np.clip(val_preds_ensemble, 1e-15, 1.0 - 1e-15)
val_preds_ensemble /= np.sum(val_preds_ensemble, axis=1, keepdims=True)

final_score = log_loss(y_val, val_preds_ensemble)


# ---------------------------------------------------------
# 7. Generate Submission File
# ---------------------------------------------------------
test_preds_ensemble = (
    opt_w[0] * test_preds_lgb
    + opt_w[1] * test_preds_xgb
    + opt_w[2] * test_preds_nn
)
test_preds_ensemble = np.clip(test_preds_ensemble, 1e-15, 1.0 - 1e-15)
test_preds_ensemble /= np.sum(test_preds_ensemble, axis=1, keepdims=True)

test_pred_indices = np.argmax(test_preds_ensemble, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_pred_indices)

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": np.arange(len(X_test_full), dtype=int), "target": test_pred_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

# ---------------------------------------------------------
# 8. Final Validation Score Output
# ---------------------------------------------------------
print(f"Final Validation Score: {final_score}")
