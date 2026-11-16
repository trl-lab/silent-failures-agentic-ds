import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Configuration & Reproducibility
SEED = 42
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# 2. Data Loading & Label Encoding
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
target_col = "target"

label_encoder = LabelEncoder()
train_df["target_encoded"] = label_encoder.fit_transform(train_df[target_col])
class_labels = label_encoder.classes_
num_classes = len(class_labels)

# 3. Leakage-Free Stratified Train/Validation Split
train_split, val_split = train_test_split(
    train_df,
    test_size=0.20,
    random_state=SEED,
    stratify=train_df["target_encoded"],
)

X_tr_raw = train_split[feature_cols].values.astype(np.float32)
y_tr = train_split["target_encoded"].values.astype(np.int64)

X_va_raw = val_split[feature_cols].values.astype(np.float32)
y_va = val_split["target_encoded"].values.astype(np.int64)

X_te_raw = test_df[feature_cols].values.astype(np.float32)
id_te = test_df["id"].values.astype(np.int64)


# 4. Feature Engineering Pipeline
def compute_row_statistics(X):
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums_safe = np.maximum(row_sums, 1e-8)

    non_zeros = (X > 0).sum(axis=1, keepdims=True)
    zeros = X.shape[1] - non_zeros

    P = X / row_sums_safe
    entropy = -np.sum(P * np.log(P + 1e-12), axis=1, keepdims=True)
    simpson = 1.0 - np.sum(P**2, axis=1, keepdims=True)

    row_max = X.max(axis=1, keepdims=True)
    row_mean = X.mean(axis=1, keepdims=True)
    row_std = X.std(axis=1, keepdims=True)

    row_features = np.hstack(
        [
            row_sums,
            non_zeros,
            zeros,
            entropy,
            simpson,
            row_max,
            row_mean,
            row_std,
        ]
    )

    return P, row_features


P_tr, row_ft_tr = compute_row_statistics(X_tr_raw)
P_va, row_ft_va = compute_row_statistics(X_va_raw)
P_te, row_ft_te = compute_row_statistics(X_te_raw)

X_tr_log = np.log1p(X_tr_raw)
X_va_log = np.log1p(X_va_raw)
X_te_log = np.log1p(X_te_raw)

# Stateful Unsupervised Fits (Strictly on Train Split)
n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
kmeans.fit(P_tr)

kmeans_dist_tr = kmeans.transform(P_tr)
kmeans_dist_va = kmeans.transform(P_va)
kmeans_dist_te = kmeans.transform(P_te)

n_svd = 10
svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
svd.fit(X_tr_log)

svd_tr = svd.transform(X_tr_log)
svd_va = svd.transform(X_va_log)
svd_te = svd.transform(X_te_log)

X_tr = np.hstack([X_tr_raw, X_tr_log, row_ft_tr, kmeans_dist_tr, svd_tr])
X_va = np.hstack([X_va_raw, X_va_log, row_ft_va, kmeans_dist_va, svd_va])
X_te = np.hstack([X_te_raw, X_te_log, row_ft_te, kmeans_dist_te, svd_te])


# 5. Model Architecture Definitions
class TabularResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.20):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class OttoTabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 9,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "min_child_samples": 20,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}

# 6. Train LightGBM
dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

lgb_params = LGBM_PARAMS.copy()
n_est_lgb = lgb_params.pop("n_estimators", 1500)

lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=n_est_lgb,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

val_probs_lgb = lgb_model.predict(X_va)
test_probs_lgb = lgb_model.predict(X_te)

# 7. Train XGBoost
xgb_params = XGB_PARAMS.copy()
n_est_xgb = xgb_params.pop("n_estimators", 1500)

xgb_model = xgb.XGBClassifier(
    **xgb_params, n_estimators=n_est_xgb, early_stopping_rounds=50
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

val_probs_xgb = xgb_model.predict_proba(X_va)
test_probs_xgb = xgb_model.predict_proba(X_te)

# 8. Train PyTorch Deep ResNet
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = X_tr.shape[1]

nn_model = OttoTabularResNet(
    input_dim=input_dim,
    num_classes=num_classes,
    hidden_dim=512,
    num_blocks=3,
    dropout_rate=0.25,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=40, eta_min=1e-5
)

train_dataset = TensorDataset(
    torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)
)
test_dataset = TensorDataset(torch.tensor(X_te, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

best_val_loss = float("inf")
best_weights = None
patience, patience_counter = 10, 0
epochs = 40

for epoch in range(1, epochs + 1):
    nn_model.train()
    train_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(by)
    train_loss /= len(train_dataset)

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            out = nn_model(bx)
            loss = criterion(out, by)
            val_loss += loss.item() * len(by)
    val_loss /= len(val_dataset)

    scheduler.step()
    print(
        f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss:"
        f" {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

if best_weights is not None:
    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

nn_model.eval()
val_preds_nn_list = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        probs = torch.softmax(nn_model(bx), dim=1)
        val_preds_nn_list.append(probs.cpu().numpy())
val_probs_nn = np.vstack(val_preds_nn_list)

test_preds_nn_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        probs = torch.softmax(nn_model(bx), dim=1)
        test_preds_nn_list.append(probs.cpu().numpy())
test_probs_nn = np.vstack(test_preds_nn_list)

# 9. Model Blending & Final Evaluation
w_lgb, w_xgb, w_nn = 0.40, 0.40, 0.20
val_probs_blend = w_lgb * val_probs_lgb + w_xgb * val_probs_xgb + w_nn * val_probs_nn
val_probs_blend = np.clip(val_probs_blend, 1e-15, 1.0 - 1e-15)
val_probs_blend /= val_probs_blend.sum(axis=1, keepdims=True)

final_val_score = log_loss(y_va, val_probs_blend)

# 10. Generate Final Submission
test_probs_blend = (
    w_lgb * test_probs_lgb + w_xgb * test_probs_xgb + w_nn * test_probs_nn
)
test_pred_indices = np.argmax(test_probs_blend, axis=1)
test_pred_targets = class_labels[test_pred_indices]

submission_df = pd.DataFrame({"id": id_te, "target": test_pred_targets})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_score}")
