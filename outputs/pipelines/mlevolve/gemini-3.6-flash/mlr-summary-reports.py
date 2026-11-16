import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
np.random.seed(42)

# Environment Setup
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# 1. Load Raw Datasets
train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

target_col = "Adjusted MLR"
if target_col not in train_raw.columns:
    possible_targets = [
        c for c in train_raw.columns if "mlr" in c.lower() or "target" in c.lower()
    ]
    target_col = possible_targets[0] if possible_targets else train_raw.columns[-1]

y = train_raw[target_col].copy()
X = train_raw.drop(columns=[target_col]).copy()
X_test = test_raw.copy()

# Preserve row identifier if present
id_col = None
for col in ["id", "ID", "row_id", "Index"]:
    if col in X.columns and col in X_test.columns:
        id_col = col
        break

X_te = X_test.reset_index(drop=True)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Feature Engineering
def engineer_domain_features(df):
    df_feat = df.copy()

    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    if id_col and id_col in num_cols:
        num_cols.remove(id_col)

    claim_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["claim", "incurred", "loss"])
    ]
    premium_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["premium", "revenue", "earned"])
    ]
    member_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["member", "month", "life", "covered"])
    ]
    expense_cols = [
        c
        for c in num_cols
        if any(k in c.lower() for k in ["expense", "quality", "admin", "tax"])
    ]

    # Domain Ratios
    for c_col in claim_cols[:3]:
        for p_col in premium_cols[:3]:
            if c_col != p_col:
                ratio_name = f"ratio_{c_col}_to_{p_col}"
                df_feat[ratio_name] = df_feat[c_col] / (df_feat[p_col].abs() + 1e-5)

    # Per Member Per Month (PMPM) Metrics
    if member_cols:
        m_col = member_cols[0]
        for mon_col in (claim_cols + premium_cols + expense_cols)[:6]:
            if mon_col != m_col:
                df_feat[f"pmpm_{mon_col}"] = df_feat[mon_col] / (
                    df_feat[m_col].abs() + 1e-5
                )

    # Differences and Additions
    if len(claim_cols) >= 1 and len(expense_cols) >= 1:
        df_feat["total_incurred_and_expense"] = (
            df_feat[claim_cols[0]] + df_feat[expense_cols[0]]
        )
        if len(premium_cols) >= 1:
            df_feat["net_margin_approx"] = (
                df_feat[premium_cols[0]] - df_feat["total_incurred_and_expense"]
            )

    # Pairwise ratios between top numerical features if domain keywords absent
    if len(claim_cols) == 0 and len(premium_cols) == 0 and len(num_cols) >= 2:
        top_nums = num_cols[:5]
        for i in range(len(top_nums)):
            for j in range(i + 1, len(top_nums)):
                c1, c2 = top_nums[i], top_nums[j]
                df_feat[f"ratio_{c1}_{c2}"] = df_feat[c1] / (df_feat[c2].abs() + 1e-5)
                df_feat[f"diff_{c1}_{c2}"] = df_feat[c1] - df_feat[c2]

    # Log1p Transformation handled consistently outside function

    return df_feat


# Determine log-eligible columns from training set to ensure feature consistency
log_cols = [c for c in X_tr.select_dtypes(include=[np.number]).columns if id_col is None or c != id_col]
log_cols = [c for c in log_cols if X_tr[c].min() >= 0]

def apply_log_transforms(df, log_columns):
    df_feat = df.copy()
    for col in log_columns:
        if col in df_feat.columns:
            df_feat[f"log1p_{col}"] = np.log1p(np.clip(df_feat[col], a_min=0, a_max=None))
    return df_feat

X_tr_fe = apply_log_transforms(engineer_domain_features(X_tr), log_cols)
X_va_fe = apply_log_transforms(engineer_domain_features(X_va), log_cols)
X_te_fe = apply_log_transforms(engineer_domain_features(X_te), log_cols)

# 3. Categoricals and Target Encoding
cat_cols = X_tr_fe.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    freq_map = X_tr_fe[col].value_counts(normalize=True).to_dict()
    X_tr_fe[f"{col}_freq"] = X_tr_fe[col].map(freq_map).fillna(0)
    X_va_fe[f"{col}_freq"] = X_va_fe[col].map(freq_map).fillna(0)
    X_te_fe[f"{col}_freq"] = X_te_fe[col].map(freq_map).fillna(0)

    target_map = y_tr.groupby(X_tr_fe[col]).mean().to_dict()
    global_mean = y_tr.mean()
    X_tr_fe[f"{col}_target_enc"] = X_tr_fe[col].map(target_map).fillna(global_mean)
    X_va_fe[f"{col}_target_enc"] = X_va_fe[col].map(target_map).fillna(global_mean)
    X_te_fe[f"{col}_target_enc"] = X_te_fe[col].map(target_map).fillna(global_mean)

    cat_type = pd.CategoricalDtype(categories=X_tr_fe[col].dropna().unique())
    X_tr_fe[col] = X_tr_fe[col].astype(cat_type).cat.codes
    X_va_fe[col] = X_va_fe[col].astype(cat_type).cat.codes
    X_te_fe[col] = X_te_fe[col].astype(cat_type).cat.codes

# 4. Imputation and Scaling strictly on Training Set
num_features = [c for c in X_tr_fe.columns if c != id_col]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_tr_imp = imputer.fit_transform(X_tr_fe[num_features])
X_va_imp = imputer.transform(X_va_fe[num_features])
X_te_imp = imputer.transform(X_te_fe[num_features])

X_tr_scaled = scaler.fit_transform(X_tr_imp)
X_va_scaled = scaler.transform(X_va_imp)
X_te_scaled = scaler.transform(X_te_imp)

# Convert to Tensors
X_tr_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32).to(device)
y_tr_tensor = torch.tensor(y_tr.values, dtype=torch.float32).to(device)

X_va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
y_va_tensor = torch.tensor(y_va.values, dtype=torch.float32).to(device)

X_te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

input_dim = len(num_features)


# 5. Model Architecture Setup
class SwiGLUBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc_gate = nn.Linear(in_dim, hidden_dim)
        self.fc_val = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        gate = F.silu(self.fc_gate(x))
        val = self.fc_val(x)
        gated = gate * val
        out = self.dropout(self.fc_out(gated))
        return self.norm(residual + out)


class TabularGatedAttentionNet(nn.Module):

    def __init__(
        self, input_dim, hidden_dim=256, num_heads=4, num_blocks=3, dropout=0.15
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                SwiGLUBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h = self.attn_norm(h + attn_out.squeeze(1))
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


model = TabularGatedAttentionNet(input_dim=input_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)

# 6. DataLoader Construction & Training Loop
batch_size = max(16, min(64, len(X_tr_tensor) // 10))
train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

y_tr_min = float(y_tr.min())

epochs = 100
patience = 20
patience_counter = 0
best_val_rmse = float("inf")
best_model_path = os.path.join(working_dir, "best_model.pt")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(batch_x)

    scheduler.step()
    epoch_train_loss = running_loss / len(X_tr_tensor)

    model.eval()
    with torch.no_grad():
        val_preds_raw = model(X_va_tensor).cpu().numpy()

    y_va_true = y_va.values

    if y_tr_min >= 0:
        val_preds_eval = np.clip(val_preds_raw, a_min=0.0, a_max=None)
    else:
        val_preds_eval = val_preds_raw

    val_rmse = float(np.sqrt(mean_squared_error(y_va_true, val_preds_eval)))
    print(
        f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.5f} | Val RMSE: {val_rmse:.5f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# 7. Final Inference and Submission
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))

model.eval()
with torch.no_grad():
    final_val_preds_raw = model(X_va_tensor).cpu().numpy()
    final_test_preds_raw = model(X_te_tensor).cpu().numpy()

if y_tr_min >= 0:
    final_val_preds = np.clip(final_val_preds_raw, a_min=0.0, a_max=None)
    final_test_preds = np.clip(final_test_preds_raw, a_min=0.0, a_max=None)
else:
    final_val_preds = final_val_preds_raw
    final_test_preds = final_test_preds_raw

final_val_rmse = float(np.sqrt(mean_squared_error(y_va.values, final_val_preds)))

if id_col and id_col in test_raw.columns:
    sub_ids = test_raw[id_col].values
else:
    sub_ids = np.arange(len(final_test_preds))

submission_df = pd.DataFrame({"id": sub_ids, "Adjusted MLR": final_test_preds})
submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

print(f"Final Validation Score: {final_val_rmse:.5f}")