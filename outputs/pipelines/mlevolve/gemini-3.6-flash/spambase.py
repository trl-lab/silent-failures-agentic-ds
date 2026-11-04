import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create working and submission directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Datasets
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "class"
id_col = "id"

if id_col in test_raw.columns:
    test_ids = test_raw[id_col].values
else:
    test_ids = np.arange(len(test_raw))

# Separate target and features
y_raw = train_raw[target_col].astype(int).values
drop_cols_train = [target_col] + ([id_col] if id_col in train_raw.columns else [])
X_raw = train_raw.drop(columns=drop_cols_train)

drop_cols_test = ([id_col] if id_col in test_raw.columns else []) + (
    [target_col] if target_col in test_raw.columns else []
)
X_test_raw = test_raw.drop(columns=drop_cols_test)

# 2. Stratified Train / Validation Split (Strict Data Isolation)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)


# 3. Domain Feature Engineering
def create_domain_features(df):
    df_feat = df.copy()

    word_cols = [c for c in df.columns if c.startswith("word_freq_")]
    char_cols = [c for c in df.columns if c.startswith("char_freq_")]
    cap_cols = [c for c in df.columns if c.startswith("capital_run_length_")]

    # Word frequency aggregations
    if word_cols:
        word_arr = df_feat[word_cols].values
        df_feat["word_freq_sum"] = word_arr.sum(axis=1)
        df_feat["word_freq_mean"] = word_arr.mean(axis=1)
        df_feat["word_freq_std"] = word_arr.std(axis=1)
        df_feat["word_freq_max"] = word_arr.max(axis=1)
        df_feat["word_freq_nonzero_count"] = (word_arr > 0).sum(axis=1)
        df_feat["word_freq_nonzero_ratio"] = df_feat["word_freq_nonzero_count"] / len(
            word_cols
        )

        spam_keywords = [
            c
            for c in word_cols
            if any(
                k in c.lower()
                for k in [
                    "free",
                    "remove",
                    "money",
                    "credit",
                    "your",
                    "000",
                    "business",
                    "email",
                    "mail",
                    "receive",
                    "order",
                    "font",
                    "3d",
                    "you",
                    "our",
                    "over",
                    "all",
                    "make",
                    "win",
                    "buy",
                ]
            )
        ]
        ham_keywords = [
            c
            for c in word_cols
            if any(
                k in c.lower()
                for k in [
                    "george",
                    "650",
                    "hp",
                    "hpl",
                    "lab",
                    "labs",
                    "telnet",
                    "857",
                    "415",
                    "data",
                    "85",
                    "technology",
                    "1999",
                    "pm",
                    "cs",
                    "meeting",
                    "original",
                    "project",
                    "re",
                    "edu",
                    "table",
                    "conference",
                ]
            )
        ]

        if spam_keywords:
            df_feat["spam_words_sum"] = df_feat[spam_keywords].sum(axis=1)
            df_feat["spam_words_count"] = (df_feat[spam_keywords] > 0).sum(axis=1)
        if ham_keywords:
            df_feat["ham_words_sum"] = df_feat[ham_keywords].sum(axis=1)
            df_feat["ham_words_count"] = (df_feat[ham_keywords] > 0).sum(axis=1)
        if spam_keywords and ham_keywords:
            df_feat["spam_ham_word_ratio"] = (df_feat["spam_words_sum"] + 1e-5) / (
                df_feat["ham_words_sum"] + 1e-5
            )

        for col in word_cols:
            df_feat[f"{col}_log"] = np.log1p(df_feat[col])

    # Char frequency aggregations
    if char_cols:
        char_arr = df_feat[char_cols].values
        df_feat["char_freq_sum"] = char_arr.sum(axis=1)
        df_feat["char_freq_nonzero_count"] = (char_arr > 0).sum(axis=1)

        excl_col = [c for c in char_cols if "!" in c or "excl" in c]
        dollar_col = [c for c in char_cols if "$" in c or "dollar" in c]
        if excl_col and dollar_col:
            df_feat["char_dollar_excl_ratio"] = (df_feat[dollar_col[0]] + 1e-5) / (
                df_feat[excl_col[0]] + 1e-5
            )

    # Capital run length interaction ratios and transforms
    avg_col = [c for c in cap_cols if "average" in c]
    long_col = [c for c in cap_cols if "longest" in c]
    tot_col = [c for c in cap_cols if "total" in c]

    if avg_col and long_col and tot_col:
        df_feat["cap_run_longest_to_avg"] = df_feat[long_col[0]] / (
            df_feat[avg_col[0]] + 1e-5
        )
        df_feat["cap_run_tot_to_longest"] = df_feat[tot_col[0]] / (
            df_feat[long_col[0]] + 1e-5
        )
        df_feat["cap_run_tot_to_avg"] = df_feat[tot_col[0]] / (
            df_feat[avg_col[0]] + 1e-5
        )

        df_feat["cap_avg_log"] = np.log1p(df_feat[avg_col[0]])
        df_feat["cap_longest_log"] = np.log1p(df_feat[long_col[0]])
        df_feat["cap_total_log"] = np.log1p(df_feat[tot_col[0]])

    return df_feat


X_train_engineered = create_domain_features(X_train_raw)
X_val_engineered = create_domain_features(X_val_raw)
X_test_engineered = create_domain_features(X_test_raw)

# 4. Latent Topic SVD Features (Fitted ONLY on Training set)
word_cols = [c for c in X_train_raw.columns if c.startswith("word_freq_")]
if len(word_cols) >= 5:
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd_train = svd.fit_transform(X_train_raw[word_cols])
    svd_val = svd.transform(X_val_raw[word_cols])
    svd_test = svd.transform(X_test_raw[word_cols])

    for i in range(5):
        X_train_engineered[f"svd_word_{i}"] = svd_train[:, i]
        X_val_engineered[f"svd_word_{i}"] = svd_val[:, i]
        X_test_engineered[f"svd_word_{i}"] = svd_test[:, i]

# Clean NaNs/Infs using training set medians
train_medians = X_train_engineered.median()
X_train_engineered = (
    X_train_engineered.replace([np.inf, -np.inf], np.nan)
    .fillna(train_medians)
    .fillna(0)
)
X_val_engineered = (
    X_val_engineered.replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0)
)
X_test_engineered = (
    X_test_engineered.replace([np.inf, -np.inf], np.nan).fillna(train_medians).fillna(0)
)

X_train = X_train_engineered.values
X_val = X_val_engineered.values
X_test = X_test_engineered.values

# 5. Train LightGBM Model
lgb_model = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective="binary",
    metric="auc",
    verbosity=-1,
    n_jobs=-1,
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
test_preds_lgb = lgb_model.predict_proba(X_test)[:, 1]

# 6. Train XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1]


# 7. Train PyTorch Tabular ResMLP
class TabularResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

nn_model = TabularResMLP(
    input_dim=X_train.shape[1], hidden_dim=256, num_blocks=3, dropout_rate=0.25
).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)

best_val_auc = 0.0
best_model_state = None

for epoch in range(1, 61):
    nn_model.train()
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = nn_model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

    nn_model.eval()
    val_logits_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            logits = nn_model(bx)
            val_logits_list.append(logits.cpu().numpy())

    val_preds_epoch = torch.sigmoid(torch.tensor(np.vstack(val_logits_list))).numpy()[
        :, 0
    ]
    epoch_auc = roc_auc_score(y_val, val_preds_epoch)
    scheduler.step(epoch_auc)

    if epoch_auc > best_val_auc:
        best_val_auc = epoch_auc
        best_model_state = {
            k: v.cpu().clone() for k, v in nn_model.state_dict().items()
        }

nn_model.load_state_dict(best_model_state)
nn_model.eval()

# Validation and Test predictions for Neural Network
val_logits_list = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device)
        logits = nn_model(bx)
        val_logits_list.append(logits.cpu().numpy())
val_preds_nn = torch.sigmoid(torch.tensor(np.vstack(val_logits_list))).numpy()[:, 0]

test_logits_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device)
        logits = nn_model(bx)
        test_logits_list.append(logits.cpu().numpy())
test_preds_nn = torch.sigmoid(torch.tensor(np.vstack(test_logits_list))).numpy()[:, 0]


# 8. Ensemble Weight Optimization on Validation Set
def blend_objective(weights):
    w1, w2, w3 = weights
    pred = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_nn
    return -roc_auc_score(y_val, pred)


res = minimize(
    blend_objective,
    [0.4, 0.4, 0.2],
    bounds=[(0, 1), (0, 1), (0, 1)],
    method="Nelder-Mead",
)
w1, w2, w3 = res.x
w_sum = w1 + w2 + w3
w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum

val_preds_blend = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_nn
test_preds_blend = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_nn

final_score = roc_auc_score(y_val, val_preds_blend)

# 9. Save Predictions to submission.csv
sub = pd.DataFrame({"id": test_ids, "class": test_preds_blend})
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
