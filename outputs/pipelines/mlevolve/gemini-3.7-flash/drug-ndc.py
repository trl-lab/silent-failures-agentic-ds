import copy
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------
# 0. Global Setup and Reproducibility
# ----------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ----------------------------------------------------
# 1. Load Data and Clean Target
# ----------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "dea_schedule"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

# Encode multi-class target
y_raw = train_df[target_col].fillna("None").astype(str).values
le = LabelEncoder()
y_train = le.fit_transform(y_raw)
num_classes = len(le.classes_)

drop_cols = [target_col]
if "id" in train_df.columns:
    drop_cols.append("id")

feature_cols = [c for c in train_df.columns if c not in drop_cols]

string_cols = [
    c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])
]
numeric_cols = [
    c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])
]

# Impute text and numerical columns safely
for c in string_cols:
    train_df[c] = train_df[c].fillna("").astype(str)
    test_df[c] = test_df[c].fillna("").astype(str)

for c in numeric_cols:
    med = train_df[c].median() if not train_df[c].dropna().empty else 0.0
    train_df[c] = train_df[c].fillna(med)
    test_df[c] = test_df[c].fillna(med)


# ----------------------------------------------------
# 2. Domain-Specific Tabular Feature Engineering
# ----------------------------------------------------
def engineer_tabular_features(df_source):
    df = df_source.copy()
    feats = pd.DataFrame(index=df.index)

    if "substance_name" in df.columns:
        feats["substance_count"] = df["substance_name"].apply(
            lambda x: len(re.split(r"[,;/|]", x)) if x else 0
        )
        feats["substance_char_len"] = df["substance_name"].apply(len)
        feats["substance_word_len"] = df["substance_name"].apply(
            lambda x: len(x.split())
        )

    if "pharm_classes" in df.columns:
        feats["pharm_class_count"] = df["pharm_classes"].apply(
            lambda x: len(re.findall(r"\[.*?\]", x)) if x else 0
        )
        feats["pharm_class_len"] = df["pharm_classes"].apply(len)
        feats["has_opioid"] = (
            df["pharm_classes"]
            .str.contains("opioid|opiate|narcotic", case=False, regex=True)
            .astype(float)
        )
        feats["has_cns"] = (
            df["pharm_classes"]
            .str.contains(
                "central nervous system|sedative|hypnotic|stimulant",
                case=False,
                regex=True,
            )
            .astype(float)
        )
        feats["has_gaba"] = (
            df["pharm_classes"]
            .str.contains("gaba|benzodiazepine|barbiturate", case=False, regex=True)
            .astype(float)
        )
        feats["has_cannabinoid"] = (
            df["pharm_classes"]
            .str.contains("cannabinoid", case=False, regex=True)
            .astype(float)
        )

    if "nonproprietary_name" in df.columns:
        feats["nonprop_char_len"] = df["nonproprietary_name"].apply(len)
        feats["nonprop_word_len"] = df["nonproprietary_name"].apply(
            lambda x: len(x.split())
        )

    if "proprietary_name" in df.columns:
        feats["prop_char_len"] = df["proprietary_name"].apply(len)
        feats["prop_word_len"] = df["proprietary_name"].apply(lambda x: len(x.split()))

    if "route_name" in df.columns and "dosage_form_name" in df.columns:
        feats["route_form_combo"] = df["route_name"] + "__" + df["dosage_form_name"]

    for col in df.columns:
        if "strength" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            feats[f"{col}_log1p"] = np.log1p(np.maximum(0, df[col].values))

    return feats


tab_train = engineer_tabular_features(train_df)
tab_test = engineer_tabular_features(test_df)

cat_fields = [
    c
    for c in [
        "dosage_form_name",
        "route_name",
        "marketing_category_name",
        "labeler_name",
        "route_form_combo",
    ]
    if c in train_df.columns or c in tab_train.columns
]

for col in cat_fields:
    series_tr = train_df[col] if col in train_df.columns else tab_train[col]
    series_te = test_df[col] if col in test_df.columns else tab_test[col]

    freq_map = series_tr.value_counts(normalize=True).to_dict()
    tab_train[f"{col}_freq"] = series_tr.map(freq_map).fillna(0).astype(float)
    tab_test[f"{col}_freq"] = series_te.map(freq_map).fillna(0).astype(float)

if "route_form_combo" in tab_train.columns:
    tab_train = tab_train.drop(columns=["route_form_combo"])
    tab_test = tab_test.drop(columns=["route_form_combo"])

remaining_cols = [c for c in feature_cols if c not in tab_train.columns]
for col in remaining_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        tab_train[col] = train_df[col].values
        tab_test[col] = test_df[col].values

tab_train = tab_train.fillna(0.0)
tab_test = tab_test.fillna(0.0)


# ----------------------------------------------------
# 3. Text Representation Helpers
# ----------------------------------------------------
def create_text_corpus(df):
    cols_to_use = [
        c
        for c in df.columns
        if c not in [target_col, "id"] and not pd.api.types.is_numeric_dtype(df[c])
    ]
    corpus = []
    for _, row in df.iterrows():
        parts = [str(row[c]) for c in cols_to_use if str(row[c]).strip() != ""]
        text = " ".join(parts).strip()
        corpus.append(text if text else "missing_drug_info")
    return corpus


text_corpus_train = np.array(create_text_corpus(train_df))
text_corpus_test = np.array(create_text_corpus(test_df))
svd_dim = 64


# ----------------------------------------------------
# 4. Neural Network and Model Definitions
# ----------------------------------------------------
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act(self.fc1(self.norm1(x))))
        out = self.drop2(self.fc2(self.norm2(out)))
        return out + residual


class PharmaceuticalResMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [256, 128],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate) for _ in range(2)]
        )

        self.transition = nn.Sequential(
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dims[1]), nn.Linear(hidden_dims[1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        h = self.transition(h)
        return self.head(h)


def get_criterion(label_smoothing: float = 0.05) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def get_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)


def get_scheduler(
    optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int
):
    total_steps = num_epochs * steps_per_epoch
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, total_steps // 2), T_mult=1, eta_min=1e-6
    )


def get_lgb_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 15,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


# ----------------------------------------------------
# 5. Stratified Cross-Validation & Model Training
# ----------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
n_train = len(train_df)
n_test = len(test_df)

oof_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)

oof_mlp = np.zeros((n_train, num_classes), dtype=np.float32)
test_mlp = np.zeros((n_test, num_classes), dtype=np.float32)

lgb_params = get_lgb_params(num_classes=num_classes, seed=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(tab_train, y_train)):
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=2500,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=2500,
        sublinear_tf=True,
        min_df=1,
    )

    w_tr_sparse = word_vec.fit_transform(text_corpus_train[train_idx])
    w_va_sparse = word_vec.transform(text_corpus_train[val_idx])
    w_te_sparse = word_vec.transform(text_corpus_test)

    c_tr_sparse = char_vec.fit_transform(text_corpus_train[train_idx])
    c_va_sparse = char_vec.transform(text_corpus_train[val_idx])
    c_te_sparse = char_vec.transform(text_corpus_test)

    n_comp_w = min(svd_dim, max(1, w_tr_sparse.shape[1] - 1))
    svd_word = TruncatedSVD(n_components=n_comp_w, random_state=SEED)
    w_tr = svd_word.fit_transform(w_tr_sparse)
    w_va = svd_word.transform(w_va_sparse)
    w_te = svd_word.transform(w_te_sparse)

    n_comp_c = min(svd_dim, max(1, c_tr_sparse.shape[1] - 1))
    svd_char = TruncatedSVD(n_components=n_comp_c, random_state=SEED)
    c_tr = svd_char.fit_transform(c_tr_sparse)
    c_va = svd_char.transform(c_va_sparse)
    c_te = svd_char.transform(c_te_sparse)

    X_tr_raw = np.hstack([tab_train.iloc[train_idx].values, w_tr, c_tr])
    X_va_raw = np.hstack([tab_train.iloc[val_idx].values, w_va, c_va])
    X_te_raw = np.hstack([tab_test.values, w_te, c_te])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)
    X_dense_test_arr = scaler.transform(X_te_raw)

    y_tr = y_train[train_idx]
    y_va = y_train[val_idx]

    # --- Train LightGBM ---
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    test_lgb += (
        lgb_model.predict(X_dense_test_arr, num_iteration=lgb_model.best_iteration)
        / n_splits
    )

    # --- Train PharmaceuticalResMLP ---
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_va_t, y_va_t), batch_size=128, shuffle=False
    )

    mlp_model = PharmaceuticalResMLP(
        input_dim=X_tr.shape[1],
        num_classes=num_classes,
        hidden_dims=[256, 128],
        dropout_rate=0.25,
    ).to(device)

    criterion = get_criterion(label_smoothing=0.05)
    optimizer = get_optimizer(mlp_model, lr=1e-3, weight_decay=1e-4)
    epochs = 40
    scheduler = get_scheduler(
        optimizer, num_epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        mlp_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = mlp_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = mlp_model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(mlp_model.state_dict())

    mlp_model.load_state_dict(best_weights)
    mlp_model.eval()

    with torch.no_grad():
        val_logits = mlp_model(X_va_t.to(device))
        oof_mlp[val_idx] = torch.softmax(val_logits, dim=-1).cpu().numpy()

        test_t = torch.tensor(X_dense_test_arr, dtype=torch.float32).to(device)
        test_logits = mlp_model(test_t)
        test_mlp += torch.softmax(test_logits, dim=-1).cpu().numpy() / n_splits

    fold_lgb_acc = accuracy_score(y_va, np.argmax(oof_lgb[val_idx], axis=1))
    fold_mlp_acc = accuracy_score(y_va, np.argmax(oof_mlp[val_idx], axis=1))
    print(
        f"Fold {fold+1}/{n_splits} | LGB Acc: {fold_lgb_acc:.4f} | ResMLP Acc: {fold_mlp_acc:.4f}"
    )

# ----------------------------------------------------
# 6. Ensemble Optimization & Final Prediction
# ----------------------------------------------------
best_alpha = 0.5
best_ensemble_acc = 0.0

for alpha in np.linspace(0.0, 1.0, 101):
    blended_oof = alpha * oof_lgb + (1.0 - alpha) * oof_mlp
    acc = accuracy_score(y_train, np.argmax(blended_oof, axis=1))
    if acc > best_ensemble_acc:
        best_ensemble_acc = acc
        best_alpha = alpha

print(
    f"Ensemble (LGB alpha={best_alpha:.2f}, MLP={1-best_alpha:.2f}) Accuracy: {best_ensemble_acc:.5f}"
)

final_test_probs = best_alpha * test_lgb + (1.0 - best_alpha) * test_mlp
final_test_pred_idx = np.argmax(final_test_probs, axis=1)
final_test_pred_labels = le.inverse_transform(final_test_pred_idx)

sub_path = "./submission/submission.csv"
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df), dtype=int), "dea_schedule": final_test_pred_labels}
)

submission_df.to_csv(sub_path, index=False)

assert os.path.exists(sub_path), f"Submission file {sub_path} was not created"
assert len(submission_df) == len(
    test_df
), f"Submission length {len(submission_df)} != test length {len(test_df)}"
assert submission_df.columns.tolist() == [
    "id",
    "dea_schedule",
], f"Incorrect submission columns {submission_df.columns}"
assert (
    not submission_df["dea_schedule"].isnull().any()
), "Submission contains null predictions"

final_validation_score = float(best_ensemble_acc)
print(f"Final Validation Score: {final_validation_score}")