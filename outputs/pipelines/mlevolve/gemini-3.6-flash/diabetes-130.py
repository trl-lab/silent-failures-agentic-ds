import json
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 0. Reproducibility & Device Configuration
# ---------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Load Parquet Datasets
# ---------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Standardize missing value tokens
for df in [df_train, df_test]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(
                {"?": "Missing", "Unknown/Invalid": "Missing", None: "Missing"}
            )

# Target mapping
target_col = "readmitted"
target_map = {"NO": 0, ">30": 1, "<30": 2}
inv_target_map = {0: "NO", 1: ">30", 2: "<30"}

if target_col in df_train.columns:
    df_train["target"] = df_train[target_col].map(target_map)
    df_train = df_train.dropna(subset=["target"])
    df_train["target"] = df_train["target"].astype(int)

# ---------------------------------------------------------
# 2. Clinical Domain & Trajectory Feature Engineering
# ---------------------------------------------------------
med_cols = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def parse_icd9_group(code):
    if pd.isna(code) or str(code) == "Missing":
        return "Missing"
    code_str = str(code).strip()
    if code_str.startswith("V"):
        return "V_code"
    if code_str.startswith("E"):
        return "E_code"
    try:
        val = float(code_str.split(".")[0])
        if val == 250:
            return "Diabetes"
        elif (390 <= val <= 459) or val == 785:
            return "Circulatory"
        elif (460 <= val <= 519) or val == 786:
            return "Respiratory"
        elif (520 <= val <= 579) or val == 787:
            return "Digestive"
        elif (580 <= val <= 629) or val == 788:
            return "Genitourinary"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif 140 <= val <= 239:
            return "Neoplasms"
        elif 800 <= val <= 999:
            return "Injury"
        else:
            return "Other"
    except ValueError:
        return "Other"


def engineer_features(df):
    df = df.copy()

    # ICD-9 broad clinical groupings
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[f"{diag_col}_group"] = df[diag_col].apply(parse_icd9_group)

    # Medication changes and active status
    available_meds = [c for c in med_cols if c in df.columns]
    if available_meds:
        df["num_med_up"] = (df[available_meds] == "Up").sum(axis=1)
        df["num_med_down"] = (df[available_meds] == "Down").sum(axis=1)
        df["num_med_steady"] = (df[available_meds] == "Steady").sum(axis=1)
        df["num_med_changed"] = df["num_med_up"] + df["num_med_down"]
        df["num_med_active"] = (df[available_meds] != "No").sum(axis=1)
        df["polypharmacy_flag"] = (df["num_med_active"] > 2).astype(int)

    # Healthcare utilization & operational ratios
    df["total_prior_visits"] = (
        df["number_outpatient"].fillna(0)
        + df["number_emergency"].fillna(0)
        + df["number_inpatient"].fillna(0)
    )
    df["emergency_visit_ratio"] = df["number_emergency"] / (
        df["total_prior_visits"] + 1.0
    )
    df["inpatient_visit_ratio"] = df["number_inpatient"] / (
        df["total_prior_visits"] + 1.0
    )

    time_hosp = df["time_in_hospital"].clip(lower=1)
    df["lab_procedures_per_day"] = df["num_lab_procedures"] / time_hosp
    df["medications_per_day"] = df["num_medications"] / time_hosp
    df["procedures_per_day"] = df["num_procedures"] / time_hosp
    df["diagnoses_per_day"] = df["number_diagnoses"] / time_hosp

    # Ordinal mappings for clinical status indicators
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3, "Missing": 0}
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3, "Missing": 0}
    age_map = {
        "[0-10)": 0,
        "[10-20)": 1,
        "[20-30)": 2,
        "[30-40)": 3,
        "[40-50)": 4,
        "[50-60)": 5,
        "[60-70)": 6,
        "[70-80)": 7,
        "[80-90)": 8,
        "[90-100)": 9,
        "Missing": 5,
    }

    if "A1Cresult" in df.columns:
        df["A1Cresult_num"] = df["A1Cresult"].map(a1c_map).fillna(0)
    if "max_glu_serum" in df.columns:
        df["max_glu_serum_num"] = df["max_glu_serum"].map(glu_map).fillna(0)
    if "age" in df.columns:
        df["age_num"] = df["age"].map(age_map).fillna(5)

    if "change" in df.columns:
        df["change_num"] = (df["change"] == "Ch").astype(int)
    if "diabetesMed" in df.columns:
        df["diabetesMed_num"] = (df["diabetesMed"] == "Yes").astype(int)

    if "A1Cresult_num" in df.columns and "change_num" in df.columns:
        df["uncontrolled_diabetes_risk"] = (
            (df["A1Cresult_num"] >= 2) & (df["change_num"] == 0)
        ).astype(int)

    return df


df_train_fe = engineer_features(df_train)
df_test_fe = engineer_features(df_test)

# ---------------------------------------------------------
# 3. Patient Group-Aware Split (Leakage Prevention)
# ---------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
groups = (
    df_train_fe["patient_nbr"].values
    if "patient_nbr" in df_train_fe.columns
    else np.arange(len(df_train_fe))
)
y_all = df_train_fe["target"].values

train_idx, val_idx = next(sgkf.split(df_train_fe, y_all, groups=groups))

df_tr = df_train_fe.iloc[train_idx].reset_index(drop=True)
df_va = df_train_fe.iloc[val_idx].reset_index(drop=True)
df_te = df_test_fe.copy().reset_index(drop=True)

# ---------------------------------------------------------
# 4. Strict Categorical Encodings Fit ON TRAIN ONLY
# ---------------------------------------------------------
cat_cols = [
    "race",
    "gender",
    "age",
    "weight",
    "payer_code",
    "medical_specialty",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]
cat_cols = [c for c in cat_cols if c in df_tr.columns]

freq_encodings = {}
label_encoders = {}

for col in cat_cols:
    # Frequency Encoding
    freq_map = df_tr[col].astype(str).value_counts(normalize=True).to_dict()
    freq_encodings[col] = freq_map

    df_tr[f"{col}_freq"] = df_tr[col].astype(str).map(freq_map).fillna(0.0)
    df_va[f"{col}_freq"] = df_va[col].astype(str).map(freq_map).fillna(0.0)
    df_te[f"{col}_freq"] = df_te[col].astype(str).map(freq_map).fillna(0.0)

    # Label Encoding
    le = LabelEncoder()
    le.fit(df_tr[col].astype(str).values)
    label_encoders[col] = le

    classes_set = set(le.classes_)
    fallback_val = le.classes_[0]

    def safe_transform_series(s):
        return s.astype(str).map(lambda x: x if x in classes_set else fallback_val)

    df_tr[f"{col}_le"] = le.transform(safe_transform_series(df_tr[col]))
    df_va[f"{col}_le"] = le.transform(safe_transform_series(df_va[col]))
    df_te[f"{col}_le"] = le.transform(safe_transform_series(df_te[col]))

ignore_cols = ["encounter_id", "patient_nbr", "target", "readmitted"] + cat_cols
features = [
    c
    for c in df_tr.columns
    if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_tr[c])
]

X_tr = df_tr[features].fillna(0).values.astype(np.float32)
y_tr = df_tr["target"].values.astype(int)

X_va = df_va[features].fillna(0).values.astype(np.float32)
y_va = df_va["target"].values.astype(int)

X_te = df_te[features].fillna(0).values.astype(np.float32)

# Save artifact metadata
os.makedirs("./working", exist_ok=True)
with open("./working/feature_names.json", "w") as f:
    json.dump(features, f)

# Standardize features for PyTorch ResNet
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)
X_te_scaled = scaler.transform(X_te)


# ---------------------------------------------------------
# 5. Neural Network Architecture Definitions
# ---------------------------------------------------------
class DenseResBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [DenseResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


class MultiClassFocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------
# 6. Model Training Sequence
# ---------------------------------------------------------
# Model A: LightGBM Classifier
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
)

val_prob_lgb = lgb_model.predict_proba(X_va)
test_prob_lgb = lgb_model.predict_proba(X_te)

# Model B: XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=40,
)

xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

val_prob_xgb = xgb_model.predict_proba(X_va)
test_prob_xgb = xgb_model.predict_proba(X_te)

# Model C: HistGradientBoosting Classifier
hgb_model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=31,
    max_depth=6,
    l2_regularization=1.0,
    random_state=SEED,
    early_stopping=True,
    n_iter_no_change=25,
)
hgb_model.fit(X_tr, y_tr)

val_prob_hgb = hgb_model.predict_proba(X_va)
test_prob_hgb = hgb_model.predict_proba(X_te)

# Model D: PyTorch Tabular ResNet
tr_dataset = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.long),
)
va_dataset = TensorDataset(
    torch.tensor(X_va_scaled, dtype=torch.float32),
    torch.tensor(y_va, dtype=torch.long),
)

tr_loader = DataLoader(tr_dataset, batch_size=256, shuffle=True, drop_last=False)
va_loader = DataLoader(va_dataset, batch_size=512, shuffle=False, drop_last=False)

nn_model = TabularResNet(
    input_dim=X_tr.shape[1],
    num_classes=3,
    hidden_dim=256,
    num_blocks=3,
    dropout_rate=0.2,
).to(device)

criterion = MultiClassFocalLoss(gamma=2.0, label_smoothing=0.05)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25, eta_min=1e-5
)

best_val_loss = float("inf")
best_nn_weights = None

for epoch in range(1, 26):
    nn_model.train()
    train_loss = 0.0
    for bx, by in tr_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(by)
    scheduler.step()

    train_loss /= len(tr_dataset)

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in va_loader:
            bx, by = bx.to(device), by.to(device)
            out = nn_model(bx)
            loss = criterion(out, by)
            val_loss += loss.item() * len(by)
    val_loss /= len(va_dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_nn_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    print(
        f"Epoch {epoch:02d}/25 | Train Loss: {train_loss:.4f} | Val Loss:"
        f" {val_loss:.4f}"
    )

if best_nn_weights is not None:
    nn_model.load_state_dict(best_nn_weights)

nn_model.eval()
with torch.no_grad():
    val_logits = []
    for bx, _ in va_loader:
        bx = bx.to(device)
        val_logits.append(nn_model(bx).cpu())
    val_prob_nn = F.softmax(torch.cat(val_logits, dim=0), dim=1).numpy()

    te_dataset = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))
    te_loader = DataLoader(te_dataset, batch_size=512, shuffle=False, drop_last=False)
    test_logits = []
    for (bx,) in te_loader:
        bx = bx.to(device)
        test_logits.append(nn_model(bx).cpu())
    test_prob_nn = F.softmax(torch.cat(test_logits, dim=0), dim=1).numpy()

# ---------------------------------------------------------
# 7. Constrained Optimization & Ensembling
# ---------------------------------------------------------
val_probs = [val_prob_lgb, val_prob_xgb, val_prob_hgb, val_prob_nn]
test_probs = [test_prob_lgb, test_prob_xgb, test_prob_hgb, test_prob_nn]


def objective_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = sum(wi * pi for wi, pi in zip(w, val_probs))
    preds = np.argmax(blend, axis=1)
    return -accuracy_score(y_va, preds)


init_weights = [0.35, 0.35, 0.15, 0.15]
bounds = [(0, 1)] * 4
res = minimize(
    objective_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    options={"maxiter": 100},
)

best_weights = res.x / np.sum(res.x)

val_prob_ens = sum(w * p for w, p in zip(best_weights, val_probs))
val_preds = np.argmax(val_prob_ens, axis=1)
val_score = accuracy_score(y_va, val_preds)

test_prob_ens = sum(w * p for w, p in zip(best_weights, test_probs))
test_preds = np.argmax(test_prob_ens, axis=1)

# ---------------------------------------------------------
# 8. Output Submission & Verification
# ---------------------------------------------------------
test_labels = [inv_target_map[p] for p in test_preds]

os.makedirs("./submission", exist_ok=True)
sub_df = pd.DataFrame({"id": np.arange(len(test_labels)), "readmitted": test_labels})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_score}")
