import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Setup Directories & Seeds
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

# 2. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_raw = pd.read_parquet(train_path)
test_raw = pd.read_parquet(test_path)

target_col = "DEATHS_TOTAL_DESCRIPTION"
if target_col not in train_raw.columns:
    raise ValueError(f"Target column {target_col} not found in train data.")

# Clean missing target rows if any
valid_target_idx = train_raw[target_col].notna()
train_raw = train_raw.loc[valid_target_idx].reset_index(drop=True)

# Encode Target Column
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(train_raw[target_col].astype(str))
target_classes = target_encoder.classes_.tolist()
num_classes = len(target_classes)

# Stratified Train/Val Split prior to any stateful transformations
train_df, val_df, y_train, y_val = train_test_split(
    train_raw.drop(columns=[target_col], errors="ignore"),
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_raw.copy().reset_index(drop=True)


def detect_spatial_cols(df):
    lat_col, lon_col, elev_col = None, None, None
    for col in df.columns:
        col_lower = col.lower()
        if "lat" in col_lower and lat_col is None:
            lat_col = col
        elif ("lon" in col_lower or "lng" in col_lower) and lon_col is None:
            lon_col = col
        elif ("elev" in col_lower or "altitude" in col_lower) and elev_col is None:
            elev_col = col
    return lat_col, lon_col, elev_col


lat_col, lon_col, elev_col = detect_spatial_cols(train_df)


def create_geospatial_features(df):
    df_feat = df.copy()

    for c in df.columns:
        if df[c].isna().sum() > 0:
            df_feat[f"{c}_isna"] = df[c].isna().astype(int)

    df_feat["missing_count_row"] = df.isna().sum(axis=1)

    if lat_col and lon_col:
        lat_rad = np.radians(pd.to_numeric(df[lat_col], errors="coerce").fillna(0))
        lon_rad = np.radians(pd.to_numeric(df[lon_col], errors="coerce").fillna(0))

        df_feat["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        df_feat["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        df_feat["coord_z"] = np.sin(lat_rad)
        df_feat["abs_latitude"] = np.abs(
            pd.to_numeric(df[lat_col], errors="coerce").fillna(0)
        )
        df_feat["abs_longitude"] = np.abs(
            pd.to_numeric(df[lon_col], errors="coerce").fillna(0)
        )

    if elev_col:
        elev_num = pd.to_numeric(df[elev_col], errors="coerce").fillna(0)
        df_feat["elev_log"] = np.log1p(np.maximum(0, elev_num))
        df_feat["elev_is_negative"] = (elev_num < 0).astype(int)

    return df_feat


train_feat = create_geospatial_features(train_df)
val_feat = create_geospatial_features(val_df)
test_feat = create_geospatial_features(test_df)

all_cols = sorted(
    list(set(train_feat.columns) | set(val_feat.columns) | set(test_feat.columns))
)
for df in [train_feat, val_feat, test_feat]:
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0

train_feat = train_feat[all_cols]
val_feat = val_feat[all_cols]
test_feat = test_feat[all_cols]

categorical_cols = [
    c
    for c in all_cols
    if train_feat[c].dtype == "object" or train_feat[c].dtype.name == "category"
]

# Frequency & Group Aggregations strictly fitted on Train
for cat in categorical_cols:
    freq_map = train_feat[cat].value_counts(normalize=True).to_dict()
    train_feat[f"{cat}_freq"] = train_feat[cat].map(freq_map).fillna(0)
    val_feat[f"{cat}_freq"] = val_feat[cat].map(freq_map).fillna(0)
    test_feat[f"{cat}_freq"] = test_feat[cat].map(freq_map).fillna(0)

    if elev_col and elev_col in train_feat.columns:
        elev_num_train = pd.to_numeric(train_feat[elev_col], errors="coerce").fillna(0)
        group_mean = (
            train_feat.assign(elev_temp=elev_num_train)
            .groupby(cat)["elev_temp"]
            .mean()
            .to_dict()
        )
        train_feat[f"{cat}_elev_mean"] = train_feat[cat].map(group_mean).fillna(0)
        val_feat[f"{cat}_elev_mean"] = val_feat[cat].map(group_mean).fillna(0)
        test_feat[f"{cat}_elev_mean"] = test_feat[cat].map(group_mean).fillna(0)

# Label Encoding
for cat in categorical_cols:
    le = LabelEncoder()
    train_feat[cat] = train_feat[cat].astype(str)
    val_feat[cat] = val_feat[cat].astype(str)
    test_feat[cat] = test_feat[cat].astype(str)

    le.fit(train_feat[cat])
    train_feat[cat] = train_feat[cat].map(
        lambda s: le.transform([s])[0] if s in le.classes_ else -1
    )
    val_feat[cat] = val_feat[cat].map(
        lambda s: le.transform([s])[0] if s in le.classes_ else -1
    )
    test_feat[cat] = test_feat[cat].map(
        lambda s: le.transform([s])[0] if s in le.classes_ else -1
    )

updated_num_cols = [c for c in train_feat.columns if train_feat[c].dtype != "object"]

# Imputation & Standard Scaling
for col in updated_num_cols:
    median_val = train_feat[col].median()
    if pd.isna(median_val):
        median_val = 0
    train_feat[col] = train_feat[col].fillna(median_val)
    val_feat[col] = val_feat[col].fillna(median_val)
    test_feat[col] = test_feat[col].fillna(median_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_feat[updated_num_cols])
X_val = scaler.transform(val_feat[updated_num_cols])
X_test = scaler.transform(test_feat[updated_num_cols])

X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

num_features = X_train.shape[1]


# 3. Neural Architecture Definition
class TabularResBlock(nn.Module):

    def __init__(self, in_features, hidden_features, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.LayerNorm(hidden_features)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.LayerNorm(in_features)
        self.drop2 = nn.Dropout(dropout_rate)
        self.act2 = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.act2(out + residual)
        return out


class VolcanicTabularResNet(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits


# 4. Train LightGBM Model
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_data_in_leaf": 10,
    "verbosity": -1,
    "random_state": 42,
}

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

lgb_val_probs = lgb_model.predict(X_val)
lgb_test_probs = lgb_model.predict(X_test)

# 5. Train XGBoost Model
xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "random_state": 42,
}

xgb_model = xgb.XGBClassifier(
    **xgb_params,
    n_estimators=1000,
    early_stopping_rounds=50,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_val_probs = xgb_model.predict_proba(X_val)
xgb_test_probs = xgb_model.predict_proba(X_test)

# 6. Train PyTorch Tabular ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = VolcanicTabularResNet(
    input_dim=num_features,
    num_classes=num_classes,
    hidden_dim=128,
    num_blocks=2,
    dropout_rate=0.3,
).to(device)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

best_val_loss = float("inf")
best_nn_weights = None

epochs = 80
for epoch in range(epochs):
    nn_model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = nn_model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = nn_model(batch_x)
            val_loss += criterion(out, batch_y).item() * batch_x.size(0)
    val_loss /= len(val_dataset)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_nn_weights = nn_model.state_dict().copy()

if best_nn_weights is not None:
    nn_model.load_state_dict(best_nn_weights)
nn_model.eval()

nn_val_probs_list = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        logits = nn_model(batch_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        nn_val_probs_list.append(probs)
nn_val_probs = np.vstack(nn_val_probs_list)

nn_test_probs_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = nn_model(batch_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        nn_test_probs_list.append(probs)
nn_test_probs = np.vstack(nn_test_probs_list)

# 7. Optimize Model Blending Weights
best_blend_acc = 0.0
best_weights = (0.33, 0.33, 0.34)

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        w3 = max(0.0, 1.0 - w1 - w2)
        if abs(w1 + w2 + w3 - 1.0) > 1e-5:
            continue
        blend_probs = w1 * lgb_val_probs + w2 * xgb_val_probs + w3 * nn_val_probs
        acc = accuracy_score(y_val, np.argmax(blend_probs, axis=1))
        if acc > best_blend_acc:
            best_blend_acc = acc
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights

# 8. Generate Final Submission
final_test_probs = w1 * lgb_test_probs + w2 * xgb_test_probs + w3 * nn_test_probs
final_test_preds = np.argmax(final_test_probs, axis=1)

test_pred_labels = [target_classes[idx] for idx in final_test_preds]

sub = pd.DataFrame(
    {
        "id": np.arange(len(test_pred_labels)),
        "DEATHS_TOTAL_DESCRIPTION": test_pred_labels,
    }
)
sub.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {best_blend_acc}")
