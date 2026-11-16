import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Separate target
target = train["NObeyesdad"].copy()
test_ids = test["id"].copy()

# Remove id and target from feature frames
train_features = train.drop(columns=["id", "NObeyesdad"])
test_features = test.drop(columns=["id"]) if "id" in test.columns else test.copy()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
ordinal_mappings = {
    "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2},
    "FAVC": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
}

categorical_mappings = {
    "Gender": {"Female": 0, "Male": 1},
    "MTRANS": {
        "Walking": 0,
        "Bike": 1,
        "Public_Transportation": 2,
        "Motorbike": 3,
        "Automobile": 4,
    },
}


def engineer_features(df, bmi_mean, bmi_std, w_mean, w_std, h_mean, h_std):
    """Apply feature engineering to a dataframe."""
    df = df.copy()

    # 1. Encode ordinal/categorical features
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(np.int8)

    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(np.int8)

    # 2. BMI and related features
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["BMI_squared"] = df["BMI"] ** 2
    df["BMI_cubed"] = df["BMI"] ** 3
    df["BMI_log"] = np.log1p(df["BMI"])
    df["BMI_inv"] = 1.0 / (df["BMI"] + 1e-8)

    # 3. Height-Weight interactions
    df["HW_ratio"] = df["Weight"] / df["Height"]
    df["HW_product"] = df["Weight"] * df["Height"]
    df["HW_diff"] = df["Weight"] - df["Height"] * 100

    # 4. Composite lifestyle scores
    df["unhealthy_eat_score"] = df["CAEC"] + df["CALC"] + df["FAVC"]
    df["healthy_eat_score"] = df["FCVC"] + df["SCC"]
    df["activity_score"] = df["TUE"] + df["NCP"] + df["FCVC"]
    df["sedentary_score"] = df["MTRANS"] + df["TUE"]
    df["lifestyle_risk"] = (
        df["unhealthy_eat_score"] + df["sedentary_score"] + df["SMOKE"]
    )

    # 5. BMI x Lifestyle interactions
    df["BMI_x_unhealthy_eat"] = df["BMI"] * df["unhealthy_eat_score"]
    df["BMI_x_activity"] = df["BMI"] * df["activity_score"]
    df["BMI_x_lifestyle_risk"] = df["BMI"] * df["lifestyle_risk"]
    df["BMI_x_smoke"] = df["BMI"] * df["SMOKE"]
    df["BMI_x_family"] = df["BMI"] * df["family_history_with_overweight"]
    df["BMI_x_age"] = df["BMI"] * df["Age"]

    # 6. Age-related features
    df["Age_squared"] = df["Age"] ** 2
    df["Age_cubed"] = df["Age"] ** 3
    df["Age_group"] = pd.cut(df["Age"], bins=[0, 20, 30, 40, 50, 100], labels=False)

    # 7. BMI z-score
    df["BMI_zscore"] = (df["BMI"] - bmi_mean) / (bmi_std + 1e-8)

    # 8. BMI bins
    df["BMI_bin"] = pd.qcut(df["BMI"], q=10, labels=False, duplicates="drop")

    # 9. Weight z-score
    df["Weight_zscore"] = (df["Weight"] - w_mean) / (w_std + 1e-8)

    # 10. Height z-score
    df["Height_zscore"] = (df["Height"] - h_mean) / (h_std + 1e-8)

    # 11. Energy balance proxy
    df["energy_balance"] = df["unhealthy_eat_score"] * 2 - df["activity_score"]

    # 12. Family history x BMI
    df["family_BMI"] = df["family_history_with_overweight"] * df["BMI"]

    # 13. Gender x BMI
    df["gender_BMI"] = df["Gender"] * df["BMI"]

    # 14. Transport x BMI
    df["transport_BMI"] = df["MTRANS"] * df["BMI"]

    # 15. Ratio features
    df["eat_activity_ratio"] = df["unhealthy_eat_score"] / (df["activity_score"] + 1e-8)
    df["sedentary_activity_ratio"] = df["sedentary_score"] / (
        df["activity_score"] + 1e-8
    )

    return df


# Create train/val split (stratified on target) BEFORE feature engineering
X_all = train_features.reset_index(drop=True)
y_all = target.reset_index(drop=True)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Compute statistics from TRAIN portion only (no leakage)
train_temp = X_train_raw.copy()
for col, mapping in ordinal_mappings.items():
    if col in train_temp.columns:
        train_temp[col] = train_temp[col].map(mapping).fillna(0).astype(np.int8)
for col, mapping in categorical_mappings.items():
    if col in train_temp.columns:
        train_temp[col] = train_temp[col].map(mapping).fillna(0).astype(np.int8)

train_temp["BMI"] = train_temp["Weight"] / (train_temp["Height"] ** 2)

bmi_mean = train_temp["BMI"].mean()
bmi_std = train_temp["BMI"].std()
w_mean = X_train_raw["Weight"].mean()
w_std = X_train_raw["Weight"].std()
h_mean = X_train_raw["Height"].mean()
h_std = X_train_raw["Height"].std()

# Apply full feature engineering using train-only statistics
X_train = engineer_features(
    X_train_raw, bmi_mean, bmi_std, w_mean, w_std, h_mean, h_std
).reset_index(drop=True)
X_val = engineer_features(
    X_val_raw, bmi_mean, bmi_std, w_mean, w_std, h_mean, h_std
).reset_index(drop=True)
test_processed = engineer_features(
    test_features, bmi_mean, bmi_std, w_mean, w_std, h_mean, h_std
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {test_processed.shape}")

# ============================================================
# ORDINAL CLASS MAPPING
# ============================================================
class_order = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
class_to_idx = {c: i for i, c in enumerate(class_order)}
idx_to_class = {i: c for i, c in enumerate(class_order)}
n_classes = len(class_order)
n_thresholds = n_classes - 1

y_train_idx = y_train.map(class_to_idx).values.astype(np.int64)
y_val_idx = y_val.map(class_to_idx).values.astype(np.int64)

print(f"Train class distribution: {np.bincount(y_train_idx, minlength=n_classes)}")

# ============================================================
# NORMALIZE FEATURES (fit on train only)
# ============================================================
feature_mean = X_train.mean().values
feature_std = X_train.std().values + 1e-8

X_train_np = ((X_train.values - feature_mean) / feature_std).astype(np.float32)
X_val_np = ((X_val.values - feature_mean) / feature_std).astype(np.float32)
X_test_np = ((test_processed.values - feature_mean) / feature_std).astype(np.float32)

n_features = X_train_np.shape[1]
print(f"Number of features: {n_features}")


# ============================================================
# ORDINAL NEURAL NETWORK (Cumulative Link Model)
# ============================================================
class OrdinalNN(nn.Module):
    def __init__(
        self, n_features, n_thresholds, hidden_dims=[128, 64, 32], dropout=0.3
    ):
        super().__init__()
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(prev_dim, 1)
        self.thresholds = nn.Parameter(torch.zeros(n_thresholds))
        with torch.no_grad():
            self.thresholds.copy_(torch.linspace(-2, 2, n_thresholds))

    def forward(self, x):
        h = self.trunk(x)
        score = self.score_head(h).squeeze(-1)
        logits = score.unsqueeze(-1) - self.thresholds.unsqueeze(0)
        return logits


def ordinal_loss(logits, targets, monotonicity_weight=0.1):
    n_thresh = logits.shape[1]
    k = torch.arange(1, n_thresh + 1, device=targets.device).unsqueeze(0)
    target_matrix = (targets.unsqueeze(1) >= k).float()
    bce = F.binary_cross_entropy_with_logits(logits, target_matrix, reduction="mean")
    if n_thresh > 1:
        violations = F.relu(logits[:, 1:] - logits[:, :-1])
        mono_penalty = violations.mean()
    else:
        mono_penalty = torch.tensor(0.0, device=logits.device)
    return bce + monotonicity_weight * mono_penalty, bce, mono_penalty


def predict_class(logits):
    probs = torch.sigmoid(logits)
    p0 = 1.0 - probs[:, 0:1]
    p_mid = probs[:, :-1] * (1.0 - probs[:, 1:])
    p_last = probs[:, -1:]
    class_probs = torch.cat([p0, p_mid, p_last], dim=1)
    return class_probs.argmax(dim=1)


# ============================================================
# SETUP DATA LOADERS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = TensorDataset(
    torch.from_numpy(X_train_np), torch.from_numpy(y_train_idx)
)
val_dataset = TensorDataset(torch.from_numpy(X_val_np), torch.from_numpy(y_val_idx))
test_dataset = TensorDataset(torch.from_numpy(X_test_np))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

# ============================================================
# INITIALIZE MODEL, OPTIMIZER, SCHEDULER
# ============================================================
model = OrdinalNN(
    n_features=n_features,
    n_thresholds=n_thresholds,
    hidden_dims=[128, 64, 32],
    dropout=0.3,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-5
)

# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_val_acc = 0.0
best_model_state = None
patience = 25
patience_counter = 0
max_epochs = 200

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss, bce, mono = ordinal_loss(logits, y_batch, monotonicity_weight=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Validation
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = predict_class(logits)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.numpy())

    val_acc = accuracy_score(val_targets, val_preds)
    avg_loss = total_loss / n_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"Best OrdinalNN Val Accuracy: {best_val_acc:.4f}")

# ============================================================
# LIGHTGBM BASELINE
# ============================================================
print("\n--- Training LightGBM baseline ---")
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

lgb_val_preds = lgb_model.predict(X_val)
lgb_val_acc = accuracy_score(y_val_idx, np.array([class_to_idx[p] for p in lgb_val_preds]))
print(f"LightGBM Val Accuracy: {lgb_val_acc:.4f}")

# ============================================================
# LOAD BEST ORDINAL NN MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()

# Validation predictions
val_preds_final = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = predict_class(logits)
        val_preds_final.extend(preds.cpu().numpy())

val_acc_final = accuracy_score(y_val_idx, val_preds_final)
print(f"\nOrdinalNN Final Val Accuracy: {val_acc_final:.4f}")

# Test predictions from OrdinalNN
test_preds_nn = []
with torch.no_grad():
    for (X_batch,) in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = predict_class(logits)
        test_preds_nn.extend(preds.cpu().numpy())

# Test predictions from LightGBM
test_preds_lgb = lgb_model.predict(test_processed)

# ============================================================
# SELECT BEST MODEL BASED ON VALIDATION PERFORMANCE
# ============================================================
if val_acc_final >= lgb_val_acc:
    print(
        f"\nUsing OrdinalNN (val_acc={val_acc_final:.4f} >= lgb_acc={lgb_val_acc:.4f})"
    )
    final_test_preds = test_preds_nn
    final_val_acc = val_acc_final
    use_nn = True
else:
    print(f"\nUsing LightGBM (val_acc={lgb_val_acc:.4f} > nn_acc={val_acc_final:.4f})")
    final_test_preds = test_preds_lgb
    final_val_acc = lgb_val_acc
    use_nn = False

# ============================================================
# CONVERT TO CLASS LABELS AND SAVE SUBMISSION
# ============================================================
if use_nn:
    test_labels = [idx_to_class[p] for p in final_test_preds]
else:
    test_labels = list(final_test_preds)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": test_ids.values.astype(int), "NObeyesdad": test_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Prediction distribution:\n{submission['NObeyesdad'].value_counts()}")

# ============================================================
# FINAL SCORE
# ============================================================
score = final_val_acc
print(f"Final Validation Score: {score}")