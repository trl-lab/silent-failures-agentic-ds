import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import re
import hashlib
import pickle
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("DATA PROCESSING AND FEATURE ENGINEERING")
print("=" * 60)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nColumns: {list(train_df.columns)}")
print(f"\nTarget distribution:\n{train_df['classification'].value_counts()}")
print(f"\nDtypes:\n{train_df.dtypes}")
print(f"\nFirst few rows:\n{train_df.head(3)}")
print(
    f"\nMissing values (train):\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}"
)
print(f"\nMissing values (test):\n{test_df.isnull().sum()[test_df.isnull().sum() > 0]}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "classification"
id_col = "id" if "id" in train_df.columns else None

y = train_df[target_col].copy()
train_df = train_df.drop(columns=[target_col])

feature_cols = [c for c in train_df.columns if c != id_col]
print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

cat_cols = []
num_cols = []
text_cols = []

for col in feature_cols:
    if train_df[col].dtype == "object" or train_df[col].nunique() < 50:
        if train_df[col].dtype == "object":
            sample = train_df[col].dropna().head(100)
            if sample.apply(lambda x: len(str(x)) > 50).mean() > 0.3:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        else:
            num_cols.append(col)
    else:
        if train_df[col].nunique() > len(train_df) * 0.5:
            cat_cols.append(col)
        else:
            num_cols.append(col)

print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")
print(f"Text features ({len(text_cols)}): {text_cols}")

# ============================================================
# 3. CREATE TRAIN/VAL SPLIT
# ============================================================
train_combined = train_df.copy()
train_combined[target_col] = y.values
train_part, val_part = train_test_split(
    train_combined, test_size=0.2, random_state=42, stratify=train_combined[target_col]
)
y_train = train_part[target_col]
y_val = val_part[target_col]
train_part = train_part.drop(columns=[target_col])
val_part = val_part.drop(columns=[target_col])

print(f"\nTrain split: {train_part.shape}, Val split: {val_part.shape}")
print(f"Val class distribution:\n{y_val.value_counts()}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================


def engineer_features(
    df,
    cat_cols,
    num_cols,
    text_cols,
    feature_cols,
    train_part,
    fit=False,
    freq_maps=None,
    label_encoders=None,
):
    """Apply feature engineering. If fit=True, learn parameters from train_part."""
    df = df.copy()

    if freq_maps is None:
        freq_maps = {}
    if label_encoders is None:
        label_encoders = {}

    # --- 4a. Handle missing values ---
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("__MISSING__")
            else:
                df[col] = df[col].fillna(-999)

    # --- 4b. Text feature engineering ---
    for col in text_cols:
        if col not in df.columns:
            continue
        base = col.lower()

        df[f"{base}_len"] = df[col].astype(str).apply(len)
        df[f"{base}_word_count"] = df[col].astype(str).apply(lambda x: len(x.split()))
        df[f"{base}_has_digit"] = (
            df[col].astype(str).apply(lambda x: int(any(c.isdigit() for c in x)))
        )
        df[f"{base}_upper_ratio"] = (
            df[col]
            .astype(str)
            .apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        )
        df[f"{base}_special_chars"] = (
            df[col]
            .astype(str)
            .apply(lambda x: sum(1 for c in x if not c.isalnum() and c != " "))
        )
        df[f"{base}_first_word"] = (
            df[col].astype(str).apply(lambda x: x.split()[0] if x.split() else "")
        )
        df[f"{base}_last_word"] = (
            df[col].astype(str).apply(lambda x: x.split()[-1] if x.split() else "")
        )

    # --- 4c. Categorical feature encoding ---
    for col in cat_cols:
        if col not in df.columns:
            continue

        if fit:
            nunique = train_part[col].nunique()
            freq = train_part[col].value_counts(normalize=True)
            freq_maps[f"{col}_freq"] = freq
            freq_maps[f"{col}_nunique"] = nunique
            all_vals = sorted(train_part[col].astype(str).unique())
            label_encoders[col] = {v: i for i, v in enumerate(all_vals)}
            le = label_encoders[col]
        else:
            nunique = freq_maps.get(f"{col}_nunique", 0)
            freq = freq_maps[f"{col}_freq"]
            le = label_encoders[col]

        df[f"{col}_freq_enc"] = df[col].astype(str).map(freq).fillna(0)
        df[f"{col}_label"] = df[col].astype(str).map(le).fillna(-1)

        if nunique > 20:
            for h in range(3):
                df[f"{col}_hash_{h}"] = (
                    df[col]
                    .astype(str)
                    .apply(
                        lambda x, h=h: int(
                            hashlib.md5(f"{h}_{x}".encode()).hexdigest(), 16
                        )
                        % 1000
                    )
                )

    # --- 4d. Numerical feature engineering ---
    for col in num_cols:
        if col not in df.columns:
            continue
        if fit:
            positive_mask = train_part[col] > 0
            if positive_mask.sum() > len(train_part) * 0.5 and positive_mask.sum() > 0:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        elif f"{col}_log" in df.columns:
            pass
        else:
            # Check if log was created during fit
            if f"{col}_log" in train_part.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        df[f"{col}_sq"] = df[col] ** 2

    # --- 4e. Interaction features ---
    if num_cols and cat_cols:
        top_cats = cat_cols[:5]
        for cat in top_cats:
            for num in num_cols[:3]:
                if cat in df.columns and num in df.columns:
                    df[f"{cat}_x_{num}"] = df[f"{cat}_label"] * df[num]

    # --- 4f. Aggregate features ---
    if fit:
        df["_n_missing"] = train_part[feature_cols].isnull().sum(axis=1)
    else:
        df["_n_missing"] = df[feature_cols].isnull().sum(axis=1)

    # --- 4g. Encode text first/last words as categoricals ---
    for col in text_cols:
        if col not in df.columns:
            continue
        base = col.lower()
        for suffix in ["first_word", "last_word"]:
            col_name = f"{base}_{suffix}"
            if col_name in df.columns:
                if fit:
                    freq = df[col_name].value_counts(normalize=True)
                    freq_maps[f"{col_name}_freq"] = freq
                    all_vals = sorted(df[col_name].astype(str).unique())
                    label_encoders[col_name] = {v: i for i, v in enumerate(all_vals)}
                    le = label_encoders[col_name]
                else:
                    freq = freq_maps[f"{col_name}_freq"]
                    le = label_encoders[col_name]

                df[f"{col_name}_freq_enc"] = (
                    df[col_name].astype(str).map(freq).fillna(0)
                )
                df[f"{col_name}_label"] = df[col_name].astype(str).map(le).fillna(-1)

    return df, freq_maps, label_encoders


# Apply feature engineering
train_part, freq_maps, label_encoders = engineer_features(
    train_part,
    cat_cols,
    num_cols,
    text_cols,
    feature_cols,
    train_part,
    fit=True,
    freq_maps=None,
    label_encoders=None,
)
val_part, _, _ = engineer_features(
    val_part,
    cat_cols,
    num_cols,
    text_cols,
    feature_cols,
    train_part,
    fit=False,
    freq_maps=freq_maps,
    label_encoders=label_encoders,
)
test_df, _, _ = engineer_features(
    test_df,
    cat_cols,
    num_cols,
    text_cols,
    feature_cols,
    train_part,
    fit=False,
    freq_maps=freq_maps,
    label_encoders=label_encoders,
)

# ============================================================
# 5. IDENTIFY FINAL FEATURE MATRIX
# ============================================================
exclude_cols = set(feature_cols) | ({id_col} if id_col else set())
exclude_cols |= set(text_cols)

train_features = [
    c
    for c in train_part.columns
    if c not in exclude_cols and train_part[c].dtype != "object"
]
val_features = [
    c
    for c in val_part.columns
    if c not in exclude_cols and val_part[c].dtype != "object"
]
test_features = [
    c for c in test_df.columns if c not in exclude_cols and test_df[c].dtype != "object"
]

common_features = list(set(train_features) & set(val_features) & set(test_features))
common_features.sort()

print(f"\nFinal feature count: {len(common_features)}")
print(f"Sample features: {common_features[:20]}")

X_train = train_part[common_features].values.astype(np.float32)
X_val = val_part[common_features].values.astype(np.float32)
X_test = test_df[common_features].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

target_le = LabelEncoder()
y_train_enc = target_le.fit_transform(y_train)
y_val_enc = target_le.transform(y_val)

print(f"\nTarget classes: {target_le.classes_}")
print(f"Num classes: {len(target_le.classes_)}")
print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================
# 6. MODEL DESIGN
# ============================================================
print("\n" + "=" * 60)
print("MODEL DESIGN")
print("=" * 60)

n_classes = len(target_le.classes_)
n_features = X_train.shape[1]

print(f"Input features: {n_features}")
print(f"Number of classes: {n_classes}")


class TabularMLP(nn.Module):
    """Regularized MLP for tabular classification."""

    def __init__(
        self, input_dim, hidden_dims=[256, 128, 64], n_classes=10, dropout_rate=0.3
    ):
        super(TabularMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


model = TabularMLP(
    input_dim=n_features,
    hidden_dims=[256, 128, 64],
    n_classes=n_classes,
    dropout_rate=0.3,
)

label_smoothing = 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")
print(f"Optimizer: AdamW(lr=1e-3, wd=1e-4)")
print(f"Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)")

# ============================================================
# 7. DATA LOADERS
# ============================================================
X_train_t = torch.FloatTensor(X_train)
X_val_t = torch.FloatTensor(X_val)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.LongTensor(y_train_enc)
y_val_t = torch.LongTensor(y_val_enc)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

# ============================================================
# 8. TRAINING LOOP WITH EARLY STOPPING
# ============================================================
print("\n" + "=" * 60)
print("TRAINING AND EVALUATION")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")
print(f"{'Epoch':<7}{'Train Loss':<12}{'Train Acc':<12}{'Val Acc':<12}{'LR':<10}")
print("-" * 53)

best_val_acc = 0.0
best_model_state = None
patience = 20
patience_counter = 0
max_epochs = 300

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(batch_y).sum().item()
        train_total += batch_X.size(0)

    scheduler.step()

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(batch_y).sum().item()
            val_total += batch_X.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"{epoch+1:<7}{train_loss/train_total:<12.4f}{train_acc:<12.4f}{val_acc:<12.4f}{current_lr:<10.2e}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(
                f"\nEarly stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}"
            )
            break

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")

# ============================================================
# 9. LOAD BEST MODEL AND GENERATE PREDICTIONS
# ============================================================
model.load_state_dict(best_model_state)
model.to(device)
model.eval()


def predict_with_model(model, data_loader, device):
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            batch_X = batch[0].to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return np.array(predictions)


val_predictions = predict_with_model(model, val_loader, device)
test_predictions = predict_with_model(model, test_loader, device)

# ============================================================
# 10. COMPUTE VALIDATION METRIC
# ============================================================
score = accuracy_score(y_val_enc, val_predictions)
print(f"\nValidation Accuracy (official metric): {score:.6f}")
print(
    f"Val prediction distribution:\n{np.bincount(val_predictions, minlength=n_classes)}"
)
print(f"True val distribution:\n{np.bincount(y_val_enc, minlength=n_classes)}")

# ============================================================
# 11. GENERATE SUBMISSION
# ============================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

torch.save(best_model_state, "./working/best_model_state.pt")

if id_col and id_col in test_df.columns:
    test_ids = test_df[id_col].values
else:
    test_ids = np.arange(len(test_df))

test_pred_labels = target_le.inverse_transform(test_predictions)

submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "classification": test_pred_labels}
)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission_df.shape[0]} rows")
print(
    f"Submission prediction distribution:\n{submission_df['classification'].value_counts()}"
)

# ============================================================
# 12. FINAL REPORT
# ============================================================
print(f"\n{'=' * 60}")
print(f"COMPLETE")
print(f"{'=' * 60}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Final Validation Score: {score}")