import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------
# 1. PyTorch Architecture Definition (TabularResNet)
# ---------------------------------------------------------
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


# ---------------------------------------------------------
# 2. Data Processing & Feature Engineering Pipeline
# ---------------------------------------------------------
def main():
    input_dir = "./input"
    output_dir = "./working"
    submission_dir = "./submission"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Load Data
    df_train = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    df_test = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    # Identify target column
    target_col = "car" if "car" in df_train.columns else "class"

    # Ordinal mappings derived from UCI Car Evaluation domain specifications
    ordinal_mappings = {
        "buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
        "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
        "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
        "persons": {"2": 2, "4": 4, "more": 6},
        "lug_boot": {"small": 1, "med": 2, "big": 3},
        "safety": {"low": 1, "med": 2, "high": 3},
    }

    def create_features(df):
        data = df.copy()

        # Apply ordinal mappings
        for col, mapping in ordinal_mappings.items():
            if col in data.columns:
                data[f"{col}_ord"] = data[col].map(mapping).fillna(0).astype(float)

        # Domain-specific Interaction Features
        data["total_cost"] = data["buying_ord"] + data["maint_ord"]
        data["cost_diff"] = data["buying_ord"] - data["maint_ord"]
        data["cost_ratio"] = data["buying_ord"] / (data["maint_ord"] + 1e-5)

        data["capacity_score"] = data["persons_ord"] * data["lug_boot_ord"]
        data["safety_capacity"] = data["safety_ord"] * data["persons_ord"]
        data["safety_lug"] = data["safety_ord"] * data["lug_boot_ord"]

        # Non-linear global utility metric
        data["overall_utility"] = (
            data["safety_ord"] * data["persons_ord"] * data["lug_boot_ord"]
        ) / (data["total_cost"] + 1e-5)

        # Critical decision boundary indicator rules
        data["is_safety_low"] = (data["safety_ord"] == 1).astype(float)
        data["is_persons_2"] = (data["persons_ord"] == 2).astype(float)
        data["reject_rule_trigger"] = (
            (data["safety_ord"] == 1) | (data["persons_ord"] == 2)
        ).astype(float)
        data["high_safety_high_cap"] = (
            (data["safety_ord"] >= 2) & (data["persons_ord"] >= 4)
        ).astype(float)

        return data

    df_train_feat = create_features(df_train)
    df_test_feat = create_features(df_test)

    # Target Mapping
    unique_targets = sorted(df_train_feat[target_col].unique())
    target_to_label = {target: idx for idx, target in enumerate(unique_targets)}
    label_to_target = {idx: target for target, idx in target_to_label.items()}

    df_train_feat["target"] = df_train_feat[target_col].map(target_to_label)

    # Feature column selection
    feature_cols = [
        c for c in df_train_feat.columns if c not in [target_col, "target", "id"]
    ]
    numeric_cols = [
        c
        for c in feature_cols
        if df_train_feat[c].dtype in [np.float64, np.int64, float, int]
    ]

    # Train / Validation Split
    X = df_train_feat[numeric_cols].copy()
    y = df_train_feat["target"].values
    X_test = df_test_feat[numeric_cols].copy()

    if "id" in df_test.columns:
        test_ids = df_test["id"].values
    else:
        test_ids = np.arange(len(df_test))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Frequency Encodings derived strictly on training split post-split
    for col in list(ordinal_mappings.keys()):
        freq_map = df_train_feat.loc[X_train.index, col].value_counts(normalize=True).to_dict()
        X_train[f"{col}_freq"] = df_train_feat.loc[X_train.index, col].map(freq_map).fillna(0)
        X_val[f"{col}_freq"] = df_train_feat.loc[X_val.index, col].map(freq_map).fillna(0)
        X_test[f"{col}_freq"] = df_test_feat[col].map(freq_map).fillna(0)

    # Standard Scaling fitted strictly on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    in_features = X_train_scaled.shape[1]
    num_classes = len(unique_targets)

    # ---------------------------------------------------------
    # 3. Model Training - PyTorch TabularResNet
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled), torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled), torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    nn_model = TabularResNet(in_features=in_features, num_classes=num_classes).to(
        device
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_tabular_resnet.pt")

    epochs = 150
    patience = 25
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        nn_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = nn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        nn_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = nn_model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss /= total
        val_acc = correct / total
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(nn_model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load Best Neural Model Weights
    nn_model.load_state_dict(torch.load(best_model_path))
    nn_model.eval()

    # ---------------------------------------------------------
    # 4. Model Training - LightGBM Classifier
    # ---------------------------------------------------------
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_estimators=1000,
        verbose=-1,
    )

    lgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # ---------------------------------------------------------
    # 5. Model Training - XGBoost Classifier
    # ---------------------------------------------------------
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method="hist",
        n_estimators=1000,
        early_stopping_rounds=50,
    )

    xgb_model.fit(
        X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False
    )

    # ---------------------------------------------------------
    # 6. Validation Evaluation (Soft Voting Ensemble)
    # ---------------------------------------------------------
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        nn_val_logits = nn_model(val_tensor)
        nn_val_probs = torch.softmax(nn_val_logits, dim=1).cpu().numpy()

    lgb_val_probs = lgb_model.predict_proba(X_val_scaled)
    xgb_val_probs = xgb_model.predict_proba(X_val_scaled)

    ensemble_val_probs = (nn_val_probs + lgb_val_probs + xgb_val_probs) / 3.0
    ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)

    val_accuracy = accuracy_score(y_val, ensemble_val_preds)

    # ---------------------------------------------------------
    # 7. Test Inference & Submission File Generation
    # ---------------------------------------------------------
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        nn_test_logits = nn_model(test_tensor)
        nn_test_probs = torch.softmax(nn_test_logits, dim=1).cpu().numpy()

    lgb_test_probs = lgb_model.predict_proba(X_test_scaled)
    xgb_test_probs = xgb_model.predict_proba(X_test_scaled)

    ensemble_test_probs = (nn_test_probs + lgb_test_probs + xgb_test_probs) / 3.0
    ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)

    test_pred_labels = [label_to_target[p] for p in ensemble_test_preds]

    submission_df = pd.DataFrame({"id": test_ids, "car": test_pred_labels})
    submission_df.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)

    # Final line required for score parser
    print(f"Final Validation Score: {val_accuracy}")


if __name__ == "__main__":
    main()