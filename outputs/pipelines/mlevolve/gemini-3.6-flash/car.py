import copy
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


def set_seed(seed=42):
    """Set random seeds for reproducibility across libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Load train and test parquet files from input directory."""
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def define_mappings():
    """Define domain-specific ordinal mappings reflecting monotonic hierarchy."""
    ordinal_mappings = {
        "buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
        "persons": {"2": 0, "4": 1, "more": 2},
        "lug_boot": {"small": 0, "med": 1, "big": 2},
        "safety": {"low": 0, "med": 1, "high": 2},
    }

    target_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}

    return ordinal_mappings, target_mapping


def generate_engineered_features(df, ordinal_maps):
    """
    Construct domain-driven hierarchical concept features based on DEX decision structure:
    - PRICE: combination of buying and maint
    - TECH / COMFORT: combination of doors, persons, lug_boot, safety
    - OVERALL UTILITY: trade-offs between price and technical/safety characteristics
    """
    feats = pd.DataFrame(index=df.index)

    if "id" in df.columns:
        feats["id"] = df["id"]

    # 1. Base Ordinal Features
    for col, mapping in ordinal_maps.items():
        feats[f"ord_{col}"] = df[col].astype(str).map(mapping).astype(int)

    b = feats["ord_buying"]
    m = feats["ord_maint"]
    d = feats["ord_doors"]
    p = feats["ord_persons"]
    l = feats["ord_lug_boot"]
    s = feats["ord_safety"]

    # 2. PRICE Concept Features
    feats["price_sum"] = b + m
    feats["price_prod"] = b * m
    feats["price_ratio"] = (b + 1.0) / (m + 1.0)
    feats["affordability_index"] = (3 - b) + (3 - m)

    # 3. COMFORT Concept Features
    feats["comfort_capacity"] = p + l
    feats["capacity_volume"] = (d + 1) * (p + 1) * (l + 1)
    feats["doors_per_person"] = (d + 1.0) / (p + 1.0)

    # 4. TECH & SAFETY Features
    feats["tech_score"] = feats["comfort_capacity"] + s
    feats["safety_person_prod"] = s * p
    feats["safety_lug_prod"] = s * l

    # 5. OVERALL UTILITY / DECISION Surrogates
    feats["utility_score"] = feats["affordability_index"] + feats["tech_score"]
    feats["safety_to_price_ratio"] = (s + 1.0) / (feats["price_sum"] + 1.0)
    feats["value_for_money"] = (feats["tech_score"] + 1.0) / (feats["price_sum"] + 1.0)

    # 6. Categorical Combination Strings
    feats["cat_buying_maint"] = df["buying"].astype(str) + "_" + df["maint"].astype(str)
    feats["cat_persons_safety"] = (
        df["persons"].astype(str) + "_" + df["safety"].astype(str)
    )
    feats["cat_lug_safety"] = (
        df["lug_boot"].astype(str) + "_" + df["safety"].astype(str)
    )

    return feats


class TabularResBlock(nn.Module):
    """Residual Block tailored for tabular data with Layer Normalization and Dropout."""

    def __init__(self, features, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features, features),
            nn.LayerNorm(features),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular Residual Architecture for multi-class classification."""

    def __init__(
        self, input_dim, num_classes=4, hidden_dim=128, num_blocks=2, dropout=0.2
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout_rate=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.classifier_head(h)
        return logits


def main():
    set_seed(42)
    os.makedirs("./working", exist_ok=True)

    # Load Raw Datasets
    raw_train, raw_test = load_data()
    ordinal_maps, target_map = define_mappings()
    reverse_target_map = {v: k for k, v in target_map.items()}

    if "id" in raw_test.columns:
        test_ids = raw_test["id"].values
    else:
        test_ids = np.arange(len(raw_test))

    # Split raw train into stratified train/val sets (Leakage Prevention)
    target_col = "class"
    X_raw = raw_train.drop(columns=[target_col])
    y_raw = raw_train[target_col].map(target_map).values

    X_tr_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

    # Generate domain engineered features
    feat_train = generate_engineered_features(X_tr_raw, ordinal_maps)
    feat_val = generate_engineered_features(X_val_raw, ordinal_maps)
    feat_test = generate_engineered_features(raw_test, ordinal_maps)

    # Apply Frequency Encoding fitted strictly on TRAIN
    combo_cols = ["cat_buying_maint", "cat_persons_safety", "cat_lug_safety"]
    for col in combo_cols:
        freq_map = feat_train[col].value_counts(normalize=True).to_dict()

        feat_train[f"{col}_freq"] = feat_train[col].map(freq_map).fillna(0.0)
        feat_val[f"{col}_freq"] = feat_val[col].map(freq_map).fillna(0.0)
        feat_test[f"{col}_freq"] = feat_test[col].map(freq_map).fillna(0.0)

        feat_train.drop(columns=[col], inplace=True)
        feat_val.drop(columns=[col], inplace=True)
        feat_test.drop(columns=[col], inplace=True)

    # One-Hot Encoding for raw categorical features
    cat_cols = list(ordinal_maps.keys())
    ohe_train = pd.get_dummies(X_tr_raw[cat_cols], drop_first=False, dtype=float)
    ohe_val = pd.get_dummies(X_val_raw[cat_cols], drop_first=False, dtype=float)
    ohe_test = pd.get_dummies(raw_test[cat_cols], drop_first=False, dtype=float)

    ohe_val = ohe_val.reindex(columns=ohe_train.columns, fill_value=0.0)
    ohe_test = ohe_test.reindex(columns=ohe_train.columns, fill_value=0.0)

    # Concatenate features
    X_train_full = pd.concat(
        [feat_train.reset_index(drop=True), ohe_train.reset_index(drop=True)], axis=1
    )
    X_val_full = pd.concat(
        [feat_val.reset_index(drop=True), ohe_val.reset_index(drop=True)], axis=1
    )
    X_test_full = pd.concat(
        [feat_test.reset_index(drop=True), ohe_test.reset_index(drop=True)], axis=1
    )

    feature_cols = [c for c in X_train_full.columns if c != "id"]

    X_train_feat = X_train_full[feature_cols].copy()
    X_val_feat = X_val_full[feature_cols].copy()
    X_test_feat = X_test_full[feature_cols].copy()

    # Scaling numeric features strictly using Scaler fitted on Train
    scaler = StandardScaler()
    X_train_scaled = X_train_feat.copy()
    X_val_scaled = X_val_feat.copy()
    X_test_scaled = X_test_feat.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train_feat[feature_cols])
    X_val_scaled[feature_cols] = scaler.transform(X_val_feat[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test_feat[feature_cols])

    num_classes = len(target_map)
    input_dim = len(feature_cols)

    # 1. LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train_feat,
        y_train,
        eval_set=[(X_val_feat, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_val_prob = lgb_model.predict_proba(X_val_feat)
    lgb_test_prob = lgb_model.predict_proba(X_test_feat)

    # 2. XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_model.fit(
        X_train_feat,
        y_train,
        eval_set=[(X_val_feat, y_val)],
        verbose=False,
    )
    xgb_val_prob = xgb_model.predict_proba(X_val_feat)
    xgb_test_prob = xgb_model.predict_proba(X_test_feat)

    # 3. Extra Trees Classifier
    ext_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    ext_model.fit(X_train_feat, y_train)
    ext_val_prob = ext_model.predict_proba(X_val_feat)
    ext_test_prob = ext_model.predict_proba(X_test_feat)

    # 4. PyTorch TabularResNet
    torch_model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout=0.2,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(torch_model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_tr_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    best_val_acc = 0.0
    best_model_weights = None
    patience, patience_counter = 20, 0

    for epoch in range(1, 101):
        torch_model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = torch_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_dataset)

        torch_model.eval()
        with torch.no_grad():
            val_logits = torch_model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean().item()

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(torch_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_weights is not None:
        torch_model.load_state_dict(best_model_weights)

    torch_model.eval()
    with torch.no_grad():
        torch_val_prob = torch.softmax(torch_model(X_val_tensor), dim=1).cpu().numpy()
        torch_test_prob = torch.softmax(torch_model(X_test_tensor), dim=1).cpu().numpy()

    # Soft Voting Ensemble
    ensemble_val_prob = (
        lgb_val_prob + xgb_val_prob + ext_val_prob + torch_val_prob
    ) / 4.0
    ensemble_val_preds = np.argmax(ensemble_val_prob, axis=1)
    final_val_accuracy = accuracy_score(y_val, ensemble_val_preds)

    # Compute Test Predictions and Save Submission
    ensemble_test_prob = (
        lgb_test_prob + xgb_test_prob + ext_test_prob + torch_test_prob
    ) / 4.0
    test_pred_indices = np.argmax(ensemble_test_prob, axis=1)
    test_pred_labels = [reverse_target_map[idx] for idx in test_pred_indices]

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub_df = pd.DataFrame({"id": test_ids, "class": test_pred_labels})
    sub_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_val_accuracy}")


if __name__ == "__main__":
    main()