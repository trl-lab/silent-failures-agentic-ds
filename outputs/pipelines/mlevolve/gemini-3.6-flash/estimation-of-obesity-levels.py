import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# ==============================================================================
# 1. Feature Engineering & Preprocessing
# ==============================================================================
def load_data(input_dir="./input"):
    """Load train and test parquet files from input directory."""
    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df


def engineer_features(df):
    """
    Engineers physiological, metabolic, and behavioral features tailored to
    obesity classification.
    """
    df = df.copy()

    # 1. Physical & Physiological Features
    height_m = df["Height"]
    weight_kg = df["Weight"]
    height_cm = height_m * 100.0

    # Body Mass Index (BMI)
    df["BMI"] = weight_kg / (height_m**2)

    # Body Surface Area (BSA) - Mosteller formula
    df["BSA"] = np.sqrt((height_cm * weight_kg) / 3600.0)

    # Height-to-Weight ratios
    df["Weight_to_Height"] = weight_kg / height_m
    df["Weight_to_Age"] = weight_kg / (df["Age"] + 1e-5)
    df["Height_to_Age"] = height_m / (df["Age"] + 1e-5)

    # Basal Metabolic Rate (BMR) Proxy (Mifflin-St Jeor formula approximation)
    is_male = (df["Gender"] == "Male").astype(float)
    df["BMR_Proxy"] = (
        (10.0 * weight_kg)
        + (6.25 * height_cm)
        - (5.0 * df["Age"])
        + (5.0 * is_male)
        - (161.0 * (1.0 - is_male))
    )

    # Hydration relative to body weight
    df["Hydration_per_kg"] = df["CH2O"] / (weight_kg + 1e-5)

    # 2. Activity & Sedentary Features
    df["Activity_to_Screen_Ratio"] = (df["FAF"] + 0.1) / (df["TUE"] + 1.0)
    df["Energy_Expenditure_Proxy"] = df["BMR_Proxy"] * (1.2 + 0.1 * df["FAF"]) - (
        30.0 * df["TUE"]
    )

    # 3. Ordinal Categorical Mappings
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}

    df["CAEC_ordinal"] = df["CAEC"].map(caec_map).fillna(0).astype(float)
    df["CALC_ordinal"] = df["CALC"].map(calc_map).fillna(0).astype(float)

    # Binary Encodings
    binary_cols = {
        "Gender": {"Female": 0, "Male": 1},
        "family_history_with_overweight": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
    }

    for col, mapping in binary_cols.items():
        if col in df.columns:
            df[f"{col}_binary"] = df[col].map(mapping).fillna(0).astype(float)

    # Transportation Physicality Mapping
    mtrans_physicality = {
        "Walking": 3.0,
        "Bike": 3.0,
        "Public_Transportation": 1.0,
        "Motorbike": 0.5,
        "Automobile": 0.0,
    }
    df["MTRANS_physicality"] = df["MTRANS"].map(mtrans_physicality).fillna(1.0)

    # Calorie Intake vs Activity Balance Index
    df["Calorie_Balance_Index"] = (
        (df["FAVC_binary"] + df["CAEC_ordinal"] + df["CALC_ordinal"])
        - (df["FAF"] * 0.5)
        + (df["FCVC"] * 0.2)
    )

    return df


def prepare_datasets():
    print("Loading raw dataset...")
    train_df, test_df = load_data()

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    target_col = "NObeyesdad"
    assert (
        target_col in train_df.columns
    ), f"Target column '{target_col}' not found in train set!"

    y_raw = train_df[target_col].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    label_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}

    train_features_raw = train_df.drop(columns=[target_col])
    if "id" in train_features_raw.columns:
        train_features_raw = train_features_raw.drop(columns=["id"])

    test_features_raw = (
        test_df.drop(columns=["id"]) if "id" in test_df.columns else test_df.copy()
    )

    print("Applying feature engineering pipeline...")
    train_engineered = engineer_features(train_features_raw)
    test_engineered = engineer_features(test_features_raw)

    categorical_cols = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
    ]

    numeric_cols = [c for c in train_engineered.columns if c not in categorical_cols]

    combined_df = pd.concat([train_engineered, test_engineered], keys=["train", "test"])
    combined_encoded = pd.get_dummies(
        combined_df, columns=categorical_cols, drop_first=False
    )

    X_train_full = combined_encoded.xs("train").copy()
    X_test_full = combined_encoded.xs("test").copy()

    X_train_full = X_train_full.astype(np.float32)
    X_test_full = X_test_full.astype(np.float32)

    print("Creating stratified split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test_full.copy()

    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_full[numeric_cols])

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        test_ids,
        label_mapping,
    )


# ==============================================================================
# 2. PyTorch Tabular Neural Network Architecture & Model Configurations
# ==============================================================================
class TabularSEGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.se_gate = TabularSEGate(hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = self.se_gate(out)
        out = self.activation(out + residual)
        return out


class TabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        logits = self.classifier(out)
        return logits


def get_lgbm_config(num_classes: int = 7) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_config(num_classes: int = 7) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# ==============================================================================
# 3. Model Training, Ensemble & Evaluation
# ==============================================================================
def main():
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        test_ids,
        label_mapping,
    ) = prepare_datasets()

    num_classes = len(label_mapping)
    in_features = X_train.shape[1]

    # --- 1. LightGBM ---
    print("\n--- Training LightGBM Model ---")
    lgb_params = get_lgbm_config(num_classes=num_classes)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_probs_lgb = lgb_model.predict(X_val)
    test_probs_lgb = lgb_model.predict(X_test)
    acc_lgb = accuracy_score(y_val, np.argmax(val_probs_lgb, axis=1))
    print(f"LightGBM Validation Accuracy: {acc_lgb:.4f}")

    # --- 2. XGBoost ---
    print("\n--- Training XGBoost Model ---")
    xgb_params = get_xgb_config(num_classes=num_classes)
    xgb_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs_xgb = xgb_clf.predict_proba(X_val)
    test_probs_xgb = xgb_clf.predict_proba(X_test)
    acc_xgb = accuracy_score(y_val, np.argmax(val_probs_xgb, axis=1))
    print(f"XGBoost Validation Accuracy: {acc_xgb:.4f}")

    # --- 3. PyTorch TabularResNet ---
    print("\n--- Training PyTorch TabularResNet ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = TabularResNet(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        nn_model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    best_nn_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, 101):
        nn_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        scheduler.step()
        train_loss /= len(X_train)

        nn_model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_x)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()

        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = nn_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping PyTorch TabularResNet at epoch {epoch}")
                break

    if best_nn_weights is not None:
        nn_model.load_state_dict(best_nn_weights)

    nn_model.eval()
    val_probs_nn_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            val_probs_nn_list.append(probs)
    val_probs_nn = np.vstack(val_probs_nn_list)

    test_probs_nn_list = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            test_probs_nn_list.append(probs)
    test_probs_nn = np.vstack(test_probs_nn_list)

    acc_nn = accuracy_score(y_val, np.argmax(val_probs_nn, axis=1))
    print(f"PyTorch TabularResNet Validation Accuracy: {acc_nn:.4f}")

    # --- 4. Ensemble Probability Blending ---
    print("\n--- Optimizing Soft-Ensemble Blending Weights ---")
    best_acc = 0.0
    best_weights = (0.33, 0.33, 0.34)

    for w_lgb in np.linspace(0.0, 1.0, 21):
        for w_xgb in np.linspace(0.0, 1.0 - w_lgb, 21):
            w_nn = round(1.0 - w_lgb - w_xgb, 4)
            if w_nn < -1e-5:
                continue
            w_nn = max(0.0, w_nn)

            blend_val = (
                w_lgb * val_probs_lgb + w_xgb * val_probs_xgb + w_nn * val_probs_nn
            )
            val_preds = np.argmax(blend_val, axis=1)
            score = accuracy_score(y_val, val_preds)

            if score > best_acc:
                best_acc = score
                best_weights = (w_lgb, w_xgb, w_nn)

    print(
        f"Optimal Blending Weights (LGB, XGB, NN): {best_weights} -> Validation Accuracy: {best_acc:.4f}"
    )

    blend_test = (
        best_weights[0] * test_probs_lgb
        + best_weights[1] * test_probs_xgb
        + best_weights[2] * test_probs_nn
    )
    test_preds_numeric = np.argmax(blend_test, axis=1)
    test_preds_string = [label_mapping[idx] for idx in test_preds_numeric]

    submission_path = os.path.join(submission_dir, "submission.csv")
    sub_df = pd.DataFrame({"id": test_ids, "NObeyesdad": test_preds_string})
    sub_df.to_csv(submission_path, index=False)
    print(
        f"Submission successfully saved to {submission_path} with {len(sub_df)} rows."
    )

    print(f"Final Validation Score: {best_acc}")


if __name__ == "__main__":
    main()
