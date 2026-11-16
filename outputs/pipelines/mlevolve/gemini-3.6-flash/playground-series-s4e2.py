import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


# ==========================================
# 1. Feature Engineering Functions
# ==========================================
def create_anthropometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-specific physical, biometric, and interaction features."""
    df = df.copy()

    height_m = df["Height"]
    weight_kg = df["Weight"]

    # Body Mass Index (BMI) & Log Scaling
    df["BMI"] = weight_kg / (height_m**2)
    df["BMI_log"] = np.log(df["BMI"])

    # Standard WHO BMI Cutoff Distances
    for threshold in [18.5, 25.0, 30.0, 35.0, 40.0]:
        df[f"BMI_diff_{str(threshold).replace('.', '_')}"] = df["BMI"] - threshold

    # Body Surface Area (BSA)
    df["BSA_Mosteller"] = np.sqrt((df["Height"] * 100 * df["Weight"]) / 3600.0)
    df["BSA_DuBois"] = (
        0.007184 * ((df["Height"] * 100) ** 0.725) * (df["Weight"] ** 0.425)
    )

    # Ideal Body Weight, Ponderal Index & Tri-Ponderal Mass Index (TMI)
    height_inches = df["Height"] * 39.3701
    gender_factor = df["Gender"].apply(lambda g: 50.0 if g == "Male" else 45.5)
    df["Ideal_Weight_Devine"] = gender_factor + 2.3 * np.maximum(
        0.0, height_inches - 60.0
    )
    df["Weight_to_Ideal_Ratio"] = df["Weight"] / (df["Ideal_Weight_Devine"] + 1e-5)
    df["Ponderal_Index"] = weight_kg / (height_m**3)
    df["TMI"] = weight_kg / (height_m**3)

    # Interactions
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Weight_Age"] = df["Weight"] * df["Age"]

    # Explicit interactions between BMI and behavioral habits
    df["BMI_FAF"] = df["BMI"] * df["FAF"]
    df["BMI_CH2O"] = df["BMI"] * df["CH2O"]
    df["BMI_TUE"] = df["BMI"] * df["TUE"]
    df["BMI_FAF_ratio"] = df["BMI"] / (df["FAF"] + 1e-5)
    df["BMI_CH2O_ratio"] = df["BMI"] / (df["CH2O"] + 1e-5)
    df["BMI_TUE_ratio"] = df["BMI"] / (df["TUE"] + 1e-5)

    return df


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer metabolic, activity, and dietary ratio features."""
    df = df.copy()

    df["CH2O_per_Weight"] = df["CH2O"] / (df["Weight"] + 1e-5)
    df["CH2O_per_Height"] = df["CH2O"] / (df["Height"] + 1e-5)

    df["Activity_Score"] = df["FAF"] - df["TUE"]
    df["Activity_per_Age"] = df["FAF"] / (df["Age"] + 1e-5)
    df["Tech_per_Age"] = df["TUE"] / (df["Age"] + 1e-5)

    df["Vegetable_per_Meal"] = df["FCVC"] / (df["NCP"] + 1e-5)

    df["Metabolic_Balance_Index"] = (df["FAF"] * df["CH2O"] * df["FCVC"]) / (
        df["NCP"] * (df["TUE"] + 0.1) + 1e-5
    )

    return df


def create_synthetic_decomposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract integer and fractional components from synthetic continuous variables."""
    df = df.copy()

    pseudo_discrete_cols = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in pseudo_discrete_cols:
        df[f"{col}_round"] = np.round(df[col])
        df[f"{col}_frac"] = df[col] - np.floor(df[col])
        df[f"{col}_is_int"] = (np.abs(df[col] - np.round(df[col])) < 0.01).astype(int)

    return df


def encode_categorical_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, orig_df: pd.DataFrame
) -> tuple:
    """Apply deterministic ordinal mappings and one-hot encoding across splits without leakage."""
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()

    ordinal_maps = {
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    }

    binary_maps = {
        "Gender": {"Female": 0, "Male": 1},
        "family_history_with_overweight": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
    }

    for df in [train, test, orig]:
        for col, mapping in ordinal_maps.items():
            df[f"{col}_ordinal"] = df[col].map(mapping).fillna(0).astype(int)
        for col, mapping in binary_maps.items():
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    for df in [train, test, orig]:
        df["Caloric_Surplus_Score"] = df["FAVC"] + df["CAEC_ordinal"] - df["SCC"]

    raw_cat_cols = ["CAEC", "CALC"]
    train.drop(columns=raw_cat_cols, inplace=True, errors="ignore")
    test.drop(columns=raw_cat_cols, inplace=True, errors="ignore")
    orig.drop(columns=raw_cat_cols, inplace=True, errors="ignore")

    mtrans_cols = ["MTRANS"]
    all_dfs = pd.concat([train, test, orig], axis=0, ignore_index=True)
    all_dfs = pd.get_dummies(all_dfs, columns=mtrans_cols, drop_first=False)

    # Ensure integer boolean types
    bool_cols = all_dfs.select_dtypes(include=["bool"]).columns
    all_dfs[bool_cols] = all_dfs[bool_cols].astype(int)

    train_len = len(train)
    test_len = len(test)

    train_encoded = all_dfs.iloc[:train_len].copy()
    test_encoded = all_dfs.iloc[train_len : train_len + test_len].copy()
    orig_encoded = all_dfs.iloc[train_len + test_len :].copy()

    return train_encoded, test_encoded, orig_encoded


# ==========================================
# 2. PyTorch Deep Tabular ResNet Model
# ==========================================
class TabularResBlock(nn.Module):
    """Residual Block with Layer Normalization, SiLU activation, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet for Multi-Class Classification."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


# ==========================================
# 3. Main Pipeline Execution
# ==========================================
def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Raw Data
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")
    orig_raw = pd.read_csv("./input/original/train.csv")

    target_col = "NObeyesdad"

    # Encode Target
    label_encoder = LabelEncoder()
    train_raw[target_col] = label_encoder.fit_transform(train_raw[target_col])
    orig_raw[target_col] = label_encoder.transform(orig_raw[target_col])

    class_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    num_classes = len(label_encoder.classes_)

    # 2. Apply Feature Engineering Transformations
    datasets = [train_raw, test_raw, orig_raw]
    processed_dfs = []
    for df in datasets:
        df_feat = create_anthropometric_features(df)
        df_feat = create_behavioral_features(df_feat)
        df_feat = create_synthetic_decomposition_features(df_feat)
        processed_dfs.append(df_feat)

    train_df, test_df, orig_df = encode_categorical_features(
        processed_dfs[0], processed_dfs[1], processed_dfs[2]
    )

    # 3. Setup CV Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        train_df.loc[val_idx, "fold"] = fold

    ignore_cols = ["id", target_col, "fold"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    n_samples_train = len(train_df)
    n_samples_test = len(test_df)
    n_folds = 5

    # Prediction arrays
    oof_lgb = np.zeros((n_samples_train, num_classes))
    oof_xgb = np.zeros((n_samples_train, num_classes))
    oof_cat = np.zeros((n_samples_train, num_classes))
    oof_nn = np.zeros((n_samples_train, num_classes))

    test_lgb = np.zeros((n_samples_test, num_classes))
    test_xgb = np.zeros((n_samples_test, num_classes))
    test_cat = np.zeros((n_samples_test, num_classes))
    test_nn = np.zeros((n_samples_test, num_classes))

    # 4. Stratified 5-Fold Cross Validation Loop
    for fold in range(n_folds):
        train_fold_mask = train_df["fold"] != fold
        val_fold_mask = train_df["fold"] == fold

        # Train set = Synthetic fold train + Original ground truth dataset
        X_tr_df = (
            pd.concat(
                [train_df.loc[train_fold_mask, feature_cols], orig_df[feature_cols]],
                axis=0,
            )
            .reset_index(drop=True)
            .astype(np.float32)
        )

        y_tr_series = pd.concat(
            [train_df.loc[train_fold_mask, target_col], orig_df[target_col]],
            axis=0,
        ).reset_index(drop=True)

        X_va_df = (
            train_df.loc[val_fold_mask, feature_cols]
            .reset_index(drop=True)
            .astype(np.float32)
        )
        y_va_series = train_df.loc[val_fold_mask, target_col].reset_index(drop=True)

        val_indices = train_df.index[val_fold_mask].values

        # --- LightGBM ---
        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            n_estimators=2000,
            learning_rate=0.025,
            num_leaves=45,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_samples=15,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(
            X_tr_df,
            y_tr_series,
            eval_set=[(X_va_df, y_va_series)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgb[val_indices] = lgbm.predict_proba(X_va_df)
        test_lgb += (
            lgbm.predict_proba(test_df[feature_cols].astype(np.float32)) / n_folds
        )

        # --- XGBoost ---
        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=2000,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=2,
            reg_alpha=0.2,
            reg_lambda=2.0,
            gamma=0.05,
            early_stopping_rounds=50,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(
            X_tr_df,
            y_tr_series,
            eval_set=[(X_va_df, y_va_series)],
            verbose=False,
        )
        oof_xgb[val_indices] = xgb.predict_proba(X_va_df)
        test_xgb += (
            xgb.predict_proba(test_df[feature_cols].astype(np.float32)) / n_folds
        )

        # --- CatBoost ---
        cat = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_strength=0.1,
            bagging_temperature=0.2,
            early_stopping_rounds=50,
            random_seed=42 + fold,
            thread_count=-1,
            verbose=0,
        )
        cat.fit(X_tr_df, y_tr_series, eval_set=(X_va_df, y_va_series))
        oof_cat[val_indices] = cat.predict_proba(X_va_df)
        test_cat += (
            cat.predict_proba(test_df[feature_cols].astype(np.float32)) / n_folds
        )

        # --- PyTorch ResNet ---
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_df.values)
        X_va_scaled = scaler.transform(X_va_df.values)
        X_te_scaled = scaler.transform(X_test_raw)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr_series.values, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va_series.values, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=512, shuffle=False, num_workers=2
        )

        nn_model = TabularResNet(
            in_features=len(feature_cols),
            num_classes=num_classes,
            hidden_dim=256,
            num_blocks=3,
            dropout=0.2,
        ).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
        optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

        best_val_loss = float("inf")
        best_nn_state = None

        for epoch in range(35):
            nn_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = nn_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = nn_model(bx)
                    val_loss += criterion(out, by).item() * len(by)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_state = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

        nn_model.load_state_dict(best_nn_state)
        nn_model.to(device)
        nn_model.eval()

        with torch.no_grad():
            va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
            oof_nn[val_indices] = (
                torch.softmax(nn_model(va_tensor), dim=1).cpu().numpy()
            )

            te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
            test_nn += torch.softmax(nn_model(te_tensor), dim=1).cpu().numpy() / n_folds

    # 5. Out-of-fold Evaluation & Metric-Aligned Probability Ensembling
    y_true = train_df[target_col].values
    oofs = [oof_lgb, oof_xgb, oof_cat, oof_nn]
    test_preds = [test_lgb, test_xgb, test_cat, test_nn]
    num_models = len(oofs)

    # Individual model OOF Accuracies
    model_names = ["LightGBM", "XGBoost", "CatBoost", "NeuralNet"]
    for name, oof in zip(model_names, oofs):
        acc = accuracy_score(y_true, np.argmax(oof, axis=1))
        print(f"OOF Accuracy [{name}]: {acc:.6f}")

    def transform_probs(probs, temp):
        """Apply temperature calibration while ensuring proper probability normalization."""
        t = max(temp, 0.1)
        p = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return p / np.sum(p, axis=1, keepdims=True)

    def compute_ensemble_probs(weights, temps, prob_list):
        """Soft probability averaging with temperature normalization."""
        w = np.maximum(weights, 0.0)
        w_sum = np.sum(w)
        if w_sum == 0:
            w = np.ones(len(w)) / len(w)
        else:
            w = w / w_sum

        blend = np.zeros_like(prob_list[0])
        for i in range(len(prob_list)):
            p_cal = transform_probs(prob_list[i], temps[i])
            blend += w[i] * p_cal
        return blend

    def objective(params):
        weights = params[:num_models]
        temps = params[num_models:]
        blend = compute_ensemble_probs(weights, temps, oofs)
        acc = accuracy_score(y_true, np.argmax(blend, axis=1))
        return -acc

    # Optimization setup: initial random search followed by Powell local optimization
    best_acc = -1.0
    best_params = None

    # Randomized initial parameter search across simplex and temperature space
    np.random.seed(42)
    for _ in range(500):
        w = np.random.dirichlet(np.ones(num_models))
        t = np.random.uniform(0.7, 1.3, size=num_models)
        p = np.concatenate([w, t])
        score = -objective(p)
        if score > best_acc:
            best_acc = score
            best_params = p

    # Fine-tune parameters directly optimizing Out-of-Fold Accuracy
    opt_res = minimize(
        objective,
        best_params,
        method="Powell",
        options={"maxiter": 500, "ftol": 1e-5},
    )

    if -opt_res.fun > best_acc:
        best_params = opt_res.x

    raw_w = np.maximum(best_params[:num_models], 0.0)
    best_weights = raw_w / (np.sum(raw_w) + 1e-12)
    best_temps = [max(t, 0.1) for t in best_params[num_models:]]

    oof_blend = compute_ensemble_probs(best_weights, best_temps, oofs)
    final_score = accuracy_score(y_true, np.argmax(oof_blend, axis=1))

    # 6. Test Predictions & Submission Generation
    test_blend = compute_ensemble_probs(best_weights, best_temps, test_preds)
    test_pred_indices = np.argmax(test_blend, axis=1)
    test_pred_labels = [class_mapping[idx] for idx in test_pred_indices]

    submission_df = pd.DataFrame(
        {"id": test_df["id"].astype(int), "NObeyesdad": test_pred_labels}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Optimized Ensemble Weights: {best_weights}")
    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
