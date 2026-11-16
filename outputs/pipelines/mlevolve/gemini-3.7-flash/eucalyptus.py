import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize

# -------------------------------------------------------------------------
# 1. Load Data and Domain-Specific Feature Engineering
# -------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Utility"
y_raw = train_df[target_col].astype(str)

# Multi-class target label encoding
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
target_mapping = {label: idx for idx, label in enumerate(target_encoder.classes_)}
num_classes = len(target_encoder.classes_)

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()


def extract_domain_features(df):
    df = df.copy()

    # Clean numeric columns
    numeric_cols = [
        "Rep",
        "Altitude",
        "Rainfall",
        "Frosts",
        "Year",
        "PMCno",
        "DBH",
        "Ht",
        "Surv",
        "Vig",
        "Ins_res",
        "Stem_Fm",
        "Crown_Fm",
        "Brnch_Fm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical columns
    cat_cols = ["Abbrev", "Locality", "Map_Ref", "Latitude", "Sp"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": "missing", "None": "missing", "?": "missing"})
            )

    # A. Missingness Indicator Features
    morphology_cols = [
        "DBH",
        "Ht",
        "Surv",
        "Vig",
        "Ins_res",
        "Stem_Fm",
        "Crown_Fm",
        "Brnch_Fm",
    ]
    for col in morphology_cols:
        if col in df.columns:
            df[f"{col}_isnan"] = df[col].isna().astype(np.float32)

    df["total_missing_morphology"] = df[
        [f"{col}_isnan" for col in morphology_cols]
    ].sum(axis=1)

    # B. Biophysical Tree Growth & Allometric Features
    dbh_safe = df["DBH"].clip(lower=0.01)

    # Slenderness Ratio (Height / Diameter)
    df["slenderness_ratio"] = df["Ht"] / (dbh_safe + 1e-5)

    # Stem Volume Index proxy
    df["stem_volume_index"] = (df["DBH"] ** 2) * df["Ht"]

    # Basal area proxy
    df["basal_area_proxy"] = np.pi * ((df["DBH"] / 2.0) ** 2)

    # C. Morphological Form & Quality Composite Scores
    df["form_quality_mean"] = df[["Stem_Fm", "Crown_Fm", "Brnch_Fm"]].mean(axis=1)
    df["form_quality_sum"] = df[["Stem_Fm", "Crown_Fm", "Brnch_Fm"]].sum(axis=1)
    df["form_quality_std"] = (
        df[["Stem_Fm", "Crown_Fm", "Brnch_Fm"]].std(axis=1).fillna(0)
    )

    # D. Tree Vitality & Health Indices
    df["vitality_index"] = (
        (df["Surv"].fillna(0) * 0.4)
        + (df["Vig"].fillna(0) * 0.3)
        + (df["Ins_res"].fillna(0) * 0.3)
    )
    df["growth_vigor_interaction"] = df["Ht"].fillna(0) * df["Vig"].fillna(0)
    df["stem_vigor_ratio"] = df["Stem_Fm"] / (df["Vig"].clip(lower=0.01) + 1e-5)
    df["tree_compactness"] = (dbh_safe / (df["Ht"].clip(lower=0.1) + 1e-5)) * (
        df["Stem_Fm"].fillna(1.0) + 1e-5
    )
    df["form_vigor_ratio"] = df["form_quality_mean"] / (
        df["Vig"].clip(lower=0.01) + 1e-5
    )
    df["ht_dbh_product"] = df["Ht"] * dbh_safe
    df["crown_stem_ratio"] = df["Crown_Fm"] / (
        df["Stem_Fm"].clip(lower=0.01) + 1e-5
    )

    # E. Site Climate & Environmental Stress Interactions
    rainfall_safe = df["Rainfall"].clip(lower=1.0)
    df["frost_to_rain_ratio"] = df["Frosts"] / (rainfall_safe + 1e-5)
    df["altitude_frost_index"] = df["Altitude"] * (df["Frosts"] + 10.0)
    df["drought_frost_stress"] = df["Frosts"] / (rainfall_safe / 100.0 + 1.0)
    df["climate_growth_potential"] = (
        rainfall_safe / (df["Altitude"].clip(lower=1.0) + 100.0)
    ) * (10.0 / (df["Frosts"].clip(lower=0.1) + 5.0))
    df["health_to_climate_ratio"] = df["vitality_index"] / (
        df["Frosts"].clip(lower=0.1) + 5.0
    )

    # F. Stand Age Proxy
    if "Year" in df.columns:
        base_year = df["Year"].min() if df["Year"].min() > 1900 else 1980
        df["stand_age"] = df["Year"] - base_year

    return df


X_train_fe = extract_domain_features(X_train_raw)
X_test_fe = extract_domain_features(X_test_raw)

# Categorical Label Encoding
categorical_features = ["Abbrev", "Locality", "Map_Ref", "Latitude", "Sp", "PMCno"]
categorical_features = [
    col for col in categorical_features if col in X_train_fe.columns
]

for col in categorical_features:
    X_train_fe[col] = X_train_fe[col].astype(str)
    X_test_fe[col] = X_test_fe[col].astype(str)
    le = LabelEncoder()
    all_categories = list(
        set(X_train_fe[col].unique()).union(set(X_test_fe[col].unique()))
    )
    le.fit(all_categories)
    X_train_fe[col] = le.transform(X_train_fe[col])
    X_test_fe[col] = le.transform(X_test_fe[col])


def add_relative_benchmark_features(train_fold, val_fold, test_fold):
    train_out = train_fold.copy()
    val_out = val_fold.copy()
    test_out = test_fold.copy()

    benchmark_cols = [
        "Ht",
        "DBH",
        "Vig",
        "Surv",
        "Stem_Fm",
        "Crown_Fm",
        "Brnch_Fm",
        "vitality_index",
        "stem_volume_index",
        "slenderness_ratio",
    ]
    group_cols = ["Sp", "Locality"]

    for grp in group_cols:
        if grp not in train_fold.columns:
            continue
        # Frequency encoding strictly from training fold
        freq = train_fold[grp].value_counts(normalize=True)
        train_out[f"{grp}_freq"] = (
            train_fold[grp].map(freq).fillna(0).astype(np.float32)
        )
        val_out[f"{grp}_freq"] = (
            val_fold[grp].map(freq).fillna(0).astype(np.float32)
        )
        test_out[f"{grp}_freq"] = (
            test_fold[grp].map(freq).fillna(0).astype(np.float32)
        )

        for col in benchmark_cols:
            if col in train_fold.columns:
                grp_means = train_fold.groupby(grp)[col].mean()
                overall_mean = train_fold[col].mean()

                train_grp_m = train_fold[grp].map(grp_means).fillna(overall_mean)
                val_grp_m = val_fold[grp].map(grp_means).fillna(overall_mean)
                test_grp_m = test_fold[grp].map(grp_means).fillna(overall_mean)

                train_out[f"{col}_diff_{grp}_mean"] = (
                    train_fold[col] - train_grp_m
                ).astype(np.float32)
                val_out[f"{col}_diff_{grp}_mean"] = (
                    val_fold[col] - val_grp_m
                ).astype(np.float32)
                test_out[f"{col}_diff_{grp}_mean"] = (
                    test_fold[col] - test_grp_m
                ).astype(np.float32)

                train_out[f"{col}_ratio_{grp}_mean"] = (
                    train_fold[col] / (train_grp_m + 1e-5)
                ).astype(np.float32)
                val_out[f"{col}_ratio_{grp}_mean"] = (
                    val_fold[col] / (val_grp_m + 1e-5)
                ).astype(np.float32)
                test_out[f"{col}_ratio_{grp}_mean"] = (
                    test_fold[col] / (test_grp_m + 1e-5)
                ).astype(np.float32)

    # Impute missing values with training fold medians
    numeric_features = [
        c
        for c in train_out.columns
        if train_out[c].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]
    train_medians = train_out[numeric_features].median()

    train_out = train_out.fillna(train_medians)
    val_out = val_out.fillna(train_medians)
    test_out = test_out.fillna(train_medians)

    return train_out, val_out, test_out


# -------------------------------------------------------------------------
# 2. Model Architecture Definitions
# -------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        return self.head(out)


def create_neural_model_and_optimizer(
    in_features: int,
    num_classes: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.2,
):
    model = TabularResNet(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


def get_lgb_params(num_classes: int, random_state: int = 42) -> dict:
    return {
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
        "min_child_samples": 15,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": random_state,
        "verbose": -1,
        "n_jobs": -1,
    }


def get_xgb_params(num_classes: int, random_state: int = 42) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# -------------------------------------------------------------------------
# 3. Cross-Validation Training and Evaluation
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_samples = len(X_train_fe)
n_test = len(X_test_fe)
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(skf.split(X_train_fe, y))

oof_lgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_xgb = np.zeros((n_samples, num_classes), dtype=np.float32)
oof_nn = np.zeros((n_samples, num_classes), dtype=np.float32)

test_preds_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_preds_nn = np.zeros((n_test, num_classes), dtype=np.float32)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr_raw_f = X_train_fe.iloc[train_idx].copy()
    X_va_raw_f = X_train_fe.iloc[val_idx].copy()
    X_te_raw_f = X_test_fe.copy()

    X_tr_df, X_va_df, X_te_df = add_relative_benchmark_features(
        X_tr_raw_f, X_va_raw_f, X_te_raw_f
    )
    y_tr = y[train_idx]
    y_va = y[val_idx]
    num_features = X_tr_df.shape[1]

    # --- A. LightGBM Classifier ---
    lgb_train_data = lgb.Dataset(X_tr_df, label=y_tr)
    lgb_val_data = lgb.Dataset(X_va_df, label=y_va, reference=lgb_train_data)
    lgb_params_fold = get_lgb_params(
        num_classes=num_classes, random_state=42 + fold_idx
    )

    lgb_model = lgb.train(
        params=lgb_params_fold,
        train_set=lgb_train_data,
        num_boost_round=1000,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va_df)
    test_preds_lgb += lgb_model.predict(X_te_df) / n_splits

    # --- B. XGBoost Classifier ---
    xgb_params_fold = get_xgb_params(
        num_classes=num_classes, random_state=42 + fold_idx
    )
    xgb_model = xgb.XGBClassifier(
        **xgb_params_fold, n_estimators=1000, early_stopping_rounds=50
    )
    xgb_model.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)

    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_df)
    test_preds_xgb += xgb_model.predict_proba(X_te_df) / n_splits

    # --- C. Deep TabularResNet Neural Model ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_df)
    X_va_scaled = scaler.transform(X_va_df)
    X_te_scaled = scaler.transform(X_te_df)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.long),
    )
    test_ds = TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    nn_model, criterion, optimizer, scheduler = create_neural_model_and_optimizer(
        in_features=num_features,
        num_classes=num_classes,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_nn_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                val_loss += criterion(logits, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()

    val_probs_nn = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            val_probs_nn.append(probs)
    oof_nn[val_idx] = np.vstack(val_probs_nn)

    test_probs_nn_fold = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            probs = torch.softmax(nn_model(batch_x), dim=1).cpu().numpy()
            test_probs_nn_fold.append(probs)
    test_preds_nn += np.vstack(test_probs_nn_fold) / n_splits

    fold_blend = (
        0.45 * oof_lgb[val_idx] + 0.45 * oof_xgb[val_idx] + 0.10 * oof_nn[val_idx]
    )
    fold_acc = accuracy_score(y_va, fold_blend.argmax(axis=1))
    print(f"Fold {fold_idx + 1}/{n_splits} Accuracy: {fold_acc:.4f}")

# -------------------------------------------------------------------------
# 4. Ensemble Weight Optimization & Submission Generation
# -------------------------------------------------------------------------
best_acc = 0.0
best_weights = (0.45, 0.45, 0.10)

for w_lgb in np.linspace(0.0, 1.0, 21):
    for w_xgb in np.linspace(0.0, 1.0 - w_lgb, 21):
        w_nn = max(0.0, 1.0 - w_lgb - w_xgb)
        blend_oof = w_lgb * oof_lgb + w_xgb * oof_xgb + w_nn * oof_nn
        blend_preds = blend_oof.argmax(axis=1)
        acc = accuracy_score(y, blend_preds)
        if acc > best_acc:
            best_acc = acc
            best_weights = (w_lgb, w_xgb, w_nn)

w1, w2, w3 = best_weights
final_test_probs = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_nn
final_test_preds_idx = final_test_probs.argmax(axis=1)
final_test_labels = target_encoder.inverse_transform(final_test_preds_idx)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "Utility": final_test_labels}
)
submission_df.to_csv("./submission/submission.csv", index=False)

final_oof_probs = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_nn
final_score = accuracy_score(y, final_oof_probs.argmax(axis=1))

print(f"Final Validation Score: {final_score}")