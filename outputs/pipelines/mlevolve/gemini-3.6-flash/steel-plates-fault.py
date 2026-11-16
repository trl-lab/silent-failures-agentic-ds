import os
import json
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# Set random seeds for reproducible execution
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Directory Setup & Data Loading
# ---------------------------------------------------------
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "target"
y_raw = train_df[target_col].values

# Label encode multi-class target
le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# Preserve ID column if present, otherwise construct 0-based integer index
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))


# Dynamic column fetcher for variant naming schemes
def get_col(df, primary, fallback):
    if primary in df.columns:
        return df[primary]
    elif fallback in df.columns:
        return df[fallback]
    else:
        raise KeyError(f"Neither '{primary}' nor '{fallback}' found in DataFrame.")


# ---------------------------------------------------------
# 2. Domain Feature Engineering Pipeline
# ---------------------------------------------------------
def engineer_features(df_input):
    df = df_input.copy()

    # Dynamic attribute extraction
    x_min = get_col(df, "V1", "X_Minimum")
    x_max = get_col(df, "V2", "X_Maximum")
    y_min = get_col(df, "V3", "Y_Minimum")
    y_max = get_col(df, "V4", "Y_Maximum")

    pixels_area = get_col(df, "V5", "Pixels_Areas")
    x_perim = get_col(df, "V6", "X_Perimeter")
    y_perim = get_col(df, "V7", "Y_Perimeter")

    sum_lum = get_col(df, "V8", "Sum_of_Luminosity")
    min_lum = get_col(df, "V9", "Minimum_of_Luminosity")
    max_lum = get_col(df, "V10", "Maximum_of_Luminosity")

    conveyer_len = get_col(df, "V11", "Length_of_Conveyer")
    steel_a300 = get_col(df, "V12", "TypeOfSteel_A300")
    steel_a400 = get_col(df, "V13", "TypeOfSteel_A400")
    steel_thick = get_col(df, "V14", "Steel_Plate_Thickness")

    edges_x = get_col(df, "V19", "Edges_X_Index")
    edges_y = get_col(df, "V20", "Edges_Y_Index")
    outside_x = get_col(df, "V18", "Outside_X_Index")
    outside_glob = get_col(df, "V21", "Outside_Global_Index")

    log_area = get_col(df, "V22", "LogOfAreas")
    log_x = get_col(df, "V23", "Log_X_Index")
    log_y = get_col(df, "V24", "Log_Y_Index")

    edges_index = get_col(df, "V15", "Edges_Index")
    empty_index = get_col(df, "V16", "Empty_Index")
    square_index = get_col(df, "V17", "Square_Index")
    orientation_index = get_col(df, "V25", "Orientation_Index")
    sigmoid_areas = get_col(df, "V27", "Sigmoid_of_Areas")

    # Geometric & Bounding Box Features
    x_length = np.maximum(x_max - x_min, 0.0)
    y_length = np.maximum(y_max - y_min, 0.0)
    bbox_area = x_length * y_length
    perim_total = x_perim + y_perim

    df["fe_x_length"] = x_length
    df["fe_y_length"] = y_length
    df["fe_bbox_area"] = bbox_area
    df["fe_area_fill_ratio"] = pixels_area / (bbox_area + 1e-5)
    df["fe_aspect_ratio"] = x_length / (y_length + 1e-5)
    df["fe_log_aspect_ratio"] = np.log1p(x_length) - np.log1p(y_length)
    df["fe_perim_total"] = perim_total
    df["fe_perim_ratio"] = x_perim / (y_perim + 1e-5)

    # Conveyer-normalized spatial fault coordinates
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    df["fe_x_center"] = x_center
    df["fe_y_center"] = y_center
    df["fe_y_min_norm"] = y_min / (conveyer_len + 1e-5)
    df["fe_y_max_norm"] = y_max / (conveyer_len + 1e-5)
    df["fe_y_center_norm"] = y_center / (conveyer_len + 1e-5)

    # Shape factors & Isoperimetric quotient & Compactness
    df["fe_isoperimetric_quotient"] = (4.0 * np.pi * pixels_area) / (
        perim_total**2 + 1e-5
    )
    df["fe_compactness"] = (perim_total**2) / (pixels_area + 1e-5)
    df["fe_shape_factor_x"] = pixels_area / (x_perim**2 + 1e-5)
    df["fe_shape_factor_y"] = pixels_area / (y_perim**2 + 1e-5)
    df["fe_convexity_x"] = x_length / (x_perim + 1e-5)
    df["fe_convexity_y"] = y_length / (y_perim + 1e-5)

    # Photometric Features & Contrast Density
    lum_range = max_lum - min_lum
    lum_mean = sum_lum / (pixels_area + 1e-5)
    df["fe_lum_range"] = lum_range
    df["fe_lum_mean"] = lum_mean
    df["fe_lum_contrast"] = lum_range / (lum_mean + 1e-5)
    df["fe_lum_contrast_density"] = lum_range / (lum_mean * pixels_area + 1e-5)
    df["fe_lum_ratio"] = min_lum / (max_lum + 1e-5)
    df["fe_lum_per_conveyer"] = sum_lum / (conveyer_len + 1e-5)

    # Material & Grade Interactions
    steel_type = steel_a300 * 1 + steel_a400 * 2
    df["fe_steel_type"] = steel_type
    df["fe_thick_per_area"] = steel_thick / (pixels_area + 1e-5)
    df["fe_thick_per_conveyer"] = steel_thick / (conveyer_len + 1e-5)
    df["fe_thick_x_lum"] = steel_thick * lum_mean

    # Log Domain Residuals & Ratios
    df["fe_log_area_residual"] = log_area - np.log10(np.maximum(pixels_area, 0.0) + 1.0)
    df["fe_log_area_ratio"] = log_area / (np.log10(np.maximum(pixels_area, 0.0) + 1.0) + 1e-5)
    df["fe_log_x_residual"] = log_x - np.log10(x_length + 1.0)
    df["fe_log_y_residual"] = log_y - np.log10(y_length + 1.0)

    # Edge and Position Aggregations
    df["fe_edges_product"] = edges_x * edges_y
    df["fe_edges_sum"] = edges_x + edges_y
    df["fe_edges_ratio"] = edges_x / (edges_y + 1e-5)
    df["fe_outside_sum"] = outside_x + outside_glob

    # Geometric density ratios & shape index interactions
    df["fe_log_x_y_ratio"] = log_x / (log_y + 1e-5)
    df["fe_sigmoid_log_area_prod"] = sigmoid_areas * log_area
    df["fe_empty_square_prod"] = empty_index * square_index
    df["fe_edges_orientation_prod"] = edges_index * orientation_index

    # Clean target and id columns from feature matrix
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


X_train_df = engineer_features(train_df)
X_test_df = engineer_features(test_df)


# ---------------------------------------------------------
# 3. Model Definitions (PyTorch SE-ResNet & GBDT Builders)
# ---------------------------------------------------------
class TabularSEModule(nn.Module):
    def __init__(self, in_features: int, reduction: int = 4):
        super().__init__()
        reduced_features = max(8, in_features // reduction)
        self.fc1 = nn.Linear(in_features, reduced_features, bias=False)
        self.relu = nn.SiLU()
        self.fc2 = nn.Linear(reduced_features, in_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc1(x)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)
        return x * weights


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = TabularSEModule(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.norm2(out)
        out = self.fc2(out)
        out = self.se(out)
        return residual + out


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.res_blocks:
            out = block(out)
        return self.head(out)


class FocalLossMultiClass(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            focal_weight = focal_weight * self.alpha[targets]

        return (focal_weight * ce_loss).mean()


def get_neural_model_and_optimizer(
    input_dim: int,
    num_classes: int = 7,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    class_weights: np.ndarray = None,
):
    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.25,
    )
    alpha_tensor = None
    if class_weights is not None:
        alpha_tensor = torch.tensor(class_weights, dtype=torch.float32)

    criterion = FocalLossMultiClass(gamma=2.0, alpha=alpha_tensor, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    return model, criterion, optimizer, scheduler


def get_gbdt_models(num_classes: int = 7, random_state: int = 42):
    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
        "verbose": -1,
        "random_state": random_state,
        "n_estimators": 1200,
    }

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_estimators=1200,
        random_state=random_state,
        tree_method="hist",
        early_stopping_rounds=50,
    )

    cat_model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=5.0,
        iterations=1200,
        random_seed=random_state,
        verbose=0,
        early_stopping_rounds=50,
    )

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    hgb_model = HistGradientBoostingClassifier(
        max_depth=4,
        max_leaf_nodes=15,
        learning_rate=0.03,
        l2_regularization=2.0,
        max_iter=500,
        random_state=random_state,
    )

    return {
        "lgb_params": lgb_params,
        "xgb_model": xgb_model,
        "cat_model": cat_model,
        "rf_model": rf_model,
        "hgb_model": hgb_model,
    }


# ---------------------------------------------------------
# 4. Stratified 5-Fold Ensemble Training Loop
# ---------------------------------------------------------
n_train = len(X_train_df)
n_test = len(X_test_df)

oof_lgb = np.zeros((n_train, num_classes))
oof_xgb = np.zeros((n_train, num_classes))
oof_cat = np.zeros((n_train, num_classes))
oof_et = np.zeros((n_train, num_classes))
oof_rf = np.zeros((n_train, num_classes))
oof_hgb = np.zeros((n_train, num_classes))
oof_nn = np.zeros((n_train, num_classes))

test_lgb = np.zeros((n_test, num_classes))
test_xgb = np.zeros((n_test, num_classes))
test_cat = np.zeros((n_test, num_classes))
test_et = np.zeros((n_test, num_classes))
test_rf = np.zeros((n_test, num_classes))
test_hgb = np.zeros((n_test, num_classes))
test_nn = np.zeros((n_test, num_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y)):
    X_tr_fold = X_train_df.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va_fold = X_train_df.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te_fold = X_test_df.copy()

    # Leak-free fold group aggregations per steel type
    num_cols_to_grp = ["fe_lum_mean", "fe_aspect_ratio", "fe_isoperimetric_quotient"]
    for col in num_cols_to_grp:
        mean_map = X_tr_fold.groupby("fe_steel_type")[col].mean().to_dict()
        std_map = X_tr_fold.groupby("fe_steel_type")[col].std().fillna(0).to_dict()

        global_mean = X_tr_fold[col].mean()
        global_std = X_tr_fold[col].std()

        for df_curr in [X_tr_fold, X_va_fold, X_te_fold]:
            df_curr[f"fe_grp_{col}_mean"] = (
                df_curr["fe_steel_type"].map(mean_map).fillna(global_mean)
            )
            df_curr[f"fe_grp_{col}_std"] = (
                df_curr["fe_steel_type"].map(std_map).fillna(global_std)
            )

    # Fold-fitted KMeans cluster prototype distances
    scaler_km = StandardScaler()
    X_tr_km_sc = scaler_km.fit_transform(X_tr_fold)
    X_va_km_sc = scaler_km.transform(X_va_fold)
    X_te_km_sc = scaler_km.transform(X_te_fold)

    kmeans = KMeans(n_clusters=8, random_state=SEED, n_init=10)
    km_tr_dist = kmeans.fit_transform(X_tr_km_sc)
    km_va_dist = kmeans.transform(X_va_km_sc)
    km_te_dist = kmeans.transform(X_te_km_sc)

    for c_idx in range(8):
        X_tr_fold[f"fe_km_dist_{c_idx}"] = km_tr_dist[:, c_idx]
        X_va_fold[f"fe_km_dist_{c_idx}"] = km_va_dist[:, c_idx]
        X_te_fold[f"fe_km_dist_{c_idx}"] = km_te_dist[:, c_idx]

    X_tr = X_tr_fold.values
    X_va = X_va_fold.values
    X_te = X_te_fold.values

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_tr), y=y_tr
    )

    gbdt_configs = get_gbdt_models(num_classes=num_classes, random_state=SEED + fold)

    # A. LightGBM
    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)
    gbm = lgb.train(
        gbdt_configs["lgb_params"],
        lgb_tr,
        valid_sets=[lgb_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = gbm.predict(X_va, num_iteration=gbm.best_iteration)
    test_lgb += gbm.predict(X_te, num_iteration=gbm.best_iteration) / skf.n_splits

    # B. XGBoost
    xgb_clf = xgb.XGBClassifier(**gbdt_configs["xgb_model"].get_params())
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)
    test_xgb += xgb_clf.predict_proba(X_te) / skf.n_splits

    # C. CatBoost
    cat_clf = cb.CatBoostClassifier(**gbdt_configs["cat_model"].get_params())
    cat_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
    oof_cat[val_idx] = cat_clf.predict_proba(X_va)
    test_cat += cat_clf.predict_proba(X_te) / skf.n_splits

    # D. ExtraTrees Classifier
    et_clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    et_clf.fit(X_tr, y_tr)
    oof_et[val_idx] = et_clf.predict_proba(X_va)
    test_et += et_clf.predict_proba(X_te) / skf.n_splits

    # E. RandomForest Classifier
    rf_clf = RandomForestClassifier(**gbdt_configs["rf_model"].get_params())
    rf_clf.fit(X_tr, y_tr)
    oof_rf[val_idx] = rf_clf.predict_proba(X_va)
    test_rf += rf_clf.predict_proba(X_te) / skf.n_splits

    # F. HistGradientBoosting Classifier
    hgb_clf = HistGradientBoostingClassifier(**gbdt_configs["hgb_model"].get_params())
    hgb_clf.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_clf.predict_proba(X_va)
    test_hgb += hgb_clf.predict_proba(X_te) / skf.n_splits

    # G. PyTorch Tabular SE-ResNet
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)
    )
    test_ds = TensorDataset(torch.tensor(X_te_sc, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    nn_model, criterion, optimizer, scheduler = get_neural_model_and_optimizer(
        input_dim=X_tr_sc.shape[1],
        num_classes=num_classes,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=60,
        class_weights=class_weights,
    )
    nn_model = nn_model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_cnt = 15, 0

    for epoch in range(60):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = nn_model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    nn_model.eval()

    val_nn_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            probs = torch.softmax(nn_model(bx), dim=1)
            val_nn_list.append(probs.cpu().numpy())
    oof_nn[val_idx] = np.vstack(val_nn_list)

    test_nn_list = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(device)
            probs = torch.softmax(nn_model(bx), dim=1)
            test_nn_list.append(probs.cpu().numpy())
    test_nn += np.vstack(test_nn_list) / skf.n_splits

    fold_acc_lgb = accuracy_score(y_va, np.argmax(oof_lgb[val_idx], axis=1))
    print(f"Fold {fold + 1} LightGBM Accuracy: {fold_acc_lgb:.5f}")


# ---------------------------------------------------------
# 5. Stacking Meta-Classifier & Final Predictions
# ---------------------------------------------------------
X_meta_train = np.hstack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf, oof_hgb, oof_nn])
X_meta_test = np.hstack([test_lgb, test_xgb, test_cat, test_et, test_rf, test_hgb, test_nn])

meta_clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=SEED)
meta_clf.fit(X_meta_train, y)

final_oof_preds = meta_clf.predict(X_meta_train)
score = accuracy_score(y, final_oof_preds)

final_test_preds = meta_clf.predict(X_meta_test)
final_test_labels = le.inverse_transform(final_test_preds)

sub_df = pd.DataFrame({"id": test_ids, "target": final_test_labels})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")