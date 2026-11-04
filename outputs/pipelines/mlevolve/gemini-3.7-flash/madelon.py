import copy
import json
import math
import os
import warnings
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import minimize
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================================================================
# 1. Paths and Data Loading
# =========================================================================
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "Class"
if target_col not in train_df.columns:
    raise KeyError(f"Target column '{target_col}' not found in train.parquet")

y_raw = train_df[target_col]
feature_cols = [col for col in train_df.columns if col != target_col]
test_feature_cols = [
    col for col in test_df.columns if col != "id" and col != target_col
]

# Map target classes to continuous indices (0, 1) and record inverse mapping
unique_classes = sorted(y_raw.unique())
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
y = y_raw.map(class_to_idx).to_numpy()

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

# =========================================================================
# 2. Informative Subspace Selection
# =========================================================================
# Shadow-feature noise thresholding combined with multi-seed ExtraTrees, RandomForest, and MI consensus
np.random.seed(42)
X_shadow = X_train_raw.copy()
for col in X_shadow.columns:
    X_shadow[col] = np.random.permutation(X_shadow[col].values)
X_shadow.columns = [f"shadow_{col}" for col in X_shadow.columns]
X_augmented = pd.concat([X_train_raw, X_shadow], axis=1)

n_real_cols = len(feature_cols)
et_scores = np.zeros(n_real_cols)
rf_scores = np.zeros(n_real_cols)
seeds = [42, 101, 2024]

for s in seeds:
    et_temp = ExtraTreesClassifier(n_estimators=200, max_depth=None, random_state=s, n_jobs=-1)
    et_temp.fit(X_augmented, y)
    imp_et = et_temp.feature_importances_
    real_et = imp_et[:n_real_cols]
    shadow_et_thresh = np.percentile(imp_et[n_real_cols:], 99)
    et_scores += (real_et - shadow_et_thresh)

    rf_temp = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=s, n_jobs=-1)
    rf_temp.fit(X_augmented, y)
    imp_rf = rf_temp.feature_importances_
    real_rf = imp_rf[:n_real_cols]
    shadow_rf_thresh = np.percentile(imp_rf[n_real_cols:], 99)
    rf_scores += (real_rf - shadow_rf_thresh)

mi_scores = mutual_info_classif(X_train_raw, y, random_state=42, n_neighbors=7)

score_et_norm = (et_scores - et_scores.mean()) / (et_scores.std() + 1e-9)
score_rf_norm = (rf_scores - rf_scores.mean()) / (rf_scores.std() + 1e-9)
score_mi_norm = (mi_scores - mi_scores.mean()) / (mi_scores.std() + 1e-9)
combined_score = score_et_norm + score_rf_norm + score_mi_norm

ranked_indices = np.argsort(combined_score)[::-1]
top_20_indices = ranked_indices[:20]
top_25_indices = ranked_indices[:25]

selected_features_20 = [feature_cols[i] for i in top_20_indices]
selected_features_25 = [feature_cols[i] for i in top_25_indices]


# =========================================================================
# 3. Geometric & Non-Linear Feature Engineering Pipeline
# =========================================================================
def engineer_features(
    df_features, top20_names, top25_names, is_train=True, fit_objects=None
):
    if fit_objects is None:
        fit_objects = {}

    df_out = pd.DataFrame(index=df_features.index)

    X_20 = df_features[top20_names].values

    for col in top20_names:
        df_out[f"raw_{col}"] = df_features[col]

    if is_train:
        scaler = RobustScaler()
        X_20_scaled = scaler.fit_transform(X_20)
        fit_objects["scaler_20"] = scaler
    else:
        scaler = fit_objects["scaler_20"]
        X_20_scaled = scaler.transform(X_20)

    # Pairwise interactions on the top 10 features
    top10_names = top20_names[:10]
    for i in range(len(top10_names)):
        col_i = top10_names[i]
        val_i = df_features[col_i].values
        df_out[f"poly_sq_{col_i}"] = val_i**2
        for j in range(i + 1, len(top10_names)):
            col_j = top10_names[j]
            val_j = df_features[col_j].values
            df_out[f"inter_prod_{col_i}_{col_j}"] = val_i * val_j
            df_out[f"inter_diff_{col_i}_{col_j}"] = np.abs(val_i - val_j)

    # Gaussian Mixture Model (32 components) soft cluster log-likelihoods
    if is_train:
        gmm_32 = GaussianMixture(n_components=32, covariance_type="full", random_state=42, n_init=3)
        gmm_32.fit(X_20_scaled)
        gmm_probs = gmm_32.predict_proba(X_20_scaled)
        gmm_scores = gmm_32.score_samples(X_20_scaled)
        fit_objects["gmm_32"] = gmm_32
    else:
        gmm_32 = fit_objects["gmm_32"]
        gmm_probs = gmm_32.predict_proba(X_20_scaled)
        gmm_scores = gmm_32.score_samples(X_20_scaled)

    for c in range(32):
        df_out[f"gmm32_prob_{c}"] = gmm_probs[:, c]
    df_out["gmm32_log_score"] = gmm_scores

    # Multi-scale K-Nearest Neighbors distance graphs (k=15, 31, 63)
    if is_train:
        nn_graph = NearestNeighbors(n_neighbors=64, algorithm="ball_tree")
        nn_graph.fit(X_20_scaled)
        fit_objects["nn_graph"] = nn_graph
        nn_dists, _ = nn_graph.kneighbors(X_20_scaled, n_neighbors=64)
        dist_15 = nn_dists[:, 15]
        dist_31 = nn_dists[:, 31]
        dist_63 = nn_dists[:, 63]
        dist_mean = np.mean(nn_dists[:, 1:64], axis=1)
    else:
        nn_graph = fit_objects["nn_graph"]
        nn_dists, _ = nn_graph.kneighbors(X_20_scaled, n_neighbors=63)
        dist_15 = nn_dists[:, 14]
        dist_31 = nn_dists[:, 30]
        dist_63 = nn_dists[:, 62]
        dist_mean = np.mean(nn_dists[:, 0:63], axis=1)

    df_out["knn_dist_15"] = dist_15
    df_out["knn_dist_31"] = dist_31
    df_out["knn_dist_63"] = dist_63
    df_out["knn_dist_mean"] = dist_mean

    # KMeans cluster vertex representations
    if is_train:
        kmeans_32 = KMeans(n_clusters=32, random_state=42, n_init=10)
        dist_32 = kmeans_32.fit_transform(X_20_scaled)
        fit_objects["kmeans_32"] = kmeans_32

        kmeans_16 = KMeans(n_clusters=16, random_state=42, n_init=10)
        dist_16 = kmeans_16.fit_transform(X_20_scaled)
        fit_objects["kmeans_16"] = kmeans_16
    else:
        dist_32 = fit_objects["kmeans_32"].transform(X_20_scaled)
        dist_16 = fit_objects["kmeans_16"].transform(X_20_scaled)

    for c in range(32):
        df_out[f"kmeans32_dist_{c}"] = dist_32[:, c]
    df_out["kmeans32_min_dist"] = np.min(dist_32, axis=1)
    df_out["kmeans32_mean_dist"] = np.mean(dist_32, axis=1)

    for c in range(16):
        df_out[f"kmeans16_dist_{c}"] = dist_16[:, c]
    df_out["kmeans16_min_dist"] = np.min(dist_16, axis=1)

    # Random Fourier RBF Kernel Approximation
    if is_train:
        rbf_sampler = RBFSampler(gamma=0.05, n_components=30, random_state=42)
        rbf_feats = rbf_sampler.fit_transform(X_20_scaled)
        fit_objects["rbf_sampler"] = rbf_sampler
    else:
        rbf_feats = fit_objects["rbf_sampler"].transform(X_20_scaled)

    for r in range(30):
        df_out[f"rbf_feat_{r}"] = rbf_feats[:, r]

    # PCA manifold decomposition
    if is_train:
        pca = PCA(n_components=5, random_state=42)
        pca_feats = pca.fit_transform(X_20_scaled)
        fit_objects["pca"] = pca
    else:
        pca_feats = fit_objects["pca"].transform(X_20_scaled)

    for p in range(5):
        df_out[f"pca_{p}"] = pca_feats[:, p]

    # Subspace summary statistics
    df_out["subspace_mean"] = np.mean(X_20, axis=1)
    df_out["subspace_std"] = np.std(X_20, axis=1)
    df_out["subspace_max"] = np.max(X_20, axis=1)
    df_out["subspace_min"] = np.min(X_20, axis=1)
    df_out["subspace_l2"] = np.linalg.norm(X_20_scaled, axis=1)

    return df_out, fit_objects


# 10-Fold Stratified Split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(train_df), dtype=int)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y)):
    fold_assignments[val_idx] = fold

y_all = y
n_samples = len(X_train_raw)
n_test = len(X_test_raw)
n_folds = 10


# =========================================================================
# 4. Neural Network & Tabular Model Architectures
# =========================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MadelonResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def build_lightgbm_model(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgboost_model(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


def build_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )


def build_knn_15_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)


def build_knn_31_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1)


def build_svc_rbf_model(random_state: int = 42) -> SVC:
    return SVC(
        C=2.5,
        gamma="scale",
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )


def build_svc_rbf_auto_model(random_state: int = 42) -> SVC:
    return SVC(
        C=5.0,
        gamma="auto",
        kernel="rbf",
        probability=True,
        random_state=random_state,
    )


def build_extratrees_model(random_state: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def build_random_forest_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pytorch_resnet(
    X_tr, y_tr, X_va, y_val, input_dim, epochs=100, batch_size=64, random_state=42
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_x_t = torch.tensor(X_va_sc, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = MadelonResNet(
        input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.2
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t)
            val_loss = criterion(
                val_logits, torch.tensor(y_val, dtype=torch.float32).to(device)
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(val_x_t)).cpu().numpy()

    return model, scaler, val_probs


# =========================================================================
# 5. Stratified Cross-Validation & Ensemble Inference
# =========================================================================
model_names = [
    "knn_15",
    "knn_31",
    "svc_rbf",
    "svc_rbf_auto",
    "lgbm",
    "xgb",
    "catboost",
    "extratrees",
    "random_forest",
    "pytorch_resnet",
]

oof_preds = {m: np.zeros(n_samples) for m in model_names}
test_preds = {m: np.zeros(n_test) for m in model_names}

for fold_idx in range(n_folds):
    train_mask = fold_assignments != fold_idx
    val_mask = fold_assignments == fold_idx

    X_tr_raw = X_train_raw.iloc[train_mask].copy()
    X_va_raw = X_train_raw.iloc[val_mask].copy()
    y_train_fold, y_val_fold = y_all[train_mask], y_all[val_mask]

    train_eng_fold, fit_objs_fold = engineer_features(
        X_tr_raw, selected_features_20, selected_features_25, is_train=True
    )
    val_eng_fold, _ = engineer_features(
        X_va_raw, selected_features_20, selected_features_25, is_train=False, fit_objects=fit_objs_fold
    )
    test_eng_fold, _ = engineer_features(
        X_test_raw, selected_features_20, selected_features_25, is_train=False, fit_objects=fit_objs_fold
    )

    X_train_fold = train_eng_fold.values
    X_val_fold = val_eng_fold.values
    X_test_mat = test_eng_fold.values
    n_features = X_train_fold.shape[1]

    # LightGBM
    lgb_model = build_lightgbm_model(random_state=42 + fold_idx)
    lgb_model.fit(X_train_fold, y_train_fold)
    oof_preds["lgbm"][val_mask] = lgb_model.predict_proba(X_val_fold)[:, 1]
    test_preds["lgbm"] += lgb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # XGBoost
    xgb_model = build_xgboost_model(random_state=42 + fold_idx)
    xgb_model.fit(X_train_fold, y_train_fold)
    oof_preds["xgb"][val_mask] = xgb_model.predict_proba(X_val_fold)[:, 1]
    test_preds["xgb"] += xgb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # CatBoost
    cb_model = build_catboost_model(random_state=42 + fold_idx)
    cb_model.fit(X_train_fold, y_train_fold)
    oof_preds["catboost"][val_mask] = cb_model.predict_proba(X_val_fold)[:, 1]
    test_preds["catboost"] += cb_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # ExtraTrees
    et_model = build_extratrees_model(random_state=42 + fold_idx)
    et_model.fit(X_train_fold, y_train_fold)
    oof_preds["extratrees"][val_mask] = et_model.predict_proba(X_val_fold)[:, 1]
    test_preds["extratrees"] += et_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # Random Forest
    rf_model = build_random_forest_model(random_state=42 + fold_idx)
    rf_model.fit(X_train_fold, y_train_fold)
    oof_preds["random_forest"][val_mask] = rf_model.predict_proba(X_val_fold)[:, 1]
    test_preds["random_forest"] += rf_model.predict_proba(X_test_mat)[:, 1] / n_folds

    # RBF Support Vector Classifier
    svc_scaler = StandardScaler()
    X_tr_svc = svc_scaler.fit_transform(X_train_fold)
    X_va_svc = svc_scaler.transform(X_val_fold)
    X_te_svc = svc_scaler.transform(X_test_mat)

    # Distance-weighted KNN 15
    knn15_model = build_knn_15_model()
    knn15_model.fit(X_tr_svc, y_train_fold)
    oof_preds["knn_15"][val_mask] = knn15_model.predict_proba(X_va_svc)[:, 1]
    test_preds["knn_15"] += knn15_model.predict_proba(X_te_svc)[:, 1] / n_folds

    # Distance-weighted KNN 31
    knn31_model = build_knn_31_model()
    knn31_model.fit(X_tr_svc, y_train_fold)
    oof_preds["knn_31"][val_mask] = knn31_model.predict_proba(X_va_svc)[:, 1]
    test_preds["knn_31"] += knn31_model.predict_proba(X_te_svc)[:, 1] / n_folds

    # Multi-scale RBF SVC (scale)
    svc_model = build_svc_rbf_model(random_state=42 + fold_idx)
    svc_model.fit(X_tr_svc, y_train_fold)
    oof_preds["svc_rbf"][val_mask] = svc_model.predict_proba(X_va_svc)[:, 1]
    test_preds["svc_rbf"] += svc_model.predict_proba(X_te_svc)[:, 1] / n_folds

    # Multi-scale RBF SVC (auto)
    svc_auto_model = build_svc_rbf_auto_model(random_state=42 + fold_idx)
    svc_auto_model.fit(X_tr_svc, y_train_fold)
    oof_preds["svc_rbf_auto"][val_mask] = svc_auto_model.predict_proba(X_va_svc)[:, 1]
    test_preds["svc_rbf_auto"] += svc_auto_model.predict_proba(X_te_svc)[:, 1] / n_folds

    # Deep Tabular Residual Network
    nn_model, nn_scaler, nn_val_probs = train_pytorch_resnet(
        X_train_fold,
        y_train_fold,
        X_val_fold,
        y_val_fold,
        input_dim=n_features,
        epochs=100,
        batch_size=64,
        random_state=42 + fold_idx,
    )
    oof_preds["pytorch_resnet"][val_mask] = nn_val_probs

    X_te_nn = torch.tensor(nn_scaler.transform(X_test_mat), dtype=torch.float32).to(
        device
    )
    with torch.no_grad():
        test_preds["pytorch_resnet"] += (
            torch.sigmoid(nn_model(X_te_nn)).cpu().numpy() / n_folds
        )

# SLSQP constrained optimization for optimal ensemble weights
num_models = len(model_names)
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])


def ensemble_loss(weights):
    weights = np.array(weights)
    pred = np.clip(np.dot(oof_matrix, weights), 1e-15, 1.0 - 1e-15)
    return -np.mean(y_all * np.log(pred) + (1.0 - y_all) * np.log(1.0 - pred))


initial_weights = np.ones(num_models) / num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9},
)

optimal_weights = opt_res.x
optimal_weights = np.maximum(optimal_weights, 0.0)
optimal_weights /= np.sum(optimal_weights)

oof_ensemble = np.dot(oof_matrix, optimal_weights)
test_ensemble = np.dot(test_matrix, optimal_weights)

# Threshold search to maximize accuracy on OOF predictions
best_threshold = 0.5
best_accuracy = accuracy_score(y_all, (oof_ensemble >= 0.5).astype(int))

for thresh in np.linspace(0.35, 0.65, 61):
    current_acc = accuracy_score(y_all, (oof_ensemble >= thresh).astype(int))
    if current_acc > best_accuracy:
        best_accuracy = current_acc
        best_threshold = thresh

# =========================================================================
# 6. Submission File Generation
# =========================================================================
final_test_binary = (test_ensemble >= best_threshold).astype(int)
final_test_classes = [idx_to_class[idx] for idx in final_test_binary]

submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), target_col: final_test_classes}
)
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {best_accuracy}")